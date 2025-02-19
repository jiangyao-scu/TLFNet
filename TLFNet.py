import cv2
import torch
from timm.models.layers import trunc_normal_
from torch import nn
import torch.nn.functional as F
from swin.transformer import SwinTransformerBackbone, TransformerBlock
from pvt.pvtv2 import pvt_v2_b2
from einops import rearrange
import os
import math
import numpy as np

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, H*4, W*4, self.output_dim)
        x= self.norm(x)

        x = rearrange(x, 'b (h p1) (w p2) c-> b c (h p1) (w p2)', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B, self.output_dim, H*4, W*4)
        return x

def PatchToImage(feature):   #(b,h*w,c)
    b, l, c = feature.shape
    h = int(math.sqrt(l))
    feature = feature.permute(0,2,1).view(b, c, h, h)
    return feature

def ImageToPatch(feature):   #(b,c,h,w)
    feature = feature.flatten(-2)                     #(b,c,h*w)
    feature = feature.permute(0, 2, 1)                #(b,h*w,c)
    return feature

class ScoreModule(nn.Module):
    def __init__(self, channels, image_size=None):
        super(ScoreModule, self).__init__()
        # 1x1,1 conv
        ##加3×3的卷积
        d = 1
        self.extra_model = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=d),
                                         nn.ReLU(),
                                         nn.Conv2d(channels, channels, 3, padding=d),
                                         nn.ReLU(),
                                         nn.Conv2d(channels, channels, 3, padding=d),
                                         nn.ReLU())
        self.conv_1 = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1)

        self.image_size = image_size

    def forward(self, x):
        # for i in range(3):
        x = self.extra_model(x)
        x = self.conv_1(x)
        if self.image_size != None:
            pred = F.interpolate(input=x, size=self.image_size, mode='bilinear', align_corners=True)
        else:
            pred = x
        # print("predict shape:", pred.shape)
        return pred

class Conv3(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv3, self).__init__()
        # 1x1,1 conv
        ##加3×3的卷积
        d = 1
        self.extra_model = nn.Sequential(nn.Conv2d(in_channel, out_channel, 3, padding=d),
                                         nn.ReLU(),
                                         nn.Conv2d(out_channel, out_channel, 3, padding=d),
                                         nn.ReLU(),
                                         nn.Conv2d(out_channel, out_channel, 3, padding=d))


    def forward(self, x):
        # for i in range(3):
        x = self.extra_model(x)
        pred = x
        return pred

class PGFormer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.intra_slice_attention = TransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.guided_feature_fusion = TransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.cross_slice_attention = TransformerBlock(dim, num_heads, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, drop_path, act_layer, norm_layer)
        self.down_channel_pixel_features = nn.Linear(12 * self.dim, self.dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, rgb_features, fs_features):  #(B,H*W,C)
        transformer_input_rgb = rgb_features
        transformer_input_fs = fs_features
        transformer_input_fs = transformer_input_fs.permute(1, 0, 2)               #(H*W,B,C)

        pixellevelfeatures = self.cross_slice_attention(transformer_input_fs, transformer_input_fs)            #(H*W,B,C)

        pixellevelfeatures = pixellevelfeatures.flatten(-2)         #(H*W,B*C)
        pixellevelfeatures = self.relu(self.down_channel_pixel_features(pixellevelfeatures))  #(H*W,C)
        pixellevelfeatures = pixellevelfeatures.unsqueeze(0)                         #(1,H*W,C)

        transformer_output_fs = self.intra_slice_attention(pixellevelfeatures, pixellevelfeatures)
        transformer_output = self.guided_feature_fusion(transformer_input_rgb,
                                                        torch.cat((transformer_input_rgb, transformer_output_fs), dim=1))

        return transformer_output

def DistanceWeight(map):
    device = map.device
    weight = torch.sigmoid(map).cpu().detach().numpy().copy()[0][0]
    weight = 255 * weight
    _, weight = cv2.threshold(weight, 128, 255, cv2.THRESH_BINARY)
    weight = np.uint8(weight)
    weight = cv2.distanceTransform(weight, cv2.DIST_L2, 3)
    weight = (weight - np.min(weight)) / (np.max(weight) - np.min(weight) + 0.000001)
    weight = torch.from_numpy(weight)
    weight = weight.unsqueeze(0).unsqueeze(0)
    weight = weight.to(device)
    return weight

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, in_dim, out_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        # self.dim = dim
        self.expand = nn.Linear(in_dim, 4 * out_dim, bias=False)
        self.norm = norm_layer(out_dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

class TLFNet(nn.Module):
    def __init__(self, backbone_type='swin'):
        super(TLFNet, self).__init__()
        self.backbone_type = backbone_type
        if self.backbone_type == 'swin':
            img_size = 224
            num_heads_cm = [3, 6, 12, 24]
            depths = [2, 2, 6, 2]
            patch_size = 4
            embed_dim = 96
            self.channels = 96
            self.backbone_rgb = SwinTransformerBackbone(img_size=img_size, patch_size=patch_size, in_chans=3,
                            embed_dim=96, depths=depths, num_heads=[3, 6, 12, 24], window_size=7)
            self.backbone_fs = SwinTransformerBackbone(img_size=img_size, patch_size=patch_size, in_chans=3,
                            embed_dim=96, depths=depths, num_heads=[3, 6, 12, 24], window_size=7)
        else:
            img_size = 256
            num_heads_cm=[1, 2, 5, 8]
            depths = [2, 2, 6, 2]
            patch_size = 4
            embed_dim = 96
            self.channels = 64
            self.backbone_rgb = pvt_v2_b2()
            self.backbone_fs = pvt_v2_b2()

        self.image_size = img_size
        self.cm_module = nn.ModuleList()
        self.num_layers = len(depths)
        self.patch_reso = img_size // patch_size
        self.upsample = nn.ModuleList()
        dims = [64, 128, 320, 512]

        for i_layer in range(self.num_layers):
            fea_reso = self.patch_reso // (2 ** i_layer)
            if self.backbone_type == 'swin':
                dim = (2**i_layer)*embed_dim
            else:
                dim = dims[i_layer]
            layer = PGFormer(dim=dim, num_heads=num_heads_cm[i_layer], mlp_ratio=4.,
                                 qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
            self.cm_module.append(layer)
            if self.backbone_type == 'swin':
                upsample = PatchExpand(input_resolution=[fea_reso, fea_reso], in_dim=dim, out_dim=int(dim/2), norm_layer=nn.LayerNorm)
            else:
                if i_layer == 0:
                    upsample = PatchExpand(input_resolution=[fea_reso, fea_reso], in_dim=dims[i_layer],
                                           out_dim=dims[i_layer],norm_layer=nn.LayerNorm)
                else:
                    upsample = PatchExpand(input_resolution=[fea_reso, fea_reso], in_dim=dims[i_layer],
                                           out_dim=dims[i_layer - 1], norm_layer=nn.LayerNorm)
            self.upsample.append(upsample)
        self.cm_module = self.cm_module[::-1]
        self.upsample = self.upsample[::-1] # reverse the list
        if self.backbone_type == 'swin':
            self.upsample_x4 = FinalPatchExpand_X4(input_resolution=[self.patch_reso, self.patch_reso], dim=embed_dim, dim_scale=4, norm_layer=nn.LayerNorm)
        else:
            self.upsample_x4 = FinalPatchExpand_X4(input_resolution=[self.patch_reso, self.patch_reso], dim=dims[0],
                                                   dim_scale=4, norm_layer=nn.LayerNorm)

        self.score_module = ScoreModule(self.channels)

        self.contour = ScoreModule(self.channels)

        self.extract_context = Conv3(3, self.channels)
        self.fuse_edge_region = Conv3(2 * self.channels, self.channels)
        self.score_module_coarse = ScoreModule(self.channels)
        self.fuse_context_region = Conv3(2 * self.channels, self.channels)

    def load_pretrained(self, load_path):
        if self.backbone_type=='swin':
            if not os.path.exists(load_path):
                print("pretrained model path not exist")
            pretrained_dict = torch.load(load_path, map_location=torch.device('cpu'))
            for k, v in pretrained_dict.items():
                pretrained_dict = v

            model_dict = self.backbone_rgb.state_dict()

            renamed_dict = dict()
            for k, v in pretrained_dict.items():
                k = k.replace('layers.0.downsample', 'downsamples.0')
                k = k.replace('layers.1.downsample', 'downsamples.1')
                k = k.replace('layers.2.downsample', 'downsamples.2')
                if k in model_dict:
                    renamed_dict[k] = v
            model_dict.update(renamed_dict)
            self.backbone_fs.load_state_dict(model_dict, strict=True)
            self.backbone_rgb.load_state_dict(model_dict, strict=True)
        else:
            self.backbone_rgb.init_weights(load_path)
            self.backbone_fs.init_weights(load_path)

    def forward(self, fs, rgb):
        side_rgb_x = self.backbone_rgb(rgb)
        side_fs_x = self.backbone_fs(fs)
        side_rgb_x = side_rgb_x[::-1]
        side_fs_x = side_fs_x[::-1]

        if self.backbone_type == 'pvt':
            side_rgb_x = [ImageToPatch(x) for x in side_rgb_x]
            side_fs_x = [ImageToPatch(x) for x in side_fs_x]

        fused_fea = self.cm_module[0](side_rgb_x[0], side_fs_x[0])

        for i in range(1, self.num_layers):
            fused_fea = self.upsample[i - 1](fused_fea)
            fused_fea = self.cm_module[i](side_rgb_x[i] + fused_fea, side_fs_x[i])

        fused_fea = self.upsample_x4(fused_fea)   # [1, h/4 * w/4, 96] -> [1, 96, h, w]
        coarse = self.score_module_coarse(fused_fea) # [1, h, w, 1]

        weight = DistanceWeight(coarse)
        context_features = self.extract_context(rgb)
        edge_feature = self.fuse_context_region(torch.cat((fused_fea, (context_features * coarse) + weight), dim=1))
        contour = self.contour(edge_feature)

        fused_fea = self.fuse_edge_region(torch.cat((edge_feature, fused_fea), dim=1))
        predict = self.score_module(fused_fea)

        return predict, contour, coarse

def init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


if __name__ == "__main__":
    # use case

    ## swin Transformer
    TLFNet = TLFNet("swin")
    TLFNet.apply(init_weights)
    TLFNet.load_pretrained('./pretrained_wight/swin_tiny_patch4_window7_224.pth')
    print(TLFNet)
    rgb_image = torch.zeros([1, 3, 224, 224])
    fs_image = torch.zeros([12, 3, 224, 224])
    h = TLFNet(fs_image, rgb_image)
    debug = 0

    # ## PVT
    # TLFNet = TLFNet("pvt")
    # TLFNet.apply(init_weights)
    # TLFNet.load_pretrained('./pretrained_wight/pvt_v2_b2.pth')
    # print(TLFNet)
    # rgb_image = torch.zeros([1, 3, 256, 256])
    # fs_image = torch.zeros([12, 3, 256, 256])
    # h = TLFNet(fs_image, rgb_image)
    # debug = 0


