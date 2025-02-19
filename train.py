from TLFNet import TLFNet
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from lib.utils import LFDataset
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
import visdom
from timm.models.layers import trunc_normal_
from torch import nn
from tqdm import tqdm
import argparse

def structure_loss(pred, mask):
    weit = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3))/weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def FocalLoss(pred, mask, weight=None, gamma=2.0, alpha=0.25, reduction='mean', avg_factor=None):
    pred_sigmod = torch.sigmoid(pred)
    pt = (1-pred_sigmod) * mask + pred_sigmod * (1 - mask)
    focal_weight = (alpha * mask + (1 - alpha) * (1 - mask)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, mask, reduction='none') * focal_weight
    return loss.mean()

def hybrid_e_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='mean')

    pred = torch.sigmoid(pred)
    mpred = pred.mean(dim=(2, 3)).view(pred.shape[0], pred.shape[1], 1, 1).repeat(1, 1, pred.shape[2], pred.shape[3])
    phiFM = pred - mpred

    mmask = mask.mean(dim=(2, 3)).view(mask.shape[0], mask.shape[1], 1, 1).repeat(1, 1, mask.shape[2], mask.shape[3])
    phiGT = mask - mmask

    EFM = (2.0 * phiFM * phiGT + 1e-8) / (phiFM * phiFM + phiGT * phiGT + 1e-8)
    QFM = (1 + EFM) * (1 + EFM) / 4.0
    eloss = 1.0 - QFM.mean(dim=(2, 3))

    inter = ((pred * mask)).sum(dim=(2, 3))
    union = ((pred + mask)).sum(dim=(2, 3))
    wiou = 1.0 - (inter + 1 + 1e-8) / (union - inter + 1 + 1e-8)

    return (bce + eloss + wiou).mean()

def evaluate(args, Net, datasets, device):
    MAE = []
    for dataset in datasets:
        mae = 0
        test_dataloader = DataLoader(
            LFDataset(location= args.eval_data_location + dataset + '/',
                       crop=False, train=False, image_size=args.image_size), batch_size=1, shuffle=False)

        with torch.no_grad():
            for allfocus, fs, gt, names in tqdm(test_dataloader, desc='Evaluating', leave=True):
                basize, dime, height, width = fs.size()
                inputs_focal = fs.view(1, basize, dime, height, width).transpose(0, 1)
                inputs_focal = torch.cat(torch.chunk(inputs_focal, 12, dim=2), dim=1)
                inputs_focal = torch.cat(torch.chunk(inputs_focal, basize, dim=0), dim=1).squeeze()

                gt = gt.to(device)
                allfocus = allfocus.to(device)
                inputs_focal = inputs_focal.to(device)

                out, edge, coarse = Net(inputs_focal, allfocus)

                pre_sal = torch.sigmoid(out)
                mae = mae + torch.abs(pre_sal - gt).mean()
            MAE.append(mae.cpu().detach().numpy().copy() / len(test_dataloader))
    return np.mean(MAE)

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

def train(args, model, train_dataloader, train_sampler, device, optimizer, local_rank=0, writer=None):
    aveGrad = 0
    MAE = 1.0
    best_epo = 0
    BATCH_SIZE = 4

    if local_rank==0:
        vis = visdom.Visdom(env='train')
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print("The number of parameters: {}".format(num_params))
        prev_time = datetime.now()

    optimizer.zero_grad()
    for epo in range(args.epochs):
        train_sampler.set_epoch(epo)
        r_sal_loss = 0

        model.train()
        for index, (allfocus, fs, GT, contour, names) in enumerate(train_dataloader):
            basize, dime, height, width = fs.size()
            inputs_focal = fs.view(1, basize, dime, height, width).transpose(0, 1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, 12, dim=2), dim=1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, basize, dim=0), dim=1).squeeze()
            GT = GT.to(device)
            contour = contour.to(device)
            allfocus = allfocus.to(device)
            inputs_focal = inputs_focal.to(device)


            out, edge, coarse = model(inputs_focal, allfocus)

            loss = hybrid_e_loss(out, GT) + \
                   FocalLoss(edge, contour) + \
                   hybrid_e_loss(coarse, GT)


            sal_loss = loss/BATCH_SIZE
            r_sal_loss += sal_loss.data
            sal_loss.backward()

            aveGrad += 1
            if aveGrad % BATCH_SIZE == 0:
                clip_gradient(optimizer, 0.5)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            if local_rank == 0:
                if np.mod(index, 10) == 0:
                    print('epoch {}, {}/{}, lr={}, train loss is {}, best epo {}, MAE {}'.format(epo, index, len(train_dataloader),optimizer.param_groups[0]['lr'], sal_loss*BATCH_SIZE,best_epo,MAE))
                    writer.add_scalar('training loss', r_sal_loss * BATCH_SIZE / 10, epo * len(train_dataloader) + index)
                    r_sal_loss = 0

                    vis.images(torch.sigmoid(out), win='sal_final', opts=dict(title='sal_final'))
                    vis.images(torch.sigmoid(coarse), win='sal_coarse', opts=dict(title='sal_coarse'))
                    vis.images(torch.sigmoid(edge), win='edge', opts=dict(title='edge'))
                    vis.images(contour, win='contour', opts=dict(title='contour'))
                    vis.images(GT, win='GT', opts=dict(title='GT'))
                    vis.images(allfocus, win='rgb', opts=dict(title='rgb'))

        if local_rank==0:
            cur_time = datetime.now()
            h, remainder = divmod((cur_time - prev_time).seconds, 3600)
            m, s = divmod(remainder, 60)
            time_str = "Time %02d:%02d:%02d" % (h, m, s)
            prev_time = cur_time
            print('%s'%(time_str))

            if epo > 25:
                torch.save(model.module.state_dict(), os.path.join(args.model_path, 'epoch_{}.pth'.format(epo)))
                print(os.path.join(args.model_path, 'epoch_{}.pth'.format(epo)))
            with torch.no_grad():
                model.eval()
                NEW_MAE = evaluate(args, model, ['DUTLF-FS'], device)

            if NEW_MAE < MAE:
                torch.save(model.module.state_dict(), os.path.join(args.model_path, 'best.pth'))
                print('change best epo to {}, save best model'.format(epo))
                MAE = NEW_MAE
                best_epo = epo

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for training TLFNet.')

    # project settings
    parser.add_argument('--model_path', default='models/TLFNet')
    parser.add_argument('--log_path', default='log/TLFNet')
    parser.add_argument("--cuda", default="0,1")
    parser.add_argument("--local_rank", default=0)

    # data settings
    parser.add_argument("--train_data_location", type=str, default='xxx/data/train/DUTLF-FS/')
    parser.add_argument("--eval_data_location", type=str, default='xxx/data/test/')
    parser.add_argument("--eval_dataset", default=['DUTLF-FS'])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_worker", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)

    # model settings
    parser.add_argument('--backbone', default='swin')
    parser.add_argument('--pretrained_model', default='xxx/swin_tiny_pathc4_window7_224.pth')

    # optimizer settings
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    data = LFDataset(location=args.train_data_location, image_size=args.image_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(data)
    train_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_worker, sampler=train_sampler)

    TLFNet = TLFNet(args.backbone)
    TLFNet.apply(init_weights)
    TLFNet.load_pretrained(args.pretrained_model)
    writer = SummaryWriter(args.log_path)

    if local_rank == 0:
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
    TLFNet.to(device)
    TLFNet = torch.nn.parallel.DistributedDataParallel(TLFNet, device_ids=[local_rank],
                                                       output_device=local_rank, find_unused_parameters=True)

    optimizer = torch.optim.AdamW(TLFNet.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train(args, TLFNet, train_dataloader, train_sampler, device, optimizer, local_rank=local_rank, writer=writer)
