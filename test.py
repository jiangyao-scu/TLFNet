import torch
from torch.utils.data import DataLoader
from lib.utils import LFDataset
import cv2
import os

from evalute.my_evalutor import my_evalutor

from TLFNet import TLFNet
from tqdm import tqdm
import argparse

def test(model, dataset, save_sal, device):
    model.eval()
    with torch.no_grad():
        for allfocus, fs, GT, names in tqdm(dataset, desc='Evaluating', leave=True):
            basize, dime, height, width = fs.size()
            inputs_focal = fs.view(1, basize, dime, height, width).transpose(0, 1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, 12, dim=2), dim=1)
            inputs_focal = torch.cat(torch.chunk(inputs_focal, basize, dim=0), dim=1).squeeze()

            allfocus = allfocus.to(device)
            inputs_focal = inputs_focal.to(device)

            out, contour, coarse = model(inputs_focal, allfocus)

            name = names[0]

            pre_sal = torch.sigmoid(out)
            pre_sal = pre_sal.permute(1, 0, 2, 3)[0]
            pre_sal = pre_sal.cpu().detach().numpy().copy()
            pre_sal = pre_sal[0]
            pre_sal = pre_sal * 255
            cv2.imwrite(os.path.join(save_sal, name.split('.')[0]+'.png'), pre_sal)

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for testing TLFNet.')

    # project settings
    parser.add_argument('--backbone', default='swin')
    parser.add_argument('--model_path', default='xxx/TLFNet.pth')
    parser.add_argument('--save_path', default='./results/TLFNet/')
    parser.add_argument("--cuda", default="0")
    parser.add_argument("--local_rank", default=0)

    # data settings
    parser.add_argument("--eval_data_location", type=str, default='xxx/data/test/')
    parser.add_argument("--eval_dataset", default=['DUTLF-FS','LFSD', 'HFUT-Lytro', 'Lytro Illum'])
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_worker", type=int, default=2)


    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TLFNet(args.backbone)
    model.load_state_dict(torch.load(args.model_path))
    model = model.to(device)

    for dataset in args.eval_dataset:
        save_sal = os.path.join(args.save_path, dataset)
        if not os.path.exists(save_sal):
            os.makedirs(save_sal)

        data = LFDataset(location=os.path.join(args.eval_data_location, dataset),
                         crop=False, train=False, image_size=args.image_size)
        test_dataloader = DataLoader(data, batch_size=1, shuffle=False)

        test(model, test_dataloader, save_sal, device)

    # torch.cuda.empty_cache()
    # my_evalutor(save_dir='./', gt_dir='/home/brl/work/dataset/dataset/LFSOD/gt',
    #             pred_dir='./results_test/', cuda=True)
