import  os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import scipy.io as sio
import cv2
import visdom
import torch

def crop(allfocus, fs, gt, contour):
    image_size = allfocus.shape[0]
    crop_size = image_size - 20

    index_x = 2*random.randint(0,9)
    index_y = 2*random.randint(0,9)

    new_allfocus = allfocus[index_x:index_x + crop_size, index_y:index_y + crop_size]
    new_gt = gt[index_x:index_x + crop_size, index_y:index_y + crop_size]
    new_contour = contour[index_x:index_x + crop_size, index_y:index_y + crop_size]
    new_fs = fs[index_x:index_x + crop_size, index_y:index_y + crop_size]

    new_allfocus = cv2.resize(new_allfocus, (image_size, image_size))
    new_fs = cv2.resize(new_fs, (image_size, image_size))
    new_gt = cv2.resize(new_gt, (image_size, image_size))
    new_contour = cv2.resize(new_contour, (image_size, image_size))

    return new_allfocus, new_fs, new_gt, new_contour


class LFDataset(Dataset):
    def __init__(self, location=None, train=True, crop=True, image_size=224):
        self.location = location
        self.train = train
        self.crop = crop
        self.image_size = image_size

        self.img_list = os.listdir(os.path.join(self.location, 'allfocus'))
        self.num = len(self.img_list)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        allfocus = Image.open(os.path.join(self.location, 'allfocus', img_name))
        allfocus = allfocus.convert('RGB')
        allfocus = allfocus.resize((self.image_size, self.image_size))
        allfocus = np.asarray(allfocus)

        focalstack = sio.loadmat(os.path.join(self.location, 'mat', img_name.split('.')[0]+'.mat'))
        focal = focalstack['img']
        focal = np.asarray(focal, dtype=np.float32)
        focal = cv2.resize(focal, (self.image_size, self.image_size))

        if self.train:
            GT = Image.open(os.path.join(self.location, 'GT', img_name.split('.')[0]+'.png'))
            GT = GT.convert('L')
            GT = GT.resize((self.image_size, self.image_size))
            GT = np.asarray(GT)

            contour = Image.open(os.path.join(self.location, 'contour', img_name.split('.')[0] + '.png'))
            contour = contour.convert('L')
            contour = contour.resize((self.image_size, self.image_size))
            contour = np.asarray(contour)

            if self.crop:
                allfocus, focal, GT, contour = crop(allfocus, focal, GT, contour)

            allfocus = transforms.ToTensor()(allfocus)
            focal = transforms.ToTensor()(focal)
            GT = GT[..., np.newaxis]
            GT = transforms.ToTensor()(GT)
            contour = contour[..., np.newaxis]
            contour = transforms.ToTensor()(contour)
            return allfocus, focal, GT, contour, img_name
        else:
            GT = Image.open(os.path.join(self.location, 'GT', img_name.split('.')[0] + '.png'))
            GT = GT.convert('L')
            GT = GT.resize((self.image_size, self.image_size))
            GT = np.asarray(GT)

            GT = transforms.ToTensor()(GT)
            allfocus = transforms.ToTensor()(allfocus)
            focal = transforms.ToTensor()(focal)
            return allfocus, focal, GT, img_name


