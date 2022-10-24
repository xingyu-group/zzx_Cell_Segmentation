from torchvision import transforms
from PIL import Image, ImageOps
import os
import torch
import glob
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, ImageFolder
import torchvision.transforms.functional as TF
import numpy as np
import cv2

import matplotlib.pyplot as plt
import random
import torch.nn.functional as F

def add_Gaussian_noise(x, noise_res, noise_std, img_size):
    x = x.unsqueeze(dim = 0)
    ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)

    ns = F.upsample_bilinear(ns, size=[img_size, img_size])

    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(128))
    roll_y = random.choice(range(128))
    ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
    mask = x.sum(dim=1, keepdim=True) > 0.01
    ns *= mask # Only apply the noise in the foreground.
    res = x + ns
    
    res = res.squeeze(dim = 0)
    return res


def cutpaste_transform(size,isize):
    cutpaste_type = CutPaste3Way
    
    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1))
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(cutpaste_type(transform = after_cutpaste_transform))
    
    gt_transforms = transforms.Compose([
                    transforms.Resize((size, size)),
                    transforms.CenterCrop(isize),
                    transforms.ToTensor()])
    
    return train_transform, gt_transforms

def get_data_transforms_augumentation(size, isize):
    mean_train = [0.485]
    std_train = [0.229]
    train_data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomRotation(10),
        # transforms.RandomCrop([10, 10]),              # 先不加crop
        transforms.RandomAffine(10),
        transforms.RandomAutocontrast(0.2),
        # transforms.RandomEqualize(0.1),                 # 先不加equalization
        transforms.ToTensor()])
        # transforms.Normalize(mean=mean_train,
        #                      std=std_train)])
    
    test_data_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.CenterCrop(isize),
        #transforms.CenterCrop(args.input_size),
        transforms.Normalize(mean=mean_train,
                             std=std_train)])
    
    gt_transforms = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.CenterCrop(isize),
        transforms.ToTensor()])
    
    return train_data_transforms, test_data_transforms, gt_transforms
    


class MVTecDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform, gt_transform, phase, args=None):
        
        self.phase = phase
        self.transform = transform
        self.root = root
        self.args = args

        self.gt_transform = gt_transform
        self.img_path = os.path.join(root, 'images')
        self.img_paths = glob.glob(self.img_path + "/*.png")
        self.img_paths.sort()
        
        
        # train_index = np.random.rand(len(self.img_paths)) =< 0.8
        # train_index = random.sample(range(0, len(self.img_paths)), len(self.img_paths)//2)
        # 
        random.shuffle(self.img_paths)
        train_index = len(self.img_paths)/2 * 1
        train_index = int(train_index)
        self.train_img_paths = self.img_paths[:train_index]
        self.test_img_paths = self.img_paths[:train_index]        
        
        self.gt_paths = os.listdir(os.path.join(root, 'labels'))
        assert len(self.img_paths) == len(self.gt_paths), "gt and image label numbers don't match!!!"

    def __len__(self):
        if self.phase == 'train':
            return len(self.train_img_paths)
        elif self.phase == 'test':
            return len(self.test_img_paths)

    def __getitem__(self, idx):
        
        if self.phase == 'train':
            self.tot_path = self.train_img_paths
        elif self.phase == 'test':
            self.tot_path = self.test_img_paths


            
        img_path = self.tot_path[idx]
        img = Image.open(img_path)
        # img = ImageOps.grayscale(img)
        
        img = self.transform(img)
        
        img_name = (img_path.split('/')[-1]).split('.')[-2]
        label_path = os.path.join(self.root, 'labels', img_name + '_label.png')
        
        # convert tensor to index tensor
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        
        label = torch.from_numpy(label).type(torch.int64)
        
        # label = self.gt_transform(label)  
        
        
        label_oneHot = torch.nn.functional.one_hot(label, num_classes = 3).permute(2,0,1).type(torch.float32)
        # img = TF.pad(img, padding=0)
        # label_oneHot= TF.pad(label_oneHot, padding=0)
        # # img = padding_func(img)
        # # label_oneHot = padding_func(label_oneHot)
        
        # i, j, h, w = transforms.RandomCrop.get_params(
        #             img, output_size=(256, 256))
        # img = TF.crop(img, i, j, h, w)
        # label_oneHot = TF.crop(label_oneHot, i, j, h, w)
        
        new_tensor = torch.cat((img, label_oneHot), dim=0)
        new_tensor = self.gt_transform(new_tensor)
        img = new_tensor[:3,:,:]
        label_oneHot = new_tensor[3:,:,:]
        
        
        # return img, label
        return img, label_oneHot


