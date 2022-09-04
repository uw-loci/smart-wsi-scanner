import os, glob
import random
import pandas as pd
from skimage import io, transform, img_as_float, exposure, img_as_uint
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
import torchvision.transforms.functional as F
from .lsm_utils import screen_background



class LSM_Dataset(Dataset):
    def __init__(self, csv_file, transform=None, norm_range=None, resolution=256):   
        self.files_list = pd.read_csv(csv_file)
        self.transform = transform
        self.norm = norm_range
        self.img_size = resolution

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        if self.norm == None:
            in_range = 'image'
        else:
            in_range = (self.norm[0]/65535, self.norm[1]/65535)

        img = io.imread(self.files_list.iloc[idx, 0])
        img = img_as_float(img)
        img = exposure.rescale_intensity(img, in_range=in_range, out_range=(0, 1))
        if self.img_size != img.shape[0] or self.img_size != img.shape[1]:
            img = transform.resize(img, (self.img_size, self.img_size))

        if self.transform:
            sample = self.transform(img)
        return sample
    

class ToTensor(object):
    def __call__(self, img):
        return {'input': F.to_tensor(np.array(img)), 'output': F.to_tensor(np.array(img))}

def show_patch(dataloader, index=0, int_type=True):
    """Print a random input sample"""
    for i_batch, sample_batched in enumerate(dataloader):
        if i_batch == index:         
            input_batch, output_batch = sample_batched['input'], sample_batched['output']
            batch_size = len(input_batch)
            im_size = input_batch.size(2)
            grid = utils.make_grid(input_batch)
            img = img_as_uint(input_batch[0].numpy()[0])
            plt.figure(figsize=(20, 10))
            if int_type:
                plt.imshow(img_as_uint(grid.numpy().transpose((1, 2, 0))), interpolation='bilinear')
            else:
                plt.imshow(grid.numpy().transpose((1, 2, 0)), interpolation='bilinear')
            break   

            
def generate_compress_csv(dataset='data-flim', ext='tif', split=0.95, exclude_bg=True):
    """Generate csv files containing the training and validation images
    
    Args:
        dataset (string): Dataset folder name.
        dim (string): Dimension of convolution.
        split (float): Ratio of images used for training.
        
    returns:
        train_csv_path (string): Training CSV file path. valid_csv_pathc (string): Validation CSV file path. 
    """
    imgs = glob.glob(os.path.join(dataset, '*.'+ext))
    if exclude_bg:
        print('Screening background...')
        exclude_names = screen_background(imgs)
        imgs = [i for i in imgs if os.path.basename(i) not in exclude_names]
    imgs_df = pd.DataFrame(imgs)
    imgs_df = imgs_df.sample(frac=1).reset_index(drop=True)
    train_df = pd.DataFrame(imgs_df)
    valid_df = pd.DataFrame(imgs_df[int(split*len(imgs_df)):])
    train_csv_path = os.path.join(dataset+'-train.csv')
    valid_csv_path = os.path.join(dataset+'-valid.csv')
    train_df.to_csv(train_csv_path, index=False)
    valid_df.to_csv(valid_csv_path, index=False)
    return train_csv_path, valid_csv_path


def data_loader(csv_path, batch_size, norm_range, threads=0, resolution=256):
    """Generate dataloader for iterations
    
    Args: dataset_name (string): Dataset folder name. csv_path (string): CSV file that defines the inputs. 
            batch_size (int): Training batch size. Norm_range (int): Normalization range for the dataset.
            threads (int): Number of dataloder workers
    """
    transformed_dataset = LSM_Dataset(csv_file=csv_path, norm_range=norm_range, resolution=resolution, transform=transforms.Compose([ToTensor()]))
    dataloader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=threads)
    
    return dataloader


class ImagePairDataset(Dataset): 
    def __init__(self, img_fnames, norm_range=None, norm_range_target=None): 
        self.img_fnames = img_fnames 
        self.norm_range = norm_range
        self.norm_range_target = norm_range_target
    
    def __len__(self): return len(self.img_fnames) 
    
    def __getitem__(self, idx): 
        img1 = io.imread(self.img_fnames[idx][0])
        img2 = io.imread(self.img_fnames[idx][1]) 
        if self.norm_range is not None:
            img1 = exposure.rescale_intensity(1.0*img1, in_range=(self.norm_range[0], self.norm_range[1]), out_range=(0, 1))
            img2 = exposure.rescale_intensity(1.0*img2, in_range=(self.norm_range_target[0], self.norm_range_target[1]), out_range=(0, 1))
        img1 = img_as_float(img1) 
        img2 = img_as_float(img2) 
        img1 = torch.from_numpy(img1).float()[None, :] 
        img2 = torch.from_numpy(img2).float()[None, :]
        return {'input': img1, 'target': img2}

        
def prepare_train_valid_loader(file_dirs, norm_range, norm_range_target, split=0.95, bs=64, nw=0, exclude_list=None): 
    if exclude_list is not None:
        file_dirs = [i for i in file_dirs if os.path.basename(i) in exclude_list]
    random.shuffle(file_dirs) 
    split_idx = int(np.floor(len(file_dirs)*split)) 
    train_dataset = ImagePairDataset(file_dirs[:split_idx], norm_range, norm_range_target) 
    train_loader = DataLoader(train_dataset, batch_size=bs, num_workers=nw, shuffle=True) 
    if split>0: 
        valid_dataset = ImagePairDataset(file_dirs[split_idx:], norm_range, norm_range_target) 
        valid_loader = DataLoader(valid_dataset, batch_size=bs, num_workers=nw, shuffle=True) 
        return train_loader, valid_loader 
    else: valid_loader=None 
    return train_loader, valid_loader


