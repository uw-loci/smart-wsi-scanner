import os.path as path
import os
import numpy as np
import pandas as pd
from glob import glob
from skimage import io, img_as_uint, exposure, transform
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import inception_v3, vgg16, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision import transforms
from pycromanager import Dataset as PDataset
from tqdm import tqdm
import random
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage.filters import threshold_otsu


def compute_norm_range(in_dir, ext, percentiles=(0, 100), sample_r=0.1):
    img_fnames = glob(path.join(in_dir, '*.'+ext))
    if sample_r < 1:
        img_fnames = random.sample(img_fnames, int(len(img_fnames)*sample_r))
    max_val = []
    min_val = []
    fail_names = []
    for fname in tqdm(img_fnames):
        try:
            img = img_as_uint(io.imread(fname))
        except:
            print(fname)
            fail_names.append(fname)
        max_val.append(np.percentile(img, percentiles[1]))
        min_val.append(np.percentile(img, percentiles[0]))
    max_val = np.percentile(np.array(max_val), 98)
    min_val = np.percentile(np.array(min_val), 2)
    
    return min_val, max_val, fail_names


def process_pmm_datasets(data_folder, out_folder, n_z=4):
    ### parse images from pmm dataset
    os.makedirs(out_folder, exist_ok=True)
    training_data = glob(path.join(data_folder, '*/'))
    for data_path in tqdm(training_data):
        img_name = data_path.split(os.sep)[-2]
        r_id = img_name.rfind('_')
        img_name = img_name[:r_id]
        try:
            dataset = PDataset(data_path)
        except:
            print('Open dataset failed: ' + data_path)
        d_array = dataset.as_array(stitched=False).squeeze()
        if d_array.shape[0] != n_z: 
            print('Dataset z slice number does not match. Swaping dimensions')
            d_array = d_array.transpose(1, 0, 2, 3)
        for i in range(d_array.shape[1]):
            for z in range(d_array.shape[0]):
                save_fname = path.join(out_folder, img_name+f'---{i}---{z}.tif')
                io.imsave(save_fname, d_array[z, i], check_contrast=False)


def normalize_16bit_images(data_folder, out_folder, ext, percentiles=(0, 100)):
    ### normalize 16-bit image
    training_images = glob(path.join(data_folder, '*.'+ext))
    min_val, max_val, fail_names = compute_norm_range(data_folder, ext, percentiles)
    for img_fname in tqdm(training_images):
        if img_fname in fail_names: continue
        img = io.imread(img_fname)
        img = img_as_uint( exposure.rescale_intensity(img, in_range=(min_val, max_val), out_range=(0, 1)) )
        save_name = path.join(out_folder, path.basename(img_fname))
        io.imsave(save_name, img, check_contrast=False)
    return min_val, max_val, fail_names

def screen_background(fnames, background_frac=0.1):
    sampled = random.sample(fnames, min(5000, len(fnames)))
    intensities = []
    remove_list = []
    for fname in sampled:
        arr = io.imread(fname)
        if np.sum(arr) == 0: continue
        else:
            mask = arr > (threshold_otsu(arr))
            arr = arr * mask
            intensity = np.sum(arr) / (1+np.count_nonzero(mask))
            intensities.append(intensity)
    threshold = 0.8*np.mean(np.array(intensities)) + 0.2*np.min(np.array(intensities))
    for fname in tqdm(fnames):
        arr = io.imread(fname)
        if np.sum(arr) == 0:
            remove_list.append(fname)
        else:
            mask = arr > (threshold_otsu(arr))
            arr = arr * mask
            intensity = np.sum(arr) / (1+np.count_nonzero(mask))
            if intensity < threshold:
                if np.random.uniform(0, 1)>background_frac:
                    remove_list.append(fname)
    remove_name_list = list(set([os.path.basename(i) for i in remove_list]))
    return remove_name_list


def eval_psnr(fnames, ref_fnames):
    psnr = []
    for output, ref in tqdm(zip(fnames, ref_fnames)):
        output = img_as_uint(io.imread(output))
        ref = img_as_uint(io.imread(ref))
        if output.shape != ref.shape:
            output = img_as_uint(transform.resize(output, ref.shape, order=3))
        psnr.append(peak_signal_noise_ratio(ref, output))
    return np.mean(np.asarray(psnr))


def eval_ssim(fnames, ref_fnames):
    ssim = []
    for output, ref in tqdm(zip(fnames, ref_fnames)):
        output = img_as_uint(io.imread(output))
        ref = img_as_uint(io.imread(ref))
        if output.shape != ref.shape:
            output = img_as_uint(transform.resize(output, ref.shape, order=3))
        ssim.append(structural_similarity(ref, output))
    return np.mean(np.asarray(ssim))


def eval_FID(in_dir, ref_dir, device='cpu', threads=0):
    fidscore = calculate_fid_given_paths(
        (in_dir, ref_dir), 
        batch_size=8, 
        device=device,
        dims=2048,
        num_workers=threads,
        )
    return fidscore


def read_CA_feats(im_folder, img_name):
    CA_feats_stats_path = os.path.join(im_folder, 'CA_Out', '_'.join((img_name, 'stats.csv')))
    CA_feats = pd.read_csv(CA_feats_stats_path, header=None, index_col=0)
    seg_var = CA_feats.iloc[2].name
    seg_var = float(''.join((ch if ch in '0123456789.' else ' ') for ch in seg_var))
    seg_align = CA_feats.iloc[4].name
    seg_align = float(''.join((ch if ch in '0123456789.' else ' ') for ch in seg_align))
    CA_feats_num_path = os.path.join(im_folder, 'CA_Out', '_'.join((img_name, 'values.csv')))
    CA_feats_num = len(pd.read_csv(CA_feats_num_path, header=None, index_col=0))
    return seg_align, CA_feats_num


class InceptionScore(nn.Module):
    def __init__(self):
        super(InceptionScore, self).__init__()

        inception = inception_v3(pretrained=True, transform_input=False).type(torch.float32)
        inception.fc = nn.Identity()
        for param in inception.parameters():
            param.requires_grad = False
        inception.eval()
        self.inception = inception


    def forward(self, x, ref, resample=True):
        device = x.device
        if resample:
            up = nn.Upsample(size=(299, 299), mode='bilinear').type(torch.float32).to(device)
            x = up(x)
            ref = up(ref)
        if x.shape[1] != 3:
            x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
            ref = ref.expand(ref.shape[0], 3, ref.shape[2], ref.shape[3])
        x_feats = self.inception(x)[0]
        ref_feats = self.inception(ref)[0]
        score = F.mse_loss(x_feats, ref_feats)
        return score



class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg16(weights=VGG16_Weights.DEFAULT).type(torch.float32)
        vgg.classifier = nn.Identity()
        self.feature_layers = {'features.15' : 'feat1', 'features.22' : 'feat2', 'features.29' : 'feat3'}
        feature_extractor = create_feature_extractor(vgg, self.feature_layers)
        for param in feature_extractor.parameters():
            param.requires_grad = False
        self.feature_extractor = feature_extractor.eval()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def forward(self, x, ref, normalize=True):
        if x.shape[1] != 3:
            x = x.expand(x.shape[0], 3, x.shape[2], x.shape[3])
            ref = ref.expand(ref.shape[0], 3, ref.shape[2], ref.shape[3])
        if normalize:
            x = self.normalize(x)
            ref = self.normalize(ref)
        x_feats = self.feature_extractor(x)
        ref_feats = self.feature_extractor(ref)
        i = 0
        for (k1, v1), (k2, v2) in zip(x_feats.items(), ref_feats.items()):
            if i==0: loss = F.mse_loss(v1, v2)
            else: loss += F.mse_loss(v1, v2)
            i += 1
        return loss