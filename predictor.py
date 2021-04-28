from collections import OrderedDict
import os, glob, sys, subprocess, shutil
from skimage import io
from skimage.color import rgb2gray, rgb2hsv, rgba2rgb
from skimage.util import img_as_ubyte, img_as_float, view_as_windows
from skimage import filters, img_as_bool, img_as_ubyte, img_as_float
from skimage import morphology, transform
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import dsmil as mil

class Predictor:
    
    def __init__(self, config):
        self.config = config        
        if config["classifier"] == 'Supervised':
            if config["classifier-backbone"] == 'ResNet18':
                model = models.resnet18(pretrained=True)  
                model.fc = nn.Linear(512, config["classifier-num-class"])
            model_path = os.path.join('model-weights', 'supervised-classifier.pth')
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            if config["gpu"] == True:
                self.model = model.cuda()
            else:
                self.model = model
        if config["classifier"] == "MIL":
            if config["classifier-backbone"] == "ResNet18":
                resnet = models.resnet18(pretrained=False)
                num_feats = 512
            resnet.fc = nn.Identity()
            embedder = mil.IClassifier(resnet, num_feats, output_class=config["classifier-num-class"])
            state_dict_weights = torch.load(os.path.join('model-weights', 'embedder.pth'), map_location=torch.device('cpu'))
            state_dict_init = embedder.state_dict()
            new_state_dict = OrderedDict()
            for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
                name = k_0
                new_state_dict[name] = v
            embedder.load_state_dict(new_state_dict, strict=False)
            i_classifier = mil.FCLayer(in_size=num_feats, out_size=config["classifier-num-class"]-1)
            b_classifier = mil.BClassifier(input_size=num_feats, output_class=config["classifier-num-class"]-1)
            milnet = mil.MILNet(i_classifier, b_classifier)
            model_path = os.path.join('model-weights', 'mil-classifier.pth')
            milnet.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            self.num_feats = num_feats
            if config["gpu"] == True:
                self.embedder = embedder.cuda()
                self.milnet = milnet.cuda()
            else:
                self.embedder = embedder
                self.milnet = milnet
                
    def compute(self, image):
        with torch.no_grad():
            if self.config["classifier"] == "Supervised":
                img = img_as_float(image)
#                 window_width = int(config["pixel-size-bf-20x"]/config["pixel-size-bf-4x"] * 1392)
#                 window_height = int(config["pixel-size-bf-20x"]/config["pixel-size-bf-4x"] * 1040)
#                 window_shape = (window_height, window_width, 3)
#                 window_step = (int(window_height*0.1), int(window_width*0.1), 3)
#                 img_lists = view_as_windows(img, window_shape, step=window_step)
                window_shape = (224, 224, 3)
                img_lists = view_as_windows(img, window_shape, step=window_shape[0])
                img_lists = img_lists.squeeze().reshape((img_lists.shape[0]*img_lists.shape[1], window_shape[0], window_shape[1], window_shape[2]))
                img_lists = img_lists.transpose((0, 3, 1, 2))
                device = next(self.model.parameters()).device
                img_tensor = torch.from_numpy(img_lists).float().to(device)
                model = self.model
                model.eval()
                prediction = model(img_tensor)
                prediction = torch.mean(prediction, 0)
                prediction = prediction.cpu().numpy()
            if self.config["classifier"] == "MIL":
                embedder = self.embedder
                embedder.eval()
                device = next(self.embedder.parameters()).device
                feats_all = np.zeros((len(image), self.num_feats))
                for i in range(len(image)):
                    img = img_as_float(io.imread(image[i]))
                    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().to(device)
                    feats, c = embedder(img_tensor[None, ])
                    feats_arr = feats.squeeze().cpu().numpy()
                    feats_all[i] = feats_arr
                feats_tensor = torch.from_numpy(feats_all).float().to(device)
                milnet = self.milnet
                milnet.eval()
                ins_prediction, bag_prediction, A, _ = milnet(feats_tensor)
                bag_prediction = bag_prediction.squeeze().cpu().numpy()
                bag_result = np.greater(bag_prediction, np.array(self.config["mil-classifier-thresh"]))
                mean = torch.mean(A).cpu().numpy() * 0.5
                patch_result = np.greater(A.cpu().numpy(), mean).astype(int)
                prediction = (bag_result, patch_result)
        return prediction
    
