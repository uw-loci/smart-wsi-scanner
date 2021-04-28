import srmodels as models
import torch
import torch.nn.functional as F
from skimage import io, transform, img_as_float, color, img_as_ubyte, exposure
import os
import numpy as np


class Enhancer:
    
    def __init__(self, config, dropout_rate=0.1, batch_size=64):
        ### store config
        self.config = config
        self.p = dropout_rate
        self.batch_size = batch_size
        ### construct model
        if config["enhancement-type"] == "Self":
            model = models.Generator(1, 8, norm='instance')
            if config["lsm-resolution"] == 512:
                model.load_state_dict(torch.load(os.path.join('model-weights', 'self-512.pth'), map_location=torch.device('cpu')))
            if config["lsm-resolution"] == 256:
                model.load_state_dict(torch.load(os.path.join('model-weights', 'self-256.pth'), map_location=torch.device('cpu')))
        if config["enhancement-type"] == "Supervised":
            model = models.Generator(1, 8, norm='instance')
            if config["lsm-resolution"] == 512:
                model.load_state_dict(torch.load(os.path.join('model-weights', 'sisr-512.pth'), map_location=torch.device('cpu')))
            if config["lsm-resolution"] == 256:
                model.load_state_dict(torch.load(os.path.join('model-weights', 'sisr-256.pth'), map_location=torch.device('cpu')))
        if config["gpu"] == True:
            self.model = model.cuda()
        else:
            self.model = model
            
    def compute(self, image):
        with torch.no_grad():
            image = img_as_float(image)
            if self.config["lsm-resolution"] == 256:
                image = exposure.rescale_intensity(image, in_range=(0.2, 0.8), out_range=(0, 1))
            if self.config["lsm-resolution"] == 512:
                image = exposure.rescale_intensity(image, in_range=(0.1, 0.9), out_range=(0, 1))
            image_tensor = torch.from_numpy(image).view(1, 1, self.config["lsm-resolution"], self.config["lsm-resolution"]) # 1x1xHxW
            device = next(self.model.parameters()).device
            image_tensor = image_tensor.float().to(device)
            image_batch = image_tensor.repeat(self.batch_size, 1, 1, 1)
            tensor_hyper = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]))
            pass_time = int(1/self.p)
            for i in range(pass_time):
                drop_mask = F.dropout(torch.ones(image_batch.shape).float().to(device), p=self.p, inplace=True).to(device)
                real_mid = torch.mul(image_batch, drop_mask) 
                prediction = self.model(image_tensor)
                prediction = torch.mul(prediction, 1/(1-self.p)-drop_mask) / self.p
                tensor_hyper = tensor_hyper + torch.mean(prediction, 0).cpu()
            prediction = tensor_hyper/pass_time
            image = np.clip(prediction.squeeze().cpu().numpy(), 0, 1)
        return image