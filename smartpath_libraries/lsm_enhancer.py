from .models import model_unet
import torch
import torch.nn.functional as F
from skimage import io, transform, img_as_float, color, img_as_ubyte, exposure
import os
import numpy as np


class LSMEnhancer:
    
    def __init__(self, config):
        ### store config
        self.config = config
        ### construct model
        model = model_unet.Generator(1, config["cnn-base-channel"], norm=None)
        if config["lsm-resolution"] == 512:
            model.load_state_dict(torch.load(os.path.join('model-weights', 'self-512.pth'), map_location=torch.device('cpu')))
        elif config["lsm-resolution"] == 256:
            model.load_state_dict(torch.load(os.path.join('model-weights', 'self-256.pth'), map_location=torch.device('cpu')))
        else:
            raise ValueError('Specified resolution is not supported.')
        model.eval()
        if config["gpu"] == True:
            self.model = model.cuda()
        else:
            self.model = model
            
    def compute(self, image):
        with torch.no_grad():
            image = img_as_float(image)
            image = exposure.rescale_intensity(image, in_range=(self.config['norm-range'][0]/65535, self.config['norm-range'][1]/65535), out_range=(0, 1))
            image_tensor = torch.from_numpy(image).view(1, 1, self.config["lsm-resolution"], self.config["lsm-resolution"]) # 1x1xHxW
            device = next(self.model.parameters()).device
            image_tensor = image_tensor.float().to(device)
            image_batch = image_tensor.repeat(self.config['batch-size'], 1, 1, 1)
            tensor_hyper = torch.zeros((1, image_tensor.shape[1], image_tensor.shape[2], image_tensor.shape[3]))
            pass_time = int(1/self.config['dropout-rate'])
            for i in range(pass_time):
                drop_mask = F.dropout(torch.ones(image_batch.shape).float().to(device), p=self.config['dropout-rate'], inplace=True).to(device)
                real_mid = torch.mul(image_batch, drop_mask) 
                prediction = self.model(image_tensor)
                prediction = torch.mul(prediction, 1/(1-self.config['dropout-rate'])-drop_mask) / self.config['dropout-rate']
                tensor_hyper = tensor_hyper + torch.mean(prediction, 0).cpu()
            prediction = tensor_hyper/pass_time
            image = np.clip(prediction.squeeze().cpu().numpy(), 0, 1)
        return image