import torch
import torch.nn.functional as F
from skimage import io, transform, img_as_float, color, img_as_ubyte, exposure
from skimage.util import view_as_windows, montage
import os
import numpy as np
from .models import stylegan, constant_encoder, models, backbones
from .image_utils import montage_blend


class BFEnhancer:
    
    def __init__(self, config):
        self.config = config
        if config['bf-enhencment'] == True:
            latent_bank = stylegan.StyledGenerator()
            encoder = constant_encoder.Encoder()
            decoder = constant_encoder.Decoder()
            generator = models.EDLatentBank(encoder, decoder, latent_bank)
            generator_weights = torch.load('model_weights/bf-enhancer.pth')
            generator.load_state_dict(generator_weights, strict=False)
        generator.eval()
        if config['gpu'] == True:
            self.model = generator.cuda()
        else:
            self.model = generator
            
        self.bf_ppm = config['pixel-size-bf-4x'] # 1.105
        self.ap_ppm = 0.5039
        self.patch_size = (256, 256, 3)
            
    def compute(self, image, overlap=32, mode=0):
        bf_ppm = self.bf_ppm
        ap_ppm = self.ap_ppm
        patch_size = self.patch_size
        device = next(self.model.parameters()).device
        with torch.no_grad():
            image = img_as_float(image)
            width = image.shape[1]
            height = image.shape[0]
            step_size = patch_size[0]-32
            image = transform.resize(image, (int(height * bf_ppm / ap_ppm), int(width * bf_ppm / ap_ppm)), order=1) # resize image to match input resolution
            pad_h = int((np.floor(image.shape[0]/step_size) * step_size + patch_size[0]) - image.shape[0])
            pad_w = int((np.floor(image.shape[1]/step_size) * step_size + patch_size[1]) - image.shape[1])
            image_pad = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            patches = view_as_windows(image_pad, patch_size, step=step_size).squeeze() # row x col x H x W x C
            
            cols = patches.shape[1]
            rows = patches.shape[0]
            batches = torch.from_numpy(patches.transpose(0, 1, 4, 2, 3)) # row x col x C x H x W
            outputs = []
            for batch in batches:
                batch = batch.float().to(device) # col x C x H x W
                output = self.model(batch)
                output = np.clip(output.cpu().numpy(), 0, 1)
                output = output.transpose(0, 2, 3, 1) # col x H x W x C
                outputs.extend(output)
            outputs = np.stack(outputs, axis=0)
            
            canvas = montage_blend(outputs, patch_size, image_pad.shape, overlap, rows, cols)
            
            if mode == 0:
                predicted = transform.resize(canvas[:image.shape[0], :image.shape[1]], (height, width), order=1, anti_aliasing=True)
            elif mode == 1:
                predicted = transform.resize(canvas[:image.shape[0], :image.shape[1]], (height*2, width*2), order=1, anti_aliasing=True) 
            elif mode == 2:
                predicted = canvas[:image.shape[0], :image.shape[1]]
        return predicted