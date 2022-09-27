import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Blender3D(nn.Module):
    def __init__(self, base=8, deep_blend=True):
        super(Blender3D, self).__init__()
        
        self.init = nn.Conv3d(1, base, 5, 1, 2)
        
        layer1 = [
            nn.Conv3d(base, base, 3, 1, 1),
            nn.Conv3d(base, base, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        self.layer1 = nn.Sequential(*layer1)
        
        if deep_blend:
            layer2 = [
                nn.AvgPool3d((2, 2, 2)),
                nn.Conv3d(base, base, 3, 1, 1),
                nn.Conv3d(base, base, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='trilinear')    
            ]
            self.layer2 = nn.Sequential(*layer2)
            
        self.out = nn.Conv3d(base, 1, 3, 1, 1)
        
        self.deep_blend = deep_blend
        
    def forward(self, x):
        x0 = self.init(x)
        x1 = self.layer1(x0)
        if self.deep_blend:
            x2 = self.layer2(x1) + x1
            out = self.out(x2+x0) 
        else:
            out = self.out(x1+x0)
        return out
    
class Blender2D(nn.Module):
    def __init__(self, base=8, deep_blend=False):
        super(Blender2D, self).__init__()
        
        layer1 = [
            nn.Conv2d(1, base, 5, 1, 2)
        ]       
        self.layer1 = nn.Sequential(*layer1)
        
        layer2 = [
            nn.Conv2d(base, base, 3, 1, 1),
            nn.Conv2d(base, base, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        self.layer2 = nn.Sequential(*layer2)
        
        if deep_blend:   
            layer3 = [
                nn.AvgPool2d((2, 2)),
                nn.Conv2d(base, base, 3, 1, 1),
                nn.Conv2d(base, base, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear')
            ]
            self.layer3 = nn.Sequential(*layer3)
        
        self.out = nn.Conv2d(base, 1, 3, 1, 1)
        self.deep_blend = deep_blend
        
        
    def forward(self, x): # N x 1 x 160 x 256 x 256
        x0 = x.transpose(0, 2) # 160 x 1 x N x 256 x 256
        x0 = x0.view(x0.shape[0], x0.shape[2], x0.shape[3], x0.shape[4])
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        if self.deep_blend:
            x3 = self.layer3(x2) + x2
            out = self.out(x1+x3) # 160 x N x 256 x 256
        else:
            out = self.out(x1+x2)
        out = out.transpose(0, 1).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        return out
    
    
class Distributor(nn.Module):
    def __init__(self, latent_size=32, mode='nearest'):
        super(Distributor, self).__init__()
        
        self.pool = nn.AdaptiveAvgPool3d((latent_size, latent_size, latent_size))
        
        self.linear = nn.Sequential(
            nn.Linear(latent_size, latent_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_size*2, latent_size*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(latent_size*2, latent_size)
        )
        
        self.mode = mode
        
    def forward(self, x): # N x 1 x 160 x 256 x 256 , N=1
        x0 = self.pool(x) # N x 1 x 24 x 32 x 32
        x1 = torch.flatten(x0, 3, 4) # N x 1 x 24 x 1024
        x1 = x1.transpose(2, 3) # N x 1 x 1024 x 24
        x1 = x1.view(x1.shape[2], 1, x1.shape[3]) # 1024 x 1 x 24
        x2 = self.linear(x1) # 1024 x 1 x 24
        x2 = x2.transpose(0, 1) # 1 x 1024 x 160
        x2 = x2.transpose(1, 2) # 1 x 160 x 65536
        x2 = x2.reshape(x0.shape[0], x0.shape[1], x0.shape[2], x0.shape[3], x0.shape[4]) # N x 1 x 24 x 32 x 32
        x3 = F.interpolate(x2, size=(x.shape[2], x.shape[3], x.shape[4]), mode=self.mode) # N x 1 x 160 x 256 x 256
        return x3 + x
        
        
class Model(nn.Module):
    def __init__(self, blender, distributor):
        super(Model, self).__init__()
        
        self.blender = blender
        self.distributor = distributor
        
    
    def forward(self, x): # N x 1 x 160 x 256 x 256 , N=1
        x = self.blender(x)
        x = self.distributor(x)

        return x