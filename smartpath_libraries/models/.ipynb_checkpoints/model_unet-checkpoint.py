import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F        
        
class ConvBlock(torch.nn.Module):   
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, norm=None):
        super(ConvBlock, self).__init__()
        self.is_norm = norm   
        self.conv0 = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias)
        if norm == 'instance':
            self.norm0 = nn.InstanceNorm2d(output_size)
            self.norm1 = nn.InstanceNorm2d(output_size)
        elif norm == 'batch':
            self.norm0 = nn.BatchNorm2d(output_size)
            self.norm1 = nn.BatchNorm2d(output_size)
        self.act = nn.PReLU()
        self.conv1 = nn.Conv2d(output_size, output_size, 1, 1, bias=False)
        
    def forward(self, x):      
        x = self.conv0(x)
        if self.is_norm is not None:
            x = self.norm0(x)
            x = self.act(x)
            x = self.conv1(x)
            x = self.norm1(x)
        else:
            x = self.act(x)
            x = self.conv1(x)              
        return x
    
    
class SkipBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, stride=2):
        super(SkipBlock, self).__init__()
        self.conv = nn.Conv2d(input_size, output_size, 1, stride, bias=False)    
    def forward(self, x):  
        x = self.conv(x)  
        return x
    
    
class DownBlock(nn.Module):
    def __init__(self, in_size, out_size, norm=None):
        super(DownBlock, self).__init__()
        self.conv0 = ConvBlock(in_size, out_size, 4, 2, 1, True, norm)
        self.skip = SkipBlock(in_size, out_size, stride=2)
        self.act = nn.PReLU()

    def forward(self, x):
        x = self.conv0(x) + self.skip(x)
        x = self.act(x)
        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_size, out_size, norm=None):
        super(UpBlock, self).__init__()
        self.conv = ConvBlock(in_size, out_size, 3, 1, 1, True, norm)
        self.skip = SkipBlock(in_size, out_size, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x) + self.skip(x)
        return x
    
    
class PixBlock(nn.Module):
    def __init__(self, in_size, out_size=3, scale=2, norm=None):
        super(PixBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size * (2**scale), 1, 1)
        self.up = nn.PixelShuffle(scale)
    def forward(self, x):
        x = self.conv1(x)
        x = self.up(x)
        return x
    
    
class Generator(nn.Module):
    def __init__(self, in_channel=3, base_channel=32, norm=None):
        super(Generator, self).__init__()
        self.down0 = DownBlock(in_channel, base_channel*2, norm=norm)
        self.down1 = DownBlock(base_channel*2, base_channel*4, norm=norm)
        self.down2 = DownBlock(base_channel*4, base_channel*8, norm=norm)
        self.down3 = DownBlock(base_channel*8, base_channel*16, norm=norm)       
        self.up0 = UpBlock(base_channel*16, base_channel*8, norm=norm)
        self.up1 = UpBlock(base_channel*16, base_channel*4, norm=norm)
        self.up2 = UpBlock(base_channel*8, base_channel*2, norm=norm)
        self.up3 = UpBlock(base_channel*4, in_channel, norm=norm)        
        self.skip0 = ConvBlock(base_channel*2, base_channel*4)
        self.skip1 = ConvBlock(base_channel*4, base_channel*8)
        self.skip2 = ConvBlock(base_channel*8, base_channel*16)
    
    def forward(self, x):
        d0 = self.down0(x)  # 224
        s0 = self.skip0(d0)
        d1 = self.down1(d0) # 112
        s1 = self.skip1(d1)
        d2 = self.down2(d1) # 56x256
        s2 = self.skip2(d2)
        d3 = self.down3(d2) # 28x512
        up2 = self.up0(d3) # 56x256
        up1 = self.up1(torch.cat((up2, d2), 1) + s2)
        up0 = self.up2(torch.cat((up1, d1), 1) + s1)
        out = self.up3(torch.cat((up0, d0), 1) + s0)      
        return out