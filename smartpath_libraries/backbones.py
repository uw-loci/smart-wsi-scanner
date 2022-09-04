from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True):
        super(ConvBlock, self).__init__()
        self.conv0 = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        self.act = nn.Sequential(nn.PReLU())
        self.conv1 = nn.Conv2d(output_size, output_size, 1, 1, bias=False)
        
    def forward(self, x):      
        x = self.conv0(x)
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
    def __init__(self, in_size, out_size, kernel_size=4, stride=2, padding=1, bias=True):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_size, out_size, kernel_size, stride, padding, bias=bias)
        self.skip = SkipBlock(in_size, out_size, stride=2)

    def forward(self, x):
        x = self.conv(x) + self.skip(x)
        return x
    
    
class UpBlock(nn.Module):
    def __init__(self, in_size, out_size):
        super(UpBlock, self).__init__()
        self.conv = ConvBlock(in_size, out_size, 3, 1, 1, bias=True)
        self.skip = SkipBlock(in_size, out_size, stride=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv(x) + self.skip(x)
        return x
    
    
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=16, pretrained=False):
        super(UNet, self).__init__()
        self.down0 = DownBlock(in_channels, init_features, 6, 2, 2)
        self.down1 = DownBlock(init_features, init_features*2)
        self.down2 = DownBlock(init_features*2, init_features*4)
        self.down3 = DownBlock(init_features*4, init_features*8)       
        self.up0 = UpBlock(init_features*8, init_features*4)
        self.up1 = UpBlock(init_features*8, init_features*2)
        self.up2 = UpBlock(init_features*4, init_features)
        self.up3 = UpBlock(init_features*2, init_features)        
        self.skip0 = ConvBlock(init_features, init_features*2)
        self.skip1 = ConvBlock(init_features*2, init_features*4)
        self.skip2 = ConvBlock(init_features*4, init_features*8)
        self.conv0 = ConvBlock(in_channels, init_features, 5, 1, 2)
        self.conv1 = ConvBlock(init_features*2, out_channels)
    
    def forward(self, x): 
        x0 = self.conv0(x)
        d0 = self.down0(x)  
        s0 = self.skip0(d0)
        d1 = self.down1(d0) 
        s1 = self.skip1(d1)
        d2 = self.down2(d1) 
        s2 = self.skip2(d2)
        d3 = self.down3(d2) 
        up2 = self.up0(d3) 
        up1 = self.up1(torch.cat((up2, d2), 1) + s2)
        up0 = self.up2(torch.cat((up1, d1), 1) + s1)
        out = self.up3(torch.cat((up0, d0), 1) + s0)
        out = self.conv1(torch.cat((out, x0), 1))
        return out


class UpScale(nn.Module):
    def __init__(self, backbone, in_channels, out_channels, scale_factor=2):
        super(UpScale, self).__init__()
        self.net = nn.Sequential(
            backbone,
            nn.Conv2d(in_channels, in_channels*2, 3, 1, 1),
            nn.PReLU(),
            nn.Conv2d(in_channels*2, out_channels*(scale_factor**2), 3, 1, 1),
            nn.PixelShuffle(scale_factor)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()

        def discriminator_block(in_channels, hidden_channels):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1)]
            # layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.PReLU())
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels*2, 16), # 256
            *discriminator_block(16, 32), # 128
            *discriminator_block(32, 64), # 64
            *discriminator_block(64, 128), # 32
            *discriminator_block(128, 256), # 16
            *discriminator_block(256, 256), # 8
            nn.Conv2d(256, 1, 3, 1, 1),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

    def forward(self, x, ref):
        ref = F.interpolate(ref, size=(x.shape[2], x.shape[3]), mode='bilinear')
        x = self.model(torch.cat((x, ref), 1))
        return x


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)