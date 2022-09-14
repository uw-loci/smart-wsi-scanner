import torch.nn as nn
from .backbones import UNet, UpScale, Discriminator, weights_init_normal
from .lsm_dataset import prepare_train_valid_loader
import os
from glob import glob
import os.path as path
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import shutil
import random
import numpy as np
from skimage import io, img_as_uint, exposure, transform
from skimage.metrics import peak_signal_noise_ratio
import warnings
from .lsm_utils import screen_background, PerceptualLoss
from .DBPN import dbpn
from pytorch_fid.fid_score import calculate_fid_given_paths
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import datetime


class Enhancer(nn.Module):
    def __init__(self, config, scale_factor=1, adversarial_loss=False, perceptual_loss=False):
        super(Enhancer, self).__init__()
        
        self.config = config
        os.makedirs('output', exist_ok=True)
        self.writer = SummaryWriter()
        self.log_train = []
        self.log_loss = []
        self.log_percep = []
        self.log_adv_g = []
        self.log_adv_d = []
        self.log_psnr = []
        self.log_fid = []

        if scale_factor > 1:
            self.backbone = dbpn.Net(
                num_channels=config['image-channel'], 
                base_filter=config['cnn-base-channel'], 
                feat=config['cnn-base-channel']*4, 
                num_stages=7, 
                scale_factor=scale_factor)
        else:
            self.backbone = UNet(
                in_channels=config['image-channel'], 
                out_channels=config['cnn-base-channel'], 
                init_features=config['image-channel'], 
                pretrained=False)

        if adversarial_loss:
            self.discriminator = Discriminator(in_channels=config['image-channel'])
            self.criterion_gan = nn.MSELoss()
            self.backbone.apply(weights_init_normal)
            self.discriminator.apply(weights_init_normal)
        self.adversarial_loss = adversarial_loss

        if perceptual_loss:
            self.perceptual = PerceptualLoss()
        self.perceptual_loss = perceptual_loss

        self.criterion = nn.L1Loss(reduction='none')
        self.alpha = config['loss-gain']

        if self.config['gpu']: self.cuda()

        mydir = os.path.join('model_weights', 'supervised'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(mydir, exist_ok=True)
        self.weights_dir = mydir


    def configure_dataset(self, exclude_bg=True):
        config = self.config
        input_fnames = glob(path.join(config['dataset'], 'input', '*.'+config['image-extension']))
        target_fnames = glob(path.join(config['dataset'], 'target', '*.'+config['image-extension']))
        n_pairs = len(input_fnames)
        if exclude_bg:
            print('Screening background...')
            exclude_names = screen_background((input_fnames+target_fnames))
            input_fnames = [i for i in input_fnames if os.path.basename(i) not in exclude_names]
            target_fnames = [i for i in target_fnames if os.path.basename(i) not in exclude_names]
            assert len(input_fnames)==len(target_fnames)
        pair_fnames = list(zip(input_fnames, target_fnames))
        print(f'All image pairs: {n_pairs}, Remaining image pairs: {len(pair_fnames)}')
        train_loader, valid_loader = prepare_train_valid_loader(pair_fnames, config['norm-range'], config['norm-range-target'], 0.95, config['batch-size'], config['threads'])
        self.valid_dataloader = valid_loader
        self.train_dataloader = train_loader
        self.valid_list = input_fnames


    def configure_optimizer(self):
        config = self.config
        n_epoch = int(config["iterations"]/len(self.train_dataloader))
        if self.adversarial_loss:
            self.optimizer = Adam(self.backbone.parameters(), lr=config['learning-rate'], weight_decay=0.0001)
            self.optimizer_d = Adam(self.discriminator.parameters(), lr=0.1*config['learning-rate'], weight_decay=0.0001)
            self.scheduler = CosineAnnealingLR(self.optimizer, n_epoch, 0.00001)
        else:
            self.optimizer = Adam(self.backbone.parameters(), lr=config['learning-rate'], weight_decay=0.0001)
            self.scheduler = CosineAnnealingLR(self.optimizer, n_epoch, 0.000005)


    def forward(self, x):
        out = self.backbone(x)
        return out


    def train_epoch(self, epoch=1, total_epoch=1):
        print('\n Training...')
        model = self.backbone
        config = self.config
        if self.adversarial_loss: self.discriminator.train()
        dataloader = self.train_dataloader
        criterion = self.criterion
        optimizer = self.optimizer
        epoch_loss = 0
        perceptual_epoch_loss = 0
        g_epoch_loss = 0
        d_epoch_loss = 0
        model.train()
        device = next(model.parameters()).device
        n_iter = min(len(dataloader), config['iter-per-epoch'])
        for iteration, batch in enumerate(dataloader):
            if iteration >= n_iter: break
            input = batch['input'].float().to(device)
            target = batch['target'].float().to(device)
            self.optimizer.zero_grad()
            output = model(input)
            loss_pixel = torch.mean(criterion(output*self.alpha, target*self.alpha))
            if self.perceptual_loss:
                perceptual_loss = self.perceptual(output, target)
                loss_pixel = config['percep-lambda']*perceptual_loss + loss_pixel   
                perceptual_epoch_loss += perceptual_loss.item()  
            if self.adversarial_loss:
                valid = torch.tensor(np.ones((input.shape[0], config['image-channel'], 4, 4)), requires_grad=False, device=device, dtype=torch.float32)
                fake = torch.tensor(np.zeros((input.shape[0], config['image-channel'], 4, 4)), requires_grad=False, device=device, dtype=torch.float32)
                pred_fake = self.discriminator(output, input)
                loss_g = self.criterion_gan(pred_fake, valid)
                loss_pixel = config['gan-lambda']*loss_g + loss_pixel
                g_epoch_loss += loss_g.item()           
            loss_pixel.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss_pixel.item()         
            if self.perceptual_loss and not self.adversarial_loss:
                print(f'[{epoch}/{total_epoch}] [{iteration}/{n_iter}] Loss: {loss_pixel.item()}, Loss perceptual: {perceptual_loss.item()}', end='\r')
            elif self.adversarial_loss and (iteration % config['adv-interval']==0):
                self.optimizer_d.zero_grad()
                pred_real = self.discriminator(target, input)
                pred_fake = self.discriminator(output.detach(), input)
                loss_d = 0.5 * (self.criterion_gan(pred_real, valid) + self.criterion_gan(pred_fake, fake))
                d_epoch_loss += loss_d.item()
                loss_d.backward()
                self.optimizer_d.step()
                if loss_d.item()<0.15 and loss_g.item()>0.8: 
                    print('Reset Discriminator due to saturation.')
                    self.discriminator.apply(weights_init_normal)
                print(f'[{epoch}/{total_epoch}] [{iteration}/{n_iter}] Loss: {loss_pixel.item()}, Loss G: {loss_g.item()} Loss D: {loss_d.item()}' , end='\r')
            else:
                print(f'[{epoch}/{total_epoch}] [{iteration}/{n_iter}] Loss: {loss_pixel.item()}', end='\r')
            
        print(f"\n ===> Epoch {epoch} Complete: Avg. Loss: {epoch_loss / n_iter:.6f}")
        self.log_train.append((epoch_loss/n_iter, epoch))


    def test(self, epoch):
        with torch.no_grad():
            model = self.backbone
            config = self.config
            if self.adversarial_loss: self.discriminator.eval()
            dataloader = self.valid_dataloader
            criterion = self.criterion
            model.eval()
            epoch_loss = 0
            perceptual_epoch_loss = 0
            g_epoch_loss = 0
            d_epoch_loss = 0
            device = next(model.parameters()).device
            for batch in tqdm(dataloader):
                input = batch['input'].float().to(device)
                target = batch['target'].float().to(device)
                output = model(input) 
                loss_pixel = torch.mean(criterion(output*self.alpha, target*self.alpha))
                if self.perceptual_loss:
                    perceptual_loss = self.perceptual(output, target)
                    loss_pixel = config['percep-lambda']*perceptual_loss + loss_pixel  
                    perceptual_epoch_loss += perceptual_loss.item()
                if self.adversarial_loss:
                    valid = torch.tensor(np.ones((input.shape[0], config['image-channel'], 4, 4)), requires_grad=False, device=device, dtype=torch.float32)
                    fake = torch.tensor(np.zeros((input.shape[0], config['image-channel'], 4, 4)), requires_grad=False, device=device, dtype=torch.float32)
                    pred_fake = self.discriminator(output, input)
                    loss_g = self.criterion_gan(pred_fake, valid)
                    loss_pixel = config['gan-lambda']*loss_g + loss_pixel
                    g_epoch_loss += loss_g.item() 
                epoch_loss = epoch_loss + loss_pixel.item()
                if self.adversarial_loss:
                    pred_real = self.discriminator(target, input)
                    pred_fake = self.discriminator(output.detach(), input)
                    loss_d = 0.5 * (self.criterion_gan(pred_real, valid) + self.criterion_gan(pred_fake, fake))
                    d_epoch_loss += loss_d.item()
            if self.perceptual_loss:
                print(f'>>>> Test Loss - epoch {epoch}: {epoch_loss/len(dataloader)}, Loss perceptual: {perceptual_epoch_loss/len(dataloader)}', end='\n')
                self.log_percep.append((perceptual_epoch_loss/len(dataloader), epoch))
            else:
                print(f'>>>> Test Loss - epoch {epoch}: {epoch_loss/len(dataloader)}', end='\n')
            if self.adversarial_loss:
                print(f'>>>> Test Loss - epoch {epoch}: Loss G {g_epoch_loss/len(dataloader)}, Loss D: {d_epoch_loss/len(dataloader)}', end='\n')
                self.log_adv_g.append((g_epoch_loss/len(dataloader), epoch))
                self.log_adv_d.append((d_epoch_loss/len(dataloader), epoch))
            self.log_loss.append((epoch_loss/len(dataloader), epoch))
            
            

    def write_log(self, write_train=True):
        if write_train:
            self.writer.add_scalar('Loss/train', self.log_train[-1][0], self.log_train[-1][1])
        self.writer.add_scalar('Loss/test', self.log_loss[-1][0], self.log_loss[-1][1])
        if self.perceptual_loss:
            self.writer.add_scalar('Loss/perceptual', self.log_percep[-1][0], self.log_percep[-1][1])
        if self.adversarial_loss:
            self.writer.add_scalar('Loss/generator', self.log_adv_g[-1][0], self.log_adv_g[-1][1])
            self.writer.add_scalar('Loss/discriminator', self.log_adv_d[-1][0], self.log_adv_d[-1][1])
        self.writer.add_scalar('Metric/FID', self.log_fid[-1][0], self.log_fid[-1][1])
        self.writer.add_scalar('Metric/PSNR', self.log_psnr[-1][0], self.log_psnr[-1][1])
        
        
        
    def save_models(self):
        torch.save(self.backbone.state_dict(), os.path.join(self.weights_dir, 'g.pth'))
        if self.adversarial_loss:
            torch.save(self.discriminator.state_dict(), os.path.join(self.weights_dir, 'd.pth'))
            


    def train(self, write_log=False, valid_r=0.2):
        self.configure_dataset()
        self.configure_optimizer()
        config = self.config
        scheduler = self.scheduler
        n_epoch = int(config["iterations"]/min(config["iter-per-epoch"], len(self.train_dataloader)))
        print('Initial testing pass...')
        self.test(0)
        self.enhance(sampling=True, sample_rate=valid_r)
        self.psnr(0)
        self.FID_score(0)
        self.write_log(write_train=False)
        for epoch in tqdm(range(1, n_epoch+1)):
            self.train_epoch(epoch=epoch, total_epoch=n_epoch)   
            # scheduler.step()
            if epoch % config["test-interval"] == 0:
                self.test(epoch)
                self.enhance(sampling=True, sample_rate=valid_r)
                self.psnr(epoch)
                self.FID_score(epoch)
            if write_log:
                self.write_log()

            self.save_models()



    def enhance(self, sampling=False, sample_rate=1):
        with torch.no_grad():
            config = self.config
            model = self.backbone
            os.makedirs(os.path.join('output', config['dataset']), exist_ok=True)
            input_path = os.path.join('output', config['dataset'], 'input')
            output_path = os.path.join('output', config['dataset'], 'output')
            target_path = os.path.join('output', config['dataset'], 'target')
            input_images = self.valid_list
            if sampling:
                shutil.rmtree(os.path.join('output', config['dataset']))
                input_images = random.sample(input_images, int(len(input_images)*sample_rate))
            os.makedirs(input_path, exist_ok=True)
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(target_path, exist_ok=True)
            device = next(model.parameters()).device
            for idx, fname in enumerate(input_images):
                img_arr = img_as_uint(io.imread(fname))
                img_arr = exposure.rescale_intensity(img_arr, in_range=(config['norm-range'][0], config['norm-range'][1]), out_range=(0, 65535)).astype(int)
                img_input = exposure.rescale_intensity(1.0*img_arr, in_range=(0, 65535), out_range=(0, 1))
                img_tensor = torch.from_numpy(img_input)[None, None, :, :].float().to(device)
                prediction = model(img_tensor)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    out_arr = img_as_uint(np.clip(prediction.cpu().numpy().squeeze(), 0, 1))
                    img_name = os.path.basename(fname)
                    io.imsave(os.path.join(output_path, img_name), out_arr)
                    io.imsave(os.path.join(input_path, img_name), img_as_uint(img_arr))
                    target_arr = img_as_uint(io.imread(os.path.join(config['dataset'], 'target', img_name)))
                    target_arr = exposure.rescale_intensity(target_arr, in_range=(config['norm-range'][0], config['norm-range'][1]), out_range=(0, 65535)).astype(int)
                    io.imsave(os.path.join(target_path, img_name), img_as_uint(target_arr))
                print(f'Processed [{idx+1}/{len(input_images)}]', end='\r')
                
    
    def compute(self, img_arr):
        with torch.no_grad():
            config = self.config
            model = self.backbone.eval()
            device = next(model.parameters()).device
            # img_arr = img_as_uint(img_arr)
            img_arr = exposure.rescale_intensity(img_arr*1.0, in_range=(config['norm-range'][0], config['norm-range'][1]), out_range=(0, 1))
            img_tensor = torch.from_numpy(img_arr)[None, None, :, :].float().to(device)
            prediction = model(img_tensor)
            out_arr = np.clip(prediction.cpu().numpy().squeeze(), 0, 1)
            return out_arr
            

    def psnr(self, epoch):
        print('Computing PSNR...')
        config = self.config
        out_fnames = glob(os.path.join('output', config['dataset'], 'output', '*'))
        target_fnames = glob(os.path.join('output', config['dataset'], 'target', '*'))
        psnr = []
        for output, ref in zip(out_fnames, target_fnames):
            output = img_as_uint(io.imread(output))
            ref = img_as_uint(io.imread(ref))
            psnr.append(peak_signal_noise_ratio(ref, output))
        self.log_psnr.append((np.mean(np.asarray(psnr)), epoch))


    def FID_score(self, epoch):
        print('Computing FID score...')
        config = self.config
        out_path = os.path.join('output', config['dataset'], 'output')
        target_path = os.path.join('output', config['dataset'], 'target')
        fidscore = calculate_fid_given_paths(
            (out_path, target_path), 
            batch_size=8, 
            device=next(self.backbone.parameters()).device,
            dims=2048,
            num_workers=config['threads'],
            )
        self.log_fid.append((fidscore, epoch))

