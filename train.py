import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import pytorch_ssim
from srgan_data import TrainDatasetFromFolder, ValDatasetFromFolder, display_transform
from srgan_loss import GeneratorLoss
from srgan import Generator, Discriminator

def train(batch,epochs,train_loader, val_loader,netG,netD,optimizerD,optimizerG,generator_criterion):
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1,epochs):
        running_results = {'d_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}
        train_bar = tqdm(train_loader)
        netG.train()
        netD.train()
        device = 'cuda' if torch.cuda.is_available() else 'cpu' 
        for data,target in train_bar:
            g_update_first = True
            real_img = Variable(target)
            real_img = real_img.to(device, dtype=torch.float32)
            #if torch.cuda.is_available():
                #real_img = real_img.cuda()
            z = Variable(data)
            z = z.to(device, dtype=torch.float32)
            #if torch.cuda.is_available():
                #z = z.cuda()
            fake_img = netG(z)
            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            netG.zero_grad()

            fake_img = netG(z)
            fake_out = netD(fake_img).mean()

            g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss.backward()
            
            fake_img = netG(z)
            fake_out = netD(fake_img).mean()
            
            
            optimizerG.step()
            batch_size = batch
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
    
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, epochs, running_results['d_loss'] / batch,
                running_results['g_loss'] / batch,
                running_results['d_score'] / batch,
                running_results['g_score'] / batch))
    
        netG.eval()
        out_path = 'training_results/SRF_' + str(2) + '/'
        if not os.path.exists(out_path):
            os.makedirs(out_path)


        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
            val_images = []
            for val_lr, val_hr_restore, val_hr in val_bar:
                batch_size = val_lr.size(0)
                valing_results['batch_sizes'] += batch_size
                lr = val_lr
                hr = val_hr
                if torch.cuda.is_available():
                    lr = lr.cuda()
                    hr = hr.cuda()
                sr = netG(lr)
        
                batch_mse = ((sr - hr) ** 2).data.mean()
                valing_results['mse'] += batch_mse * batch_size
                batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                valing_results['ssims'] += batch_ssim * batch_size
                valing_results['psnr'] = 10 * log10((hr.max()**2) / (valing_results['mse'] / valing_results['batch_sizes']))
                valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                val_bar.set_description(
                    desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                        valing_results['psnr'], valing_results['ssim']))
        
                val_images.extend(
                    [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                     display_transform()(sr.data.cpu().squeeze(0))])
            val_images = torch.stack(val_images)
            val_images = torch.chunk(val_images, val_images.size(0) // 15)
            val_save_bar = tqdm(val_images, desc='[saving training results]')
            index = 1
            for image in val_save_bar:
                image = utils.make_grid(image, nrow=3, padding=5)
                utils.save_image(image, out_path + 'epoch_%d_index_%d.png' % (epoch, index), padding=5)
                index += 1
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (4, epoch))
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (4, epoch))
        results['d_loss'].append(running_results['d_loss'] / batch)
        results['g_loss'].append(running_results['g_loss'] / batch)
        results['d_score'].append(running_results['d_score'] / batch)
        results['g_score'].append(running_results['g_score'] / batch)
        results['psnr'].append(valing_results['psnr'])
        results['ssim'].append(valing_results['ssim'])
    
        if epoch % 10 == 0 and epoch != 0:
            out_path = 'statistics/'
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            data_frame.to_csv(out_path + 'srf_' + str(2) + '_train_results.csv', index_label='Epoch')

upscale_factor = 2

from torch.utils.data import DataLoader
train_img = TrainDatasetFromFolder(
    dataset_dir= '/content/DIV2K_train_HR',
    crop_size=88,
    upscale_factor=upscale_factor,
)
print(train_img)
val_img = ValDatasetFromFolder(
    dataset_dir= '/content/DIV2K_valid_HR',
    upscale_factor=upscale_factor,
)

''' val_set = ValDatasetFromFolder('data/DIV2K_valid_HR', upscale_factor=UPSCALE_FACTOR)'''
train_loader = DataLoader(dataset=train_img,num_workers=4, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset=val_img,num_workers=4, batch_size=1, shuffle=False)
netG = Generator(upscale_factor)
netD = Discriminator()
generator_criterion = GeneratorLoss()
if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()




optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

train(
    batch=32,
    epochs = 50, 
    train_loader = train_loader,
    val_loader = val_loader,
    netD = netD,
    netG = netG,
    optimizerD=optimizerD,
    optimizerG=optimizerG,
    generator_criterion=generator_criterion)