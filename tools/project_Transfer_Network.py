# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import print_function, division

import numpy as np
import cv2
from caffe2.python import workspace

import os
import torch
import pandas as pd
from skimage import transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
from collections import defaultdict


import logging
import time

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils
import functools

import utils_p
from vgg import Vgg16

#initializing densepose
c2_utils.import_detectron_ops()
cv2.ocl.setUseOpenCL(False)
cfgs = 'configs/DensePose_ResNet101_FPN_s1x-e2e.yaml'
weights = 'https://dl.fbaipublicfiles.com/densepose/DensePose_ResNet101_FPN_s1x-e2e.pkl'
logger = logging.getLogger(__name__)
merge_cfg_from_file(cfgs)
cfg.NUM_GPUS = 1
weights = cache_url(weights, cfg.DOWNLOAD_CACHE)
assert_and_infer_cfg(cache_urls=False)
model = infer_engine.initialize_model_from_cfg(weights)
dummy_coco_dataset = dummy_datasets.get_coco_dataset()
workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
timers = defaultdict(Timer)
t = time.time()

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

class FaceLandmarksDataset(Dataset):
    """fashion dataset."""

    def __init__(self, csv_file, root_dir1, root_dir2, model,timers,transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir1 = root_dir1
        self.root_dir2 = root_dir2
        self.transform = transform
        self.model = model
        self.timers = timers

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        im_name = 'DensePoseData/demo_data/xyz.jpg'
        output_dir = 'DensePoseData/'
        img_name1 = os.path.join(self.root_dir1,
                                self.landmarks_frame.iloc[idx, 0])
        image1 = cv2.imread(img_name1)
        img_name2 = os.path.join(self.root_dir2,
                                self.landmarks_frame.iloc[idx, 1])
        image2 = cv2.imread(img_name2)
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                    self.model, image1, None, timers=self.timers
            )
        im1 = vis_utils.vis_one_image(
                image1[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                output_dir,
                cls_boxes,
                cls_segms,
                cls_keyps,
                cls_bodys,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
                )
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps, cls_bodys = infer_engine.im_detect_all(
                    self.model, image2, None, timers=self.timers
            )
        im2 = vis_utils.vis_one_image(
                image2[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                output_dir,
                cls_boxes,
                cls_segms,
                cls_keyps,
                cls_bodys,
                dataset=dummy_coco_dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
                )
        image3 = cv2.merge((image1,im1,im2))
        sample = {'image1': image1, 'image2': image2, 'image3': image3}

        if self.transform:
            sample = self.transform(sample)

        return sample
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image1, image2,image3 = sample['image1'], sample['image2'],sample['image3']

        h, w = image1.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img1 = transform.resize(image1, (new_h, new_w))
        img2 = transform.resize(image2, (new_h, new_w))
        img3 = transform.resize(image3, (new_h, new_w))
        

        return {'image1': img1, 'image2': img2, 'image3': img3}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image1, image2,image3 = sample['image1'], sample['image2'],sample['image3']
        h, w = image1.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image1 = image1[top: top + new_h,
                      left: left + new_w]
        image2 = image2[top: top + new_h,
                      left: left + new_w]
        image3 = image3[top: top + new_h,
                      left: left + new_w]

        

        return {'image1': image1, 'image2': image2, 'image3': image3}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image1, image2,image3 = sample['image1'], sample['image2'],sample['image3']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image1 = image1.transpose((2, 0, 1))
        image2 = image2.transpose((2, 0, 1))
        image3 = image3.transpose((2, 0, 1))
        
        return {'image1': torch.from_numpy(image1),
                'image2': torch.from_numpy(image2),
                'image3': torch.from_numpy(image3)
                }




def weights_init(m):
    if type(m) == nn.Linear:
        m.weight.data.normal_(0.0, 1e-3)
        m.bias.data.fill_(0.)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#--------------------------------
# Device configuration
#--------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

#--------------------------------
# Hyper-parameters
#--------------------------------
num_epochs = 200
batch_size = 30
learning_rate = 2e-4
learning_rate_decay = 0.95
reg=0.001
val_size = 0
train_size = 0
test_size = 0

csv_file=r'C:\MVC\export.csv'
root_dir1 =r'C:\MVC\Pair1'
root_dir2=r'C:\MVC\Pair2'
train_dataset = FaceLandmarksDataset(csv_file, root_dir1 ,root_dir2, model,timers,
                                           transform=transforms.Compose([ToTensor()]))
csv_file=r'C:\MVC\export.csv'
root_dir1 =r'C:\MVC\Pair1'
root_dir2=r'C:\MVC\Pair2'
Val_dataset = FaceLandmarksDataset(csv_file, root_dir1 ,root_dir2, model,timers,
                                           transform=transforms.Compose([ToTensor()]))
            
csv_file=r'C:\MVC\export.csv'
root_dir1 =r'C:\MVC\Pair1'
root_dir2=r'C:\MVC\Pair2'
test_dataset = FaceLandmarksDataset(csv_file, root_dir1 ,root_dir2, model,timers,
                                           transform=transforms.Compose([ToTensor()]))


train_loader = DataLoader(train_dataset, batch_size=20,shuffle=True)
val_loader = DataLoader(Val_dataset, batch_size=20,shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=20,shuffle=True)



def ResBlock(in_channel,out_channel,stride = 1):
    layers = []
    layers.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1))
    layers.append(nn.BatchNorm2d(out_channel))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1))
    layers.append(nn.BatchNorm2d(out_channel))
    return layers
                                                                                                          


class ConvNet(nn.Module):
    def __init__(self, input_size, hidden_layers, num_classes, norm_layer=None):
        super(ConvNet, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=9, out_channels=64, kernel_size=3, stride=1, padding=1))
        layers.append(nn.InstanceNorm2d(64))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1))
        layers.append(nn.InstanceNorm2d(256))
        layers.append(nn.ReLU())
        self.encode = nn.Sequential(*layers)
        
        layers = ResBlock(256,256)
        self.res1 = nn.Sequential(*layers)
        
        
        self.res2 = nn.Sequential(*layers)
        
        
        self.res3 = nn.Sequential(*layers)
        
        
        self.res4 = nn.Sequential(*layers)
        
        
        self.res5 = nn.Sequential(*layers)
        
        
        self.res6 = nn.Sequential(*layers)
        self.relu = nn.ReLU()
        
        
        layers = []
        layers.append(nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0))
        
        layers.append(nn.ReLU())
        
        layers.append(nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0))
        
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1))
        layers.append(nn.Tanh())
        self.decode  = nn.Sequential(*layers)
        
         


    def forward(self, x):
        
        
        out1 = self.encode(x)
        
        out = self.res1(out1)
        out += out1
        out2 = self.relu(out)
        
        out = self.res2(out2)
        out += out2
        out3 = self.relu(out)
        
        out = self.res3(out3)
        out += out3
        out4 = self.relu(out)
        
        out = self.res4(out4)
        out += out4
        out5 = self.relu(out)
        
        out = self.res5(out5)
        out += out5
        out6 = self.relu(out)
        
        out = self.res6(out6)
        out += out6
        out = self.relu(out)
        
        out = self.decode(out)
        
        return out



def VisualizeFilter(model):

    fig = plt.figure()
    
    for idx, filt  in enumerate(model.layers1[0].weight.data.cpu().numpy()):
   
       plt.subplot(8,16, idx + 1)
       plt.imshow(filt[0, :, :], cmap="viridis")
       plt.axis('off')
    
    plt.pause(0.001)
    fig.show()
    

input_size = []
hidden_size = []
num_classes = []
model = ConvNet(input_size , hidden_size, num_classes, norm_layer=None).to(device)
nc = 3 # number of channels
netD = NLayerDiscriminator(nc)

model.apply(weights_init)

print(model)

VisualizeFilter(model)


# optimizer

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.5, 0.999))
optimizerD = torch.optim.Adam(model.parameters(), lr=learning_rate,betas=(0.5, 0.999))

# Train the model
val = np.inf
lr = learning_rate
total_step = len(train_loader)
vgg = Vgg16(requires_grad=False).to(device)
mse_loss = nn.MSELoss()
real = torch.ones(batch_size,1,30,30)
fake = torch.zeros(batch_size,1,30,30)
for epoch in range(num_epochs):
    for i, sampled_batch in enumerate(train_loader):
        
        # Move tensors to the configured device
        images = sampled_batch['image3']
        labels = sampled_batch['image2']
        images = images.to(device)
        labels = labels.to(device)
        real = real.to(device)
        fake = fake.to(device)

        # Forward pass
        outputs = model(images)
        #Gan discriminator forward pass
        netD.zero_grad()
        outputD = netD(labels)
        errD_real = 0.5 * torch.mean((outputD - real)**2)
        errD_real.backward()
        
        outputD = netD(outputs)
        errD_fake = 0.5 * torch.mean((outputD - fake)**2)
        errD_fake.backward()
        
        errD = errD_fake + errD_real
        optimizerD.step()
        
        
        #gan generator train
       
        
        
        
        x = utils_p.normalize_batch(outputs)
        y = utils_p.normalize_batch(labels)
        features_y = vgg(y)
        features_x = vgg(x)
        style_loss = 0.
        perceptual_loss = 0.
        for ft_y, ft_x in zip(features_y, features_x):
            gm_y = utils_p.gram_matrix(ft_y)
            gm_x = utils_p.gram_matrix(ft_x)
            style_loss += mse_loss(gm_y, gm_x)
            perceptual_loss += mse_loss(ft_y,ft_x)
        
        L1 = torch.abs(outputs - labels)
        L1 = L1.sum()/batch_size
        outputD = netD(outputs)
        lossGan = 0.5 * torch.mean((outputD - real)**2)    
        loss = 0.5 * perceptual_loss + 5*(10**5)*style_loss + L1 + 0.1 * lossGan
        

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
  
    
    # Code to update the lr
    lr *= learning_rate_decay
    update_lr(optimizer, lr)
    model.eval()
    with torch.no_grad():
        lossx = 0
        total = 0
        for images, labels in train_loader:
            images = sampled_batch['image3']
            labels = sampled_batch['image2']
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            x = utils_p.normalize_batch(outputs)
            y = utils_p.normalize_batch(labels)
            features_y = vgg(y)
            features_x = vgg(x)
            style_loss = 0.
            perceptual_loss = 0.
            for ft_y, ft_x in zip(features_y, features_x):
                gm_y = utils_p.gram_matrix(ft_y)
                gm_x = utils_p.gram_matrix(ft_x)
                style_loss += mse_loss(gm_y, gm_x)
                perceptual_loss += mse_loss(ft_y,ft_x)
        
            L1 = torch.abs(outputs - labels)
            L1 = L1.sum()/batch_size
            
            outputD = netD(outputs)
            lossGan = 0.5 * torch.mean((outputD - real)**2)    
            loss = 0.5 * perceptual_loss + 5*(10**5)*style_loss + L1 + 0.1 * lossGan
            lossx += loss.item()
            total += batch_size
            if total == 1000:
              break
         
          

    print('train loss is: {} %'.format(lossx/1000))   
    with torch.no_grad():
        lossx = 0
        for images, labels in val_loader:
            images = sampled_batch['image3']
            labels = sampled_batch['image2']
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            x = utils_p.normalize_batch(outputs)
            y = utils_p.normalize_batch(labels)
            features_y = vgg(y)
            features_x = vgg(x)
            style_loss = 0.
            perceptual_loss = 0.
            for ft_y, ft_x in zip(features_y, features_x):
                gm_y = utils_p.gram_matrix(ft_y)
                gm_x = utils_p.gram_matrix(ft_x)
                style_loss += mse_loss(gm_y, gm_x)
                perceptual_loss += mse_loss(ft_y,ft_x)
        
            L1 = torch.abs(outputs - labels)
            L1 = L1.sum()/batch_size
            
            outputD = netD(outputs)
            lossGan = 0.5 * torch.mean((outputD - real)**2)    
            loss = 0.5 * perceptual_loss + 5*(10**5)*style_loss + L1 + 0.1 * lossGan
            lossx += loss.item()
            

        print('Validataion loss is: {} %'.format(lossx/val_size))
        
        
        valr = lossx/val_size
        if valr < val:
          val = valr
          torch.save(model.state_dict(), 'best_model.ckpt')
        


    model.train()

best_model = torch.load('best_model.ckpt') 
model.load_state_dict(best_model)
with torch.no_grad():
    for images, labels in test_loader:
        images = sampled_batch['image3']
        labels = sampled_batch['image2']
        images = images.to(device)
        labels = labels.to(device)

            # Forward pass
        outputs = model(images)
        x = utils_p.normalize_batch(outputs)
        y = utils_p.normalize_batch(labels)
        features_y = vgg(y)
        features_x = vgg(x)
        style_loss = 0.
        perceptual_loss = 0.
        for ft_y, ft_x in zip(features_y, features_x):
            gm_y = utils_p.gram_matrix(ft_y)
            gm_x = utils_p.gram_matrix(ft_x)
            style_loss += mse_loss(gm_y, gm_x)
            perceptual_loss += mse_loss(ft_y,ft_x)
        
        L1 = torch.abs(outputs - labels)
        L1 = L1.sum()/batch_size
            
        outputD = netD(outputs)
        lossGan = 0.5 * torch.mean((outputD - real)**2)    
        loss = 0.5 * perceptual_loss + 5*(10**5)*style_loss + L1 + 0.1 * lossGan
        lossx += loss.item()
        

    print('Accuracy of the network on the {} test images: {} %'.format(lossx/test_size))



