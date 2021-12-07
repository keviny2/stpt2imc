#!/usr/bin/env python
# coding: utf-8

# ## Import Packages

# For plotting
import numpy as np
import matplotlib.pyplot as plt
from skimage import io
# For everything
import torch
import torch.nn as nn
import torch.nn.functional as F
# For our model
import torchvision.models as models
from torchvision import datasets, transforms
# For utilities
import os, shutil, time
import cv2 as cv
import subprocess
from multiprocessing.pool import Pool
import time


# Check if GPU is available
use_gpu = torch.cuda.is_available()
# use_gpu = False

# remove .ipynb_chaeckpoint files
subprocess.run('./rm_ipynbcheckpoints.sh', shell=True)


class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    MIDLEVEL_FEATURE_SIZE = 128

    ## First half: ResNet
    resnet = models.resnet18() 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1 = nn.Conv2d(8, 64, kernel_size=3)
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(     
      nn.Conv2d(MIDLEVEL_FEATURE_SIZE, 128, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.Upsample(scale_factor=2),
      nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1),
      nn.Upsample(scale_factor=2)
    )

  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    print('the input for the forward pass has shape:', input.shape)
    midlevel_features = self.midlevel_resnet(input)
    
    print('midlevel_features has shape:', midlevel_features)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    print('output has shape:', output)
    return output


model = ColorizationNet().double()

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.0)


class STPT_IMC_ImageFolder(datasets.ImageFolder):    
    """
    Preprocesses
    """
    def __init__(self, root, transform, bits=8, batch_size=64):
        self.root = root
        self.transform = transform
        self.imc_folder = os.path.join(self.root, 'IMC')
        self.stpt_folder = os.path.join(self.root, 'STPT')
        self.bits = bits # num bits for each pixel in image
        self.batch_size = batch_size
        
    def __len__(self):
        # length of dataset dictated by num aligned IMC images b/c len(IMC) < len(STPT)
        return len(os.listdir(self.imc_folder))
        
    def __getitem__(self, index):
        
        # ====== LOAD IMC IMAGES ======
        # define folder paths for physical section defined by index
        imc_section_folder = os.path.join(self.imc_folder,
                                          'SECTION_{}'.format(str(index).zfill(2)))
        
        # get a list of all .tif images inside imc_section_folder
        imc_img_paths = [os.path.join(imc_section_folder, imc_img_path)
                         for imc_img_path in os.listdir(imc_section_folder)
                         if imc_img_path.endswith('.tif')]
        
        # load imc images
        start = time.time()
        with Pool(maxtasksperchild=100) as p_imc:
            imc_imgs = list(p_imc.imap(self.process_imc_image, imc_img_paths))
        end = time.time()
        print('Loading IMC images took', end-start, 'seconds')
            
        imc_imgs = [torch.unsqueeze(img, 0) for img in imc_imgs] # add an extra dimesion for channel
        imc_imgs = torch.cat(imc_imgs, 0) # (40, 18720, 18720)
        
        
        # ====== LOAD STPT IMAGES ======
        # get path to images
        stpt_img_paths = [os.path.join(self.stpt_folder,
                                       'S{0}_Z{1}.tif'.format(str(index).zfill(3),
                                                          optical_section.zfill(2)))
                          for optical_section in ['0', '1']]  
        
        # load stpt images
        start = time.time()
        with Pool(maxtasksperchild=100) as p_stpt:
            stpt_imgs = list(p_stpt.imap(self.process_stpt_image, stpt_img_paths))
        end = time.time()
        print('Loading STPT images took', end-start, 'seconds')
        
        stpt_imgs = [img.permute((2,0,1)) for img in stpt_imgs] # (C,H,W) tensor
        stpt_imgs = torch.cat(stpt_imgs, 0) # concatenate two stpt images (8, 20800, 20800)
        
        
        # ====== TRANSFORMS ======
        
        transforms.Resize(imc_imgs.shape[1])(stpt_imgs)  # make STPT img same size as IMC (..., 18720, 18720)
        combine = torch.cat((imc_imgs, stpt_imgs), 0) # combine imc and stpt -> (48, 18720, 18720)
        
        # obtain a batch of random crops
        img_set = [self.transform(combine) for i in range(len(self.batch_size))]
            
        # separate imc and stpt -> (40, 18720, 18720), (8, 18720, 18720)
        imc_imgs = [torch.split(img, 40)[0] for img in img_set]
        stpt_imgs = [torch.split(img, 40)[1] for img in img_set]
        
        return stpt_imgs, imc_imgs
    
    def process_stpt_image(self, file_name):
        img = io.imread(file_name)
        return torch.from_numpy(img)
    
    def process_imc_image(self, file_name):
        # read image file
        img = cv.imread(file_name, cv.IMREAD_UNCHANGED)

        # normalize image
        norm_img = img.copy()
        cv.normalize(img, norm_img, alpha=0, beta=2**self.bits - 1, norm_type=cv.NORM_MINMAX)

        # Apply log transformation method
        c = (2**self.bits - 1) / np.log(1 + np.max(norm_img))
        
        log_image = c * (np.log(norm_img + 1))
        
        # Specify the data type so that
        # float value will be converted to int
        return torch.from_numpy(log_image)


def merge_cropped_images(batch):
    """
    takes in a batch of cropped images from a single physical section and forms
    a mini-batch for the DataLoader class
    """
    imgs, targets = zip(*batch)
    return torch.cat(imgs), torch.cat(targets)

# Training
train_transforms = transforms.Compose([transforms.RandomCrop(256)])
train_imagefolder = STPT_IMC_ImageFolder(root='data/train',
                                         transform=train_transforms)
train_loader = torch.utils.data.DataLoader(train_imagefolder,
                                           batch_size=1,
                                           shuffle=True,
                                           collate_fn=merge_cropped_images)

# Validation 
# val_transforms = transforms.Compose([transforms.Resize(256)])
# val_imagefolder = STPT_IMC_ImageFolder(root='data/val', transform=val_transforms)
# val_loader = torch.utils.data.DataLoader(val_imagefolder, batch_size=1, shuffle=False)


class AverageMeter(object):
  '''A handy class from the PyTorch ImageNet tutorial''' 
  def __init__(self):
    self.reset()
  def reset(self):
    self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def validate(val_loader, model, criterion, epoch):
  print('Starting validation epoch {}'.format(epoch)) 
  model.eval()

  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  already_saved_images = False
  for i, (stpt, imc) in enumerate(val_loader):
    data_time.update(time.time() - end)

    # Use GPU
    if use_gpu: 
        stpt, imc = stpt.cuda(), imc.cuda()

    # Run model and record loss
    imc_recons = model(stpt) # throw away class predictions
    loss = criterion(imc_recons, imc)
    losses.update(loss.item(), stpt.size(0))

    # Record time to do forward passes and save images
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to both value and validation
    if i % 25 == 0:
      print('Validate: [{0}/{1}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
             i, len(val_loader), batch_time=batch_time, loss=losses))

  print('Finished validation.')
  return losses.avg


def train(train_loader, model, criterion, optimizer, epoch):
  print('Starting training epoch {}'.format(epoch))
  model.train()
  
  # Prepare value counters and timers
  batch_time, data_time, losses = AverageMeter(), AverageMeter(), AverageMeter()

  end = time.time()
  for i, (stpt, imc) in enumerate(train_loader):
    print('Training iteration {}'.format(i))
    
    # Use GPU if available
    if use_gpu: 
        print('Using GPU!')
        stpt, imc = stpt.cuda(), imc.cuda()
    else:
        print('Not using GPU!')

    # Record time to load data (above)
    data_time.update(time.time() - end)

    # Run forward pass
    imc_recons = model(stpt.double())   # DEBUG: make model dimensions work for stpt (currently still using dimensions from tutorial)
    loss = criterion(imc_recons, imc) 
    losses.update(loss.item(), stpt.size(0))

    # Compute gradient and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Record time to do forward and backward passes
    batch_time.update(time.time() - end)
    end = time.time()

    # Print model accuracy -- in the code below, val refers to value, not validation
    if i % 25 == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
              epoch, i, len(train_loader), batch_time=batch_time,
             data_time=data_time, loss=losses)) 

  print('Finished training epoch {}'.format(epoch))


# Move model and loss function to GPU
if use_gpu: 
  criterion = criterion.cuda()
  model = model.cuda()

best_losses = 1e10
epochs = 20

# Train model
for epoch in range(epochs):
  # Train for one epoch, then validate
  train(train_loader, model, criterion, optimizer, epoch)
  with torch.no_grad():
    losses = validate(val_loader, model, criterion, epoch)
  # Save checkpoint and replace old best model if current model is better
  if losses < best_losses:
    best_losses = losses
    torch.save(model.state_dict(), 'checkpoints/model-epoch-{}-losses-{:.3f}.pth'.format(epoch+1,losses))




