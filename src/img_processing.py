from utils import save_imc, save_stpt

import os, glob
import numpy as np
from skimage import io
import cv2 as cv
import torch
from torch.multiprocessing import Pool
from torchvision import transforms

def process_stpt_image(file_name):
    img = io.imread(file_name)
    
    # normalize image (8 bits)
    norm_img = img.copy()
    cv.normalize(img, norm_img, alpha=0, beta=2**8 - 1, norm_type=cv.NORM_MINMAX)

    # Apply log transformation method
    c = (2**8 - 1) / np.log(1 + np.max(norm_img))

    log_image = c * (np.log(norm_img + 1))
    # Specify the data type so that
    # float value will be converted to int
    return torch.from_numpy(log_image)

def process_imc_image(file_name):
    # read image file
    img = cv.imread(file_name, cv.IMREAD_UNCHANGED)

    # normalize image (8 bits)
    norm_img = img.copy()
    cv.normalize(img, norm_img, alpha=0, beta=2**8 - 1, norm_type=cv.NORM_MINMAX)

    # Apply log transformation method
    c = (2**8 - 1) / np.log(1 + np.max(norm_img))

    log_image = c * (np.log(norm_img + 1))

    # Specify the data type so that
    # float value will be converted to int
    return torch.from_numpy(log_image) 

def crop_imc(phys_sec, grid_size=256, verbose=True):
    '''
    phys_sec = physical section where the image came from
    grid_size = how large each cropped image will be
    
    1. get image paths corresponding to phys_sec
    2. concatenate IMC images within folder to form a single 40-channel IMC image
    3. crop 16 pixels from each side
    4. crop images into 256x256 squares
    5. free up memory
    6. save processed tensors sequentially
    '''
    
    # ====== GET IMAGE PATHS ======
    
    imc_section_folder = os.path.join('/data/meds1_b/msa51/data_endpoint/4T1_STPT_IMC/PROCESSED_IMC_DATA/',
                                      'SECTION_{}*'.format(str(phys_sec).zfill(2)),
                                      'TIF/1*')

    # get a list of all .tif images inside imc_section_folder
    imc_img_paths = [os.path.join(imc_section_folder, imc_img_path)
                     for imc_img_path in glob.glob(imc_section_folder)]
    
    # ====== LOAD IMAGES ======
    with Pool(maxtasksperchild=100) as p:
        imc_imgs = list(p.imap(process_imc_image, imc_img_paths))

    imc_imgs = [torch.unsqueeze(img, 0) for img in imc_imgs] # add an extra dimesion for channel
    imc_imgs_cat = torch.cat(imc_imgs, 0) # (40, 18720, 18720)

    cropped = imc_imgs_cat[:, 16:18704, 16:18704]  # crop 16 pixels from each side (40, 18688, 18688)

    # ====== CONSTRUCT GRID ======
    
    temp = torch.split(cropped, grid_size, dim=1) # row slices; each slice has shape (40, 256, 18688)
    grid = [torch.split(curr, grid_size, dim=2) for curr in temp] # grid is 73x73; each slice has shape (40, 256, 256)

    # ====== FREE UP MEMORY ======
    
    del imc_imgs
    del imc_imgs_cat
    del cropped
    del temp
    
    # ====== SAVE PROCESSED TENSORS ======
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            save_imc(phys_sec, grid, i, j)
    
    if verbose:
        print('IMC: Done physical section:', phys_sec)

def crop_stpt(phys_sec, grid_size=256, verbose=True):
    '''
    phys_sec = physical section where the image came from
    grid_size = how large each cropped image will be
    
    1. get image paths corresponding to phys_sec
    2. concatenate STPT images within folder to form a single 8-channel STPT image
    3. crop 16 pixels from each side
    4. crop images into 256x256 squares
    5. free up memory
    6. save processed tensors sequentially
    '''
    
    # ====== GET IMAGE PATHS ======
    
    stpt_img_paths = [os.path.join('/data/meds1_b/msa51/data_endpoint/4T1_STPT_IMC/STPT_DATA/TIF/STPT-191127-1125/1',
                                   'S{0}_Z{1}.tif'.format(str(phys_sec).zfill(3), optical_section.zfill(2)))
                      for optical_section in ['0', '1']]  
    
    # ====== LOAD IMAGES ======
    stpt_imgs = []
    for path in stpt_img_paths:
        stpt_imgs.append(process_stpt_image(path))
        
    stpt_imgs = [img.permute((2,0,1)) for img in stpt_imgs] # (C,H,W) tensor
    stpt_imgs_cat = torch.cat(stpt_imgs, 0) # concatenate two stpt images (8, 20800, 20800)
    del stpt_imgs

    stpt_imgs_cat = transforms.Resize(18720)(stpt_imgs_cat)  # make STPT img same size as IMC (..., 18720, 18720)
    cropped = stpt_imgs_cat[:, 16:18704, 16:18704]  # crop 16 pixels from each side (8, 18688, 18688)

    # ====== CONSTRUCT GRID ====== 
    temp = torch.split(cropped, grid_size, dim=1) # row slices; each slice has shape (8, 256, 18688)
    grid = [torch.split(curr, grid_size, dim=2) for curr in temp] # grid is 73x73; each slice has shape (8, 256, 256)

    # ====== FREE UP MEMORY ======
    del stpt_imgs_cat
    del cropped
    del temp
    
    # ====== SAVE PROCESSED TENSORS ======
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            save_stpt(phys_sec, grid, i, j) 
    
    if verbose:
        print('STPT: Done physical section:', phys_sec)