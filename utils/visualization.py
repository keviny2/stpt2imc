from skimage import io
import numpy as np
import cv2 as cv
import torch
from io import BytesIO
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
import re
import ipywidgets as widgets
from IPython.display import display


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

def save_imc(phys_sec, grid, row, column):
    # creates directory if doesn't exist
    if not(os.path.isdir('processed_data/IMC/{0}'.format(str(phys_sec).zfill(2)))):
        os.mkdir('processed_data/IMC/{0}'.format(str(phys_sec).zfill(2)))
    torch.save(grid[row][column].double(), 'processed_data/IMC/{0}/{1}_{2}.pt'.format(str(phys_sec).zfill(2),
                                                                             str(row).zfill(2),
                                                                             str(column).zfill(2)))
    
def save_stpt(phys_sec, grid, row, column):
    # creates directory if doesn't exist
    if not(os.path.isdir('processed_data/STPT/{0}'.format(str(phys_sec).zfill(2)))):
        os.mkdir('processed_data/STPT/{0}'.format(str(phys_sec).zfill(2)))    
    torch.save(grid[row][column].clone().double(), 'processed_data/STPT/{0}/{1}_{2}.pt'.format(str(phys_sec).zfill(2),
                                                                             str(row).zfill(2),
                                                                             str(column).zfill(2))) 
                    
def display_img(file, cmap='gray'):
    img = io.imread(file)
    plt.imshow(img, cmap=cmap)
    
        
def display_imgs(folder):
    """
    render images in grid format 
    
    folder: folder containing images to render
    """
    # Define a useful function
    def get_image(f_path):
        '''
        Returns the image from a path
        '''
        img_labs = ['jpg','png']
        if any(x in img_labs for x in f_path.split('.')):
            file = os.path.join(folder, f_path)
            image = open(file, 'rb').read()
            return image
    
    # Do the actual work here
    files  = os.listdir(folder)
    images = [get_image(x) for x in files]
    children = [widgets.Image(value = img, format='png') for img in images if str(type(img)) != '<class \'NoneType\'>']
    
    nrows = math.ceil(math.sqrt(len(children)))
    template = '50% ' * nrows

    # Create the widget
    grid = widgets.GridBox(children=children,
                           layout=widgets.Layout(
                               width='70%',
                               grid_template_columns=template,
                               grid_template_rows=template,
                               grid_gap='18px 2px'
                           )
                          )
    display(grid)