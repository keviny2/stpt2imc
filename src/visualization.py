from skimage import io
from io import BytesIO
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
import re
import ipywidgets as widgets
from IPython.display import display
                    
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