from skimage import io
from io import BytesIO
import matplotlib.pyplot as plt
import os
import math
from pathlib import Path
import re
import ipywidgets as widgets
from IPython.display import display


def save_stpts(ress=['1'], phys_secs=['001'], opt_secs=['00']):
    """
    res: list of resolutions
    phys_secs: list of physical sections
    opt_secs: list of optical sections
    """
    
    # we will iterate over every possible combination of the elements in each list
    for res in ress:  # iterate over resolutions 
        os.chdir('/home/jimaxt/Shared/Notebooks/Kevin/stpt_2_imc/data/STPT-191127-1125/' + str(res) +'/')

        for phys_sec in phys_secs:  # iterate over physical sections
            for opt_sec in opt_secs:  # iterate over optical sections
                file_name = 'S' + phys_sec + '_Z' + opt_sec + '.tif'
                img = io.imread(file_name)
                plt.title(file_name)
                plt.imshow(img, cmap='gray')
                
                img_file_name = Path(file_name).stem + ".png"
                path = '/home/jimaxt/Shared/Notebooks/Kevin/stpt_2_imc/data/STPT/{}'.format(res)
                Path(path).mkdir(parents=True, exist_ok=True)  # make directories if they do not exist
                plt.savefig('{0}/{1}'.format(path, img_file_name))
                plt.clf()
                
def save_imcs(secs=['01'], img_types=['TIF'], data_files=['131Xe.tif']):
    """
    secs: list of sections
    img_types: what type of image to save (this will determine which directory to pull images from)
    opt_secs: list of optical sections
    """
    os.chdir('/home/jimaxt/Shared/Notebooks/Kevin/stpt_2_imc/data/PROCESSED_IMC_DATA/')
    for sec in secs:
        pattern = re.compile('SECTION_{}_MATCH_STPT_*'.format(sec))
        imc_sec_dirs = [f for f in os.listdir('.') if (os.path.isdir(f) and pattern.match(f))]
        for imc_sec_dir in imc_sec_dirs: 
            for img_type_dir in img_types:
                for data_file in data_files:
                    file_name = imc_sec_dir + '/{}/'.format(img_type_dir) + data_file
                    img = io.imread(file_name)
                    plt.title(file_name)
                    plt.imshow(img)
                    img_file_name = Path(file_name).stem + ".png"
                    path = '/home/jimaxt/Shared/Notebooks/Kevin/stpt_2_imc/data/IMC/{}'.format(sec)
                    Path(path).mkdir(parents=True, exist_ok=True)  # make directories if they do not exist
                    plt.savefig('{0}/{1}'.format(path, img_file_name))
                    plt.clf()

                    
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