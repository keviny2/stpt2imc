import os
from torchvision import datasets
import torch
import numpy as np

class STPT_IMC_ImageFolder(datasets.ImageFolder):    
    """
    Preprocesses
    """
    def __init__(self, root, bits=8, batch_size=64):
        self.root = root
        self.imc_folder = os.path.join(self.root, 'IMC')
        self.stpt_folder = os.path.join(self.root, 'STPT')
        self.bits = bits # num bits for each pixel in image
        self.batch_size = batch_size
        
        # map each physical section folder with number of files in it
        self.num_imgs_per_phys_sec = {}
        acc = 0
        for dirpath, dirnames, filenames in os.walk(self.imc_folder):
            if os.path.basename(dirpath) in ['.ipynb_checkpoints', '.']:
                continue
            self.num_imgs_per_phys_sec[os.path.basename(dirpath)] = len(filenames) + acc
            acc += len(filenames)
        
        self.num_imgs = acc
    
    def get_phys_sec(self, index):
        """Determine which physical section folder index resides
        
        """
        for key in list(self.num_imgs_per_phys_sec.keys()):
            if index < self.num_imgs_per_phys_sec[key]:
                return key
            
    def get_chunk_idx(self, index):
        "
        prev = 0
        for key in list(self.num_imgs_per_phys_sec.keys()):
            if index < self.num_imgs_per_phys_sec[key]:
                return index - prev
            prev = self.num_imgs_per_phys_sec[key]
            
    def __len__(self):
        return self.num_imgs
        
    def __getitem__(self, index):
        
        phys_sec = self.get_phys_sec(index)
                                                         
        # ====== GET LIST OF IMAGE FILES ======
        stpt_imgs = sorted(os.listdir(os.path.join(self.stpt_folder, '{}'.format(str(phys_sec).zfill(2)))))
                                                         
        imc_imgs = sorted(os.listdir(os.path.join(self.imc_folder, '{}'.format(str(phys_sec).zfill(2)))))
        
        # ====== GET IMAGE FILE PATH ======
        stpt_path = os.path.join(self.stpt_folder,
                                 '{}'.format(str(phys_sec).zfill(2)),
                                 stpt_imgs[self.get_chunk_idx(index)])
        
        imc_path = os.path.join(self.imc_folder,
                                          '{}'.format(str(phys_sec).zfill(2)),
                                          imc_imgs[self.get_chunk_idx(index)])

        # make sure the files line up
        try:
            assert(os.path.basename(stpt_path) == os.path.basename(imc_path))
        except:
            print('stpt path:', os.path.basename(stpt_path))
            print('imc path:', os.path.basename(imc_path))
                                       
        # ====== LOAD IMAGES ======
        stpt_img = torch.load(stpt_path)

        imc_img = torch.load(imc_path)
                                                                     
        return stpt_img, imc_img   