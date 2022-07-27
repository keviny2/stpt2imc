import os
import torch

def save_imc(phys_sec, grid, row, column):
    # creates directory if doesn't exist
    os.makedirs('processed_data/IMC/{0}'.format(str(phys_sec).zfill(2)), exist_ok=True)
    torch.save(grid[row][column].double(), 'processed_data/IMC/{0}/{1}_{2}.pt'.format(str(phys_sec).zfill(2),
                                                                             str(row).zfill(2),
                                                                             str(column).zfill(2)))
    
def save_stpt(phys_sec, grid, row, column):
    # creates directory if doesn't exist
    os.makedirs('processed_data/STPT/{0}'.format(str(phys_sec).zfill(2)), exist_ok=True)    
    torch.save(grid[row][column].clone().double(), 'processed_data/STPT/{0}/{1}_{2}.pt'.format(str(phys_sec).zfill(2),
                                                                             str(row).zfill(2),
                                                                             str(column).zfill(2)))
    
class AverageMeter(object):
    '''A handy class from the PyTorch ImageNet tutorial''' 
    def __init__(self):
        self.reset()
        self.vals = []
        self.avgs = []
    def reset(self):
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(self.val)
        self.avgs.append(self.avg)