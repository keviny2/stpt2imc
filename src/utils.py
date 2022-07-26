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