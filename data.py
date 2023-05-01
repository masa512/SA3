
 

import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import .data as data

def read_image(image_path = ''):
    """
    Wrapper function to read image and raw image
    """
    I = plt.imread(image_path)
    # Rescale to 0-1
    I = I/255

    return I

class simple_dataset(Dataset):
  
  def __init__(self,path = 'data', mode = 'train', box_dim = (50,50)):
    self.path = os.path.join(path,mode)
    self.file_list = [fn for fn in os.listdir(self.path) if fn.endswith('.jpg')]
    self.box_dim = box_dim
  
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self,idx):
    
    # Image Domain
    image_path = os.path.join(self.path,self.file_list[idx])
    I = np.mean(read_image(image_path),axis=-1,keepdims=False)
    
    # Fourier Transform
    F = torch.fft.fft2(I)
    
    I = torch.Tensor(I).view(1,I.shape[1],I.shape[0]).float()
    
    
    
    # Masking
    
    
