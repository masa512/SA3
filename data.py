
 

import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from torch.utils.data import Dataset
import SA_sample

def read_image(image_path = ''):
    """
    Wrapper function to read image and raw image
    """
    I = plt.imread(image_path)
    # Rescale to 0-1
    I = I/255

    return I

class simple_dataset(Dataset):
  
  def __init__(self,path = 'data', mode = 'train'):
    self.path = os.path.join(path,mode)
    self.file_list = [fn for fn in os.listdir(self.path) if fn.endswith('.jpg')]
  
  def __len__(self):
    return len(self.file_list)

  def __getitem__(self,idx):
    
    # Image Domain
    image_path = os.path.join(self.path,self.file_list[idx])
    I = np.mean(read_image(image_path),axis=-1,keepdims=False)
    M,N = I.shape
    I = torch.Tensor(I).reshape(1,M,N).float()
    
    # Fourier Transform (2 Channel)
    F = torch.fft.fftshift(torch.fft.fft2(I))
    F = torch.concatenate([torch.real(F),torch.imag(F)],dim=0)
    
    # SA Fourier (2 channel)
    F_SA,M = SA_sample.synthetic_sample(F,5,30)

    return (I,F,F_SA,M.float())
    
    

