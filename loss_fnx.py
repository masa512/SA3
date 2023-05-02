import perceptual
import torch
import torch.nn as nn
import numpy as np


def MSE_wrapper(A,B):
  crit = nn.MSELoss()
  return crit(A,B)


def Perc_wrapper(I,Ipred,crit):
  L = crit(I,Ipred,idx=[0], norm=True)
  return L

def TVL(img):
  bs_img, c_img, h_img, w_img = img.size()
  tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
  tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
  return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


def loss_wrapper(I,F,Fpred,crit,M,weights = [1,1,1,1,1]):

  # Define Ifuse image
  Ffuse = (1-M)*Fpred + M*F
  Ifuse = torch.real(torch.fft.ifft2(torch.fft.ifftshift(Ffuse[:,0:1,:,:]+1j*Ffuse[:,1:,:,:])))

  # Defile Ipred image
  Ipred = torch.real(torch.fft.ifft2(torch.fft.ifftshift(Fpred[:,0:1,:,:]+1j*Fpred[:,1:,:,:])))

  # Part 1 
  loss1 = weights[0] * Perc_wrapper(Ifuse,I,crit)

  # Part 2
  loss2 = weights[1] * Perc_wrapper(Ipred,I,crit)

  # Part 3 
  loss3 = weights[2] * MSE_wrapper(Ipred,I)

  # Part 4
  loss4 = weights[3] * TVL(Ffuse)
  
  # Part 5
  loss5 = weights[4] * TVL(Fpred)

  # Part 6
  loss6 = weights[5] * MSE_wrapper(Fpred,F)
  return loss1,loss2,loss3,loss4,loss5,loss6

  

