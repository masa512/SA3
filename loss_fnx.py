import perceptual
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def MSE_wrapper(A,B):
  crit = nn.L1Loss()
  return crit(A,B)


def Perc_wrapper(I,Ipred,crit):
  L = crit(I,Ipred,idx=[0], norm=True)
  return L

def TVL(img):
  bs_img, c_img, h_img, w_img = img.size()
  tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
  tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
  return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)


def loss_wrapper(F,Fpred, crit,M,weights = [1,1,1,1,1]):

  # Define Ifuse image
  Ffuse = (1-M)*Fpred + M*F
  Ifuse = torch.real(torch.fft.ifft2(torch.fft.ifftshift(Ffuse[:,0:1,:,:]+1j*Ffuse[:,1:,:,:])))
  I = torch.real(torch.fft.ifft2(torch.fft.ifftshift(F[:,0:1,:,:]+1j*F[:,1:,:,:])))

  # Defile Ipred image
  Ipred = torch.real(torch.fft.ifft2(torch.fft.ifftshift(Fpred[:,0:1,:,:]+1j*Fpred[:,1:,:,:])))

  # Part 1 
  loss1 = weights[0] * crit(Fpred[:,:1,:,:],F[:,:1,:,:],idx=[0], norm=False)

  # Part 2
  loss2 = weights[1] * crit(Ffuse[:,:1,:,:],F[:,:1,:,:],idx=[0], norm=True)

  # Part 3 
  loss3 = weights[2] * crit(Fpred[:,1:,:,:],F[:,1:,:,:],idx=[0], norm=True)

  # Part 4
  loss4 = weights[3] * crit(Ffuse[:,1:,:,:],F[:,1:,:,:],idx=[0], norm=True)
  
  # Part 5
  loss5 = weights[4] * MSE_wrapper(Fpred*(M),F*(M))

  # Part 6
  loss6 = weights[5] * MSE_wrapper(Fpred*(1.0-M),F*(1.0-M))

  return loss1,loss2,loss3,loss4,loss5,loss6

  

