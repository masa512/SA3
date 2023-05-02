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


def loss_wrapper(I,F,Fpred,crit,M):
  weight = [1,1,1,1,1]

  # Define Ifuse image
  Ffuse = M*Fpred + (1-M)*F
  Ifuse = torch.real(torch.fft.ifft2(torch.fft.ifftshift(Ffuse)))

  # Defile Ipred image
  Ipred = torch.fft.ifft2(torch.fft.ifftshift(Fpred))

  # Part 1 
  loss1 = weight[0] * Perc_wrapper(Ifuse, I)

  # Part 2
  loss2 = weight[1] * Perc_wrapper(Ipred,I)

  # Part 3 
  loss3 = weight[2] * MSE_wrapper(Ipred,I)

  # Part 4
  loss4 = weight[3] * TVL(Ffuse)
  
  # Part 5
  loss5 = weight[4] * TVL(Fpred)

  return loss1,loss2,loss3,loss4,loss5

  
