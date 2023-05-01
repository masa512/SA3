import torch 
import torch.nn as nn
from torchvision.models import vgg16
"""
USAGE OF PERCEPTUAL LOSS 
We will use the pretrained VGG16 but with frozen layers (only used for forward)
We only will use the convolutional part (vgg16.features())
The following blocks of VGG16 are defined
Block 1 : Layers 0-3
Block 2 : Layers 4-8
Block 3 : Layers 9-15
Block 4 : Layers 16-22
"""

class perceptual_loss(nn.Module):
  
  def __init__(self,crit = 'L1'):
    super(perceptual_loss,self).__init__()
    self.vgg_conv = vgg16(pretrained=True).features
    self.vgg_conv.requires_grad = False
    
    p = [0,4,9,16,23]
    self.blocks = [self.vgg_conv[p[i]:p[i+1]] for i in range(len(p)-1)]

    # Define average and std as side variable (buffers are not part of model parameters)
    # These parameters were already set using the 
    self.register_buffer(name='mean', tensor = torch.Tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
    self.register_buffer(name='sigma', tensor = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

    # error fnx
    if crit == 'L2':
      self.crit = nn.MSELoss()
    else:
      self.crit = nn.L1Loss()


  def forward(self,Igt,Ipred,idx=[0],norm=False):
    
    if norm:
      Igt = (Igt - self.mean)/self.sigma
      Ipred = (Ipred - self.mean)/self.sigma
    
    if Igt.shape[1] != 3:
      Igt = Igt.repeat(1,3,1,1)
      Ipred = Ipred.repeat(1,3,1,1)


    # Save losses separately
    max_idx = max(idx)
    losses = 0
    hgt = Igt
    hpred = Ipred
    for i in range(max_idx+1):
      hgt = self.blocks[i](hgt)
      hpred = self.blocks[i](hpred)
      if i in idx:
        losses = losses + self.crit(hgt,hpred)

    # Sum of the losses
    return losses
