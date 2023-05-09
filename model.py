################# NOTE!!!!!!!! #####################################
# This implementation was inspired by the paper
# https://arxiv.org/pdf/1804.07723.pdf
# And the python implementation was heavily influenced by the previous repository
# https://github.com/tanimutomo/partialconv/blob/master/src/model.py
#####################################################################


import torch
import torch.nn as nn

class partial_conv(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size):
    super(partial_conv,self).__init__()
    self.kernel_size = kernel_size
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.conv = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=kernel_size//2)
    self.kernel = 1/((self.kernel_size)**2*self.in_channels) * torch.ones(self.out_channels,self.in_channels,self.kernel_size,self.kernel_size,requires_grad=False)
    self.kernel = self.kernel.to('cuda')
  def forward(self,x,m,eps=1e-10):
    x_masked = x * m
    y = self.conv(x_masked)
    # Pool the mask
    m_pool = nn.functional.conv2d(m,self.kernel,padding=self.kernel_size//2)
    y = y/(m_pool+eps)
    # update the mask
    m_pool[m_pool>0] = 1
    return y,m_pool

class double_conv(nn.Module):
  # Conv2d, Batch_norm, ReLU twice
  def __init__(self,in_channels,out_channels,kernel_size):
    super(double_conv,self).__init__()
    self.kernel_size = kernel_size
    self.in_channels = in_channels
    self.out_channels = out_channels

    self.conv1 = partial_conv(self.in_channels,self.out_channels,self.kernel_size)
    self.conv2 = partial_conv(self.out_channels,self.out_channels,self.kernel_size)
    self.bn1 = nn.BatchNorm2d(self.out_channels)
    self.bn2 = nn.BatchNorm2d(self.out_channels)
    self.relu = nn.ReLU()

  def forward(self,x,m):
    # First conv bn relu
    y1,m1 = self.conv1(x,m)
    y1 = self.bn1(y1)
    y1 = self.relu(y1)

    # Second conv bn relu
    y2,m2 = self.conv2(y1,m1)
    y2 = self.bn2(y2)
    y2 = self.relu(y2)

    return y2,m2

class upsample_conv(nn.Module):
  def __init__(self,in_channels,out_channels):
    super(upsample_conv,self).__init__()
    self.upsample = torch.nn.Upsample(scale_factor=2)
    #self.cconv = partial_conv(in_channels,out_channels,1)
  
  def forward(self,x,m):

    # Upsample the x,m
    x = self.upsample(x)
    m = self.upsample(m)
    #y1,m1 = self.cconv(x,m)

    return x,m
  
class out_conv(nn.Module):
  def __init__(self,in_channels,num_classes):
    super(out_conv,self).__init__()
    self.in_channels = in_channels
    self.num_classes = num_classes
    #self.softmax = nn.Softmax(dim=1)
    self.conv = partial_conv(self.in_channels,self.num_classes,kernel_size=1)

  def forward(self,x,m):
    y1,m1 = self.conv(x,m)
    return y1


class encoder_block(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size):
    super(encoder_block,self).__init__()
    self.double_conv = double_conv(in_channels,out_channels,kernel_size)
    self.pool = nn.MaxPool2d(2,2)
  
  def forward(self,m,x):
    # Start with a pooling layer
    y1 = self.pool(m)
    m1 = self.pool(m)

    # Now the double conv
    y2,m2 = self.double_conv(y1,m1)

    return y2,m2 

class decoder_block(nn.Module):
  def __init__(self,in_channels,out_channels,kernel_size):
    super(decoder_block,self).__init__()
    self.upsample = upsample_conv(in_channels,out_channels)
    self.dconv = double_conv(in_channels,out_channels,3)
  
  def forward(self,x,m,ex,em):
    # Upsample x,m 
    x1,m1 = self.upsample(x,m)
    # Concatenate x and m, respectively
    X = torch.concatenate([x1,ex],dim=1)
    M = torch.concatenate([m1,em],dim=1)
    # Apply double conv to finish off
    y,m = self.dconv(X,M)
    return y,m

class PUnet(nn.Module):
  def __init__(self,in_channels,base_channels,out_channels):
    super(PUnet,self).__init__()
    self.in_channels = in_channels
    self.base_channels = base_channels
    self.num_features = out_channels

    self.input_layer = double_conv(self.in_channels,self.base_channels,3)
    self.encoder1 = encoder_block(1*self.base_channels,2*self.base_channels,3)
    self.encoder2 = encoder_block(2*self.base_channels,4*self.base_channels,3)
    self.encoder3 = encoder_block(4*self.base_channels,8*self.base_channels,3)
    self.encoder4 = encoder_block(8*self.base_channels,16*self.base_channels,3)

    self.decoder1 = decoder_block(16*self.base_channels,8*self.base_channels,3)
    self.decoder2 = decoder_block(8*self.base_channels,4*self.base_channels,3)
    self.decoder3 = decoder_block(4*self.base_channels,2*self.base_channels,3)
    self.decoder4 = decoder_block(2*self.base_channels,1*self.base_channels,3)
    self.output_layer = out_conv(1*self.base_channels,self.num_features)

  
  def forward(self,x,m):
    x1,m1 = self.input_layer(x,m)
    x2,m2 = self.encoder1(x1,m1)
    x3,m3 = self.encoder2(x2,m2)
    x4,m4 = self.encoder3(x3,m3)
    ex,em = self.encoder4(x4,m4)

    x5,m5 = self.decoder1(ex,em,x4,m4)
    x6,m6 = self.decoder2(x5,m5,x3,m3)
    x7,m7 = self.decoder3(x6,m6,x2,m2)
    x8,m8 = self.decoder4(x7,m7,x1,m1)
    y = self.output_layer(x8,m8)

    return y


