# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1k3UOqytvyHSaycqgWebe6THrzaKemoEc
"""

import torch
import torch.nn as nn


class downsample(nn.Module): 

  def __init__(self, input_cha, output_cha, size, apply_batchnorm=True):
    super().__init__()

    if apply_batchnorm:
      self.model = nn.Sequential(
          nn.Conv2d(input_cha, output_cha, size, bias=False, stride=2, padding=1),
          nn.BatchNorm2d((output_cha)),
          nn.LeakyReLU(0.2,True)
      )
    else:
      self.model = nn.Sequential(
          nn.Conv2d(input_cha, output_cha, size, bias=False, stride=2, padding=1),
          nn.LeakyReLU(0.2,True)
      )  

  def forward(self, x):
        y = self.model(x)
        return y

class upsample(nn.Module): 

  def __init__(self, input_cha, output_cha, size, apply_dropout=False):
    super().__init__()

    if apply_dropout:
      self.model = nn.Sequential(
          nn.ConvTranspose2d(input_cha, output_cha, 4, bias=False, stride=2, padding=1),
          nn.BatchNorm2d((output_cha)),
          nn.Dropout(0.5),
          nn.ReLU(True)
      )
    else:
      self.model = nn.Sequential(
          nn.ConvTranspose2d(input_cha, output_cha, 4, bias=False, stride=2, padding=1),
          nn.BatchNorm2d((output_cha)),
          nn.ReLU(True)
      )  

  def forward(self, x):
        y = self.model(x)
        return y  

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        #nn.init.xavier_uniform_(m.weight.data)
    if classname.find('Linear') != -1:
        #nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.xavier_uniform_(m.weight.data)
    if classname.find('Norm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)    
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class U_net(nn.Module):

   def __init__(self):
        super().__init__()

        self.l1 = downsample(1, 64, 4, apply_batchnorm=False)  # (batch_size, 64, 128, 128)
        self.l2 = downsample(64, 128, 4)  # (batch_size, 128, 64, 64)
        self.l3 = downsample(128, 256, 4)  # (batch_size, 256, 32, 32)
        self.l4 = downsample(256, 512, 4)  # (batch_size, 512, 16, 16)

        
        self.l5 = upsample(512, 256, 4, apply_dropout=True)  # (batch_size, 512, 32, 32)
        self.l6 = upsample(512, 128, 4)  # (batch_size, 256, 64, 64)
        self.l7 = upsample(256, 64, 4)  # (batch_size, 128, 128, 128)

        self.last = nn.Sequential(
          nn.ConvTranspose2d(128, 2, 4, stride=2, padding=1),# (batch_size, 2, 256, 256)
          nn.Tanh()  )
    

   def forward(self,x):

     l1 = self.l1(x)
     l2 = self.l2(l1)
     l3 = self.l3(l2)
     l4 = self.l4(l3)

     l5 = self.l5(l4)
     cat1 = torch.cat([l3, l5], dim=1)
     l6 = self.l6(cat1)
     cat2 = torch.cat([l2, l6], dim=1)
     l7 = self.l7(cat2)
     cat3 = torch.cat([l1, l7], dim=1)
    
     output = self.last(cat3)

     return output
