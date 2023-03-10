# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18co1ULTOXVib8B3ulYbvU4LSGo1mPtIq
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchdata.datapipes.iter import FileLister
import random
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.transforms import functional
from torchvision.transforms import Resize
from PIL import Image
from skimage import color
from IPython import embed
from skimage.color import rgb2lab
from fastai.data.external import untar_data, URLs

##function for load and save the data pathes /// 
## if saving = True, function use the saved files otherwise it will load and save them
def imagenet_path( PATH, number_of_data, saving = True, file_name = "train.npy"):  
  if saving:
    with open(file_name, 'rb') as f:
        path_list_saved = np.load(f)
        random.shuffle(path_list_saved)
        path_list_short = path_list_saved[0: number_of_data]
    list_final = path_list_short

  else:
    total_path = FileLister(root=PATH, recursive=True)
    path_list=list(total_path)
    random.shuffle(path_list)
    path_list_short = path_list[0: number_of_data]
    list_final = path_list_short
    with open(file_name, 'wb') as f:
      np.save(f, path_list)

  return list_final  

def coco_path(num_train, num_val):
  path = untar_data(URLs.COCO_SAMPLE)
  train_path = str(path) + "/train_sample"
  total_path = FileLister(root=train_path, recursive=True)
  path_list=list(total_path)
  np.random.seed(123)
  path_list_10k = np.random.choice(path_list, 10000, replace=False)
  random.shuffle(path_list_10k)
  path_list_train = path_list_10k[0: num_train]
  path_list_val = path_list_10k[num_train:(num_val+num_train)]
  return path_list_train, path_list_val

class preprocess(Dataset):
    def __init__(self, paths_list, train_mode = True):

        self.transToTensor = transforms.Compose([transforms.ToTensor()])
        if train_mode:
            self.transforms = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.RandomHorizontalFlip()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((256,256))
            ])

        self.paths_list = paths_list    
  
              
    def __getitem__(self, idx):
      img = Image.open(self.paths_list[idx]).convert("RGB")
      img_T = self.transforms(img)
      img_arr = np.array(img_T)
      img_lab_arr = rgb2lab(img_arr).astype("float32") # Converting RGB to L*a*b
      img_lab_tensor = self.transToTensor(img_lab_arr)
      l_tensor = img_lab_tensor[[0], ...] 
      ab_tensor = img_lab_tensor[[1, 2], ...]   

        #normalizing between [-1,1]
      img_orig = (ab_tensor / 128) #a & b ranges are [-128,127]
      img_input = (l_tensor / 50) - 1 #l range is [0,100]
      return  img_input, img_orig 

    def __len__(self):
         return len(self.paths_list) 


def data(number_of_data, saving, batch_size, dataset):

  if dataset == "imagenet":
    PATH_train = '/content/drive/MyDrive/imagenet-mini/train'
    PATH_vt = '/content/drive/MyDrive/imagenet-mini/val'
    path_train = imagenet_path( PATH_train, number_of_data[0], saving[0], file_name = "train.npy")
    path_vt = imagenet_path( PATH_vt, number_of_data [1], saving[1], file_name = "val_test.npy")
  if dataset == "coco":
    path_train, path_vt = coco_path(number_of_data[0], number_of_data[1])
    
  path_test = path_vt[0:20]
  path_val = path_vt[20:]
  
  datset_tensor_train = preprocess(path_train)
  datset_tensor_val = preprocess(path_val, train_mode = False)
  datset_tensor_test = preprocess(path_test, train_mode = False)
  
  
  train_dataloader = DataLoader(datset_tensor_train, batch_size[0], shuffle=True,drop_last=True, num_workers=os.cpu_count())
  val_dataloader = DataLoader(datset_tensor_val, batch_size[1], shuffle=True,drop_last=True, num_workers=os.cpu_count())

  return train_dataloader, val_dataloader, datset_tensor_val, datset_tensor_test

