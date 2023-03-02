# -*- coding: utf-8 -*-
"""Untitled6.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Gmw0r_SjDstcYzXzJ-z9e_zjpbWAm6z2
"""

import matplotlib.pyplot as plt
from torchvision.transforms import functional
from PIL import Image
import torch
import torch.autograd as autograd
import os
import torch.nn as nn
import time as time
import numpy as np
from IPython import display
from skimage.color import lab2rgb

##################################################

def generate_images(model, input_val, target_val, num, device):
  input_val_l = input_val.to(device)
  target_val_ab = target_val.to(device)

  model.eval()
  with torch.no_grad():
    test_input = torch.unsqueeze(input_val_l,0).to(device)
    prediction = model(test_input).detach()

  val_l = (input_val_l + 1) * 50
  val_ab = target_val_ab * 128
  val_Lab = torch.cat([val_l, val_ab], dim=0).permute(1,2,0).cpu().numpy()
  val_rgb= lab2rgb(val_Lab)
 
  prediction_ab = torch.squeeze(prediction,0) * 128
  prediction_Lab = torch.cat([val_l, prediction_ab], dim=0).permute(1,2,0).cpu().numpy()
  prediction_rgb= lab2rgb(prediction_Lab)
  plt.figure(figsize=(15, 15))
  display_list = [val_l, val_rgb, prediction_rgb]
  title = ['Input Image', 'Ground Truth', 'colorized Image']

  for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      image = display_list[i]
      if i==0:
        plt.imshow(torch.squeeze(image,0).cpu(), cmap = 'gray')
      else:
        plt.imshow(image)
      plt.axis('off')
  script_dir = os.path.dirname('./')
  results_dir = os.path.join(script_dir, 'image_folder/')
  if not os.path.isdir(results_dir):
      os.makedirs(results_dir)


  plt.savefig(results_dir +'image_{:04d}.png'.format(num))
  plt.show()
  model.train()

####################################################

LAMBDA = 10

def generator_loss(critic_generated_output):
  gen_loss = -1 *torch.mean(critic_generated_output)
  return gen_loss

def critic_loss(critic_real_output, critic_generated_output, g_p):
  real_loss = torch.mean(critic_real_output)
  generated_loss = torch.mean(critic_generated_output)
  total_critic_loss = generated_loss - real_loss +  + (LAMBDA * g_p)
  return total_critic_loss

def gradient_penalty(critic, input_image, target, gen_output, device):
  batch_size = input_image.shape[0]
  epsilon = torch.rand(batch_size, 1,1,1).to(device)
  interpolates = (epsilon * target + ((1 - epsilon) * gen_output)).requires_grad_(True)
  c_interpolates = critic(input_image, interpolates)
  gradients = autograd.grad(outputs=c_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(c_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True)[0]
  gradients = gradients.view(gradients.size(0), -1)
  gradients_norm = gradients.norm(2, dim=1)
  g_p = torch.mean((gradients_norm - 1) ** 2)   
  return g_p

######################################################

def critic_train(input_image, target, generator, critic, critic_opt, device):
  generator.eval()
  critic.train()

  critic_opt.zero_grad()
  gen_output = generator(input_image).detach()
  critic_generated_output = critic(input_image, gen_output)
  critic_real_output = critic(input_image, target)
  g_p = gradient_penalty(critic, input_image, target, gen_output, device)
  c_loss = critic_loss(critic_real_output, critic_generated_output, g_p)
  c = c_loss/3
  c_loss.backward(retain_graph=True)
  critic_opt.step()
  return c
  


def generator_train(input_image, target, generator, critic, gen_opt):
  critic.eval()
  generator.train()
  
  gen_opt.zero_grad()
  gen_output = generator(input_image) 
  critic_generated_output = critic(input_image, gen_output)
  gen_loss = generator_loss(critic_generated_output) 
  gen_loss.backward()
  gen_opt.step()
  return gen_loss


def valid_step(input_image, target, generator, critic, device):

  generator.eval()
  critic.eval()
  with torch.no_grad():

    gen_output = generator(input_image)
    critic_generated_output = critic(input_image, gen_output.detach())
    critic_real_output = critic(input_image, target)

    gen_loss = generator_loss(critic_generated_output)
  g_p = gradient_penalty(critic, input_image, target, gen_output, device)
  c_loss = critic_loss(critic_real_output, critic_generated_output, g_p)
  generator.train()
  critic.train()
  return gen_loss, c_loss

##################################################################

n_critic = 3

def train(data_train, data_val, val_visual, epochs, generator, critic, critic_opt, gen_opt, device):
  gloss_train = []
  closs_train = []

  gloss_val = []
  closs_val = []
  
  start_koli = time.time()
  for epoch in range(epochs):
    start = time.time()

    gen_losses_t = []
    critic_losses_t = []

    gen_losses_v = []
    critic_losses_v = []
    
    for input_image,target  in data_train:
      input_image = input_image.to(device)
      target = target.to(device)
      
      c_l_mean_b = 0
      for j in range(n_critic):
        critic_loss_t_b = critic_train(input_image, target, generator, critic, critic_opt, device)
        c_l_mean_b+=critic_loss_t_b

      gen_loss_t = generator_train(input_image, target, generator, critic, gen_opt)

      gen_losses_t.append(gen_loss_t.detach().cpu())
      critic_losses_t.append(c_l_mean_b.detach().cpu())

    for input_image,target  in data_val:
      input_image = input_image.to(device)
      target = target.to(device)
      gen_loss_v, critic_loss_v = valid_step(input_image, target, generator, critic, device)
      gen_losses_v.append(gen_loss_v.detach().cpu())
      critic_losses_v.append(critic_loss_v.detach().cpu())

    gloss_train.append(np.mean(gen_losses_t))
    closs_train.append(np.mean(critic_losses_t))

    gloss_val.append(np.mean(gen_losses_v))
    closs_val.append(np.mean(critic_losses_v))    

    #Produce images
    #display.clear_output(wait=True)
    if epoch%5 == 0:
      generate_images(generator, val_visual[0][0], val_visual[0][1], epoch,device)
      print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    

  # Generate after the final epoch
  #display.clear_output(wait=True)
  generate_images(generator, val_visual[0][0], val_visual[0][1], epoch,device)
  print ('Total runtime for {} epoches is {} sec'.format(epoch + 1, time.time()-start_koli))
  total_time = time.time()-start_koli 
  torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': critic.state_dict(),

            'gen_opt_state_dict': critic_opt.state_dict(),
            'dis_opt_state_dict': gen_opt.state_dict(),

            'gloss_train': gloss_train,
            'dloss_train': closs_train,

            'gloss_val': gloss_val,
            'dloss_val': closs_val,
  
            
            'Total_time' : total_time

            
            }, 'model_orig.tar')
  
  return gloss_train, closs_train , gloss_val, closs_val
