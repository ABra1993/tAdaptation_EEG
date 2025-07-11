import torch as torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

def sequence_test(imgs, value, t_steps, t_steps_label):

    # initiate ax
    ax = []
    # ax = torch.ones(batch_size, t_steps, imgs.shape[1], imgs.shape[2], imgs.shape[3])*0.5

    # create noise
    noise = [torch.empty((imgs.shape[1], imgs.shape[2], imgs.shape[3])).normal_(0, 1) for _ in range(imgs.shape[0])] # gaussian noise
    noise = torch.stack(noise)

    # shift
    shifted_noise = torch.roll(noise, shifts=(-value, 0), dims=(3, 2))

    # initiate figure
    for t in range(t_steps):

        if t_steps_label[t] == 0:               # blank
        
            blank = torch.ones(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])*0.5
            ax.append(blank)

        elif t_steps_label[t] == 1:               # adapter

            noise_clamp = torch.clamp(noise, 0, 1)
            ax.append(noise_clamp)

        elif t_steps_label[t] == 2:             # test

            target_clamp = torch.clamp(shifted_noise + imgs, 0, 1)
            ax.append(target_clamp)

    return ax

