import torch as torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

def sequence_test(ax, imgs, value, t_steps, t_steps_label):

    # create noise
    noise = [torch.empty((ax.shape[2], ax.shape[3], ax.shape[4])).normal_(0, 1) for _ in range(ax.shape[0])] # gaussian noise
    noise = torch.stack(noise)

    # shift
    shifted_noise = torch.roll(noise, shifts=(-value, 0), dims=(3, 2))


    # initiate figure
    for t in range(t_steps):
        if t_steps_label[t] == 1:               # adapter

            noise_clamp = torch.clamp(noise, 0, 1)
            ax[:, t, :, : , :] = noise_clamp

        elif t_steps_label[t] == 2:             # test

            target_clamp = torch.clamp(shifted_noise + imgs, 0, 1)
            ax[:, t, :, : , :] = target_clamp

    return ax

