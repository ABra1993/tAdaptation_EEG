# %%
# import packages
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
from torch import optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random
import seaborn as sns
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

# define root
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# fontsizes 
global fontsize_title
global fontsize_legend
global fontsize_label
global fontsize_tick

fontsize_title          = 13
fontsize_legend         = 10
fontsize_label          = 10
fontsize_tick           = 10

def main():

    # input specification
    dataset                 = 'mnist'

    # noise
    adapters            = ['same', 'different']

    # EEG paper (investigate contrast gain)
    tempDynamics            = ['none', 'add_supp', 'div_norm', 'lat_recurrence',  'lat_recurrence_mult']

    # other settings
    init        = 5
    batch_size  = 100

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # define contrast values
    contrast_values      = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    # training length
    epochs = [10]

    # initiate frame to save accu
    actvs   = torch.zeros((len(tempDynamics), len(epochs), len(contrast_values), init, len(adapters), 2)) # 0: figure/ 1: ground  

    # import activations
    for iE, epoch in enumerate(epochs):
        actvs_current = torch.load(root + 'models/ratio_actvs')
        actvs[:, iE, :, :, :, :] = actvs_current
    print(actvs.shape)

    # plot settings
    markersize          = 7.5
    lw                  = 0.5

    # visualize ratio's F:G
    fig, axs = plt.subplots(2, 2, figsize=(5, 4), sharey=True)

    sns.despine(offset=5)

    # color
    color_none          = plt.cm.get_cmap('tab20c')
    color_none          = color_none.colors[16:]

    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    colors_adapt        = [color_same[-2], color_diff[-2]]

    offset_iC = np.array([0, 3, 6, 9, 12])

    iT = 1
    for i in range(2):
        for j in range(2):
            for iA, adapter in enumerate(adapters):

                # compute ratio
                ratio = actvs[iT, :, :, :, iA, 0]/actvs[iT, :, :, :, iA, 1]
                ratio = ratio.mean(0)
                # print(ratio.shape)

                # mean and standard deviation
                data_mean = torch.mean(ratio, 1)
                data_std = torch.std(ratio, 1)/math.sqrt(init)

                # visualize error
                axs[i, j].plot(offset_iC, data_mean, color=colors_adapt[iA], marker='o', markeredgewidth=1, markersize=markersize, markerfacecolor=colors_adapt[iA], markeredgecolor='white', lw=lw)
                for iC in range(len(contrast_values)):
                    axs[i, j].plot([offset_iC[iC], offset_iC[iC]], [data_mean[iC]-data_std[iC], data_mean[iC]+data_std[iC]], lw=lw, color=colors_adapt[iA], zorder=-1)  

                # adjust axes
                axs[i, j].spines['top'].set_visible(False)
                axs[i, j].spines['right'].set_visible(False)
                axs[i, j].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                axs[i, j].set_xticks(offset_iC)
                axs[i, j].set_xlim(-1, offset_iC[-1]+1)
                if i == 1:
                    axs[i, j].set_xlabel('Contrast (%)', fontsize=fontsize_label)
                    axs[i, j].set_xticklabels([50, 60, 70, 80, 90])
                else:
                    axs[i, j].set_xticklabels([' ', ' ', ' ', ' ', ' '])
                if j == 0:
                    axs[i, j].set_ylabel('Object:Noise', fontsize=fontsize_title)

            # increment count
            iT = iT + 1

    # save plot
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig8C', dpi=300)
    plt.savefig(root + 'visualization/Fig8C.svg')


if __name__ == '__main__':
    main()
