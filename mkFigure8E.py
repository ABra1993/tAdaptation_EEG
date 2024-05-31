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
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import glob
from scipy import stats
from mpl_toolkits.axes_grid.inset_locator import inset_axes

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

# set seed
random.seed(0)

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
fontsize_label          = 11
fontsize_tick           = 10

def main():

    # define root
    root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

    # temporal dynamics
    tempDynamics                = ['div_norm', 'add_supp', 'lat_recurrence_mult', 'lat_recurrence']
    tempDynamics_label          = ['$\it{divisive}$ $\it{normalization}$', r'$additive$ $suppression$', r'$lateral$ $recurrence_{M}$', r'$lateral$ $recurrence_{A}$']

    # other settings
    init        = 5
    epochs      = [10]
    layers      = ['conv1', 'conv2', 'conv3']

    # parameters
    param_AS                    = [r'$\beta$', r'$\alpha$']
    param_LR                    = [r'W']
    param_DN                    = [r'$K$', r'$\sigma$', r'$\alpha$']
    param                       = [param_DN, param_AS, param_LR, param_LR]

    param_AS_value              = np.zeros((len(epochs), len(layers), len(param_AS), init))
    param_LRM_value              = np.zeros((len(epochs), len(layers), len(param_LR), init))
    param_LRA_value              = np.zeros((len(epochs), len(layers), len(param_LR), init))
    param_DN_value              = np.zeros((len(epochs), len(layers), len(param_DN), init))
    param_value                 = [param_DN_value, param_AS_value, param_LRM_value, param_LRA_value]

    # loop and retrieve accuracies
    for iT, current_tempDynamics in enumerate(tempDynamics):

        for iE, epch in enumerate(epochs):

            for iInit in range(init):

                # save model
                model = cnn_feedforward(tempDynamics=current_tempDynamics)
                model.initialize_tempDynamics()
                model.load_state_dict(torch.load(root + 'models/weights/' + current_tempDynamics + '_' + str(iInit+1)))
                model.to(device)
                model.eval()
                    
                if current_tempDynamics == 'add_supp':

                    # beta 
                    param_value[iT][iE, 0, 0, iInit] = model.sconv1.beta.item()
                    param_value[iT][iE, 1, 0, iInit] = model.sconv2.beta.item()
                    param_value[iT][iE, 2, 0, iInit] = model.sconv3.beta.item()

                    # alpha 
                    param_value[iT][iE, 0, 1, iInit] = model.sconv1.alpha.item()
                    param_value[iT][iE, 1, 1, iInit] = model.sconv2.alpha.item()
                    param_value[iT][iE, 2, 1, iInit] = model.sconv3.alpha.item()


                elif (current_tempDynamics == 'lat_recurrence') | (current_tempDynamics == 'lat_recurrence_mult'):

                    # weight 
                    param_value[iT][iE, 0, 0, iInit] = torch.mean(model.sconv1.weight).item()
                    param_value[iT][iE, 1, 0, iInit] = torch.mean(model.sconv2.weight).item()
                    param_value[iT][iE, 2, 0, iInit] = torch.mean(model.sconv3.weight).item()

                elif current_tempDynamics == 'div_norm':

                    # K 
                    param_value[iT][iE, 0, 0, iInit] = model.sconv1.K.item()
                    param_value[iT][iE, 1, 0, iInit] = model.sconv2.K.item()
                    param_value[iT][iE, 2, 0, iInit] = model.sconv3.K.item()

                    # sigma 
                    param_value[iT][iE, 0, 1, iInit] = model.sconv1.sigma.item()
                    param_value[iT][iE, 1, 1, iInit] = model.sconv2.sigma.item()
                    param_value[iT][iE, 2, 1, iInit] = model.sconv3.sigma.item()

                    # alpha 
                    param_value[iT][iE, 0, 2, iInit] = model.sconv1.alpha.item()
                    param_value[iT][iE, 1, 2, iInit] = model.sconv2.alpha.item()
                    param_value[iT][iE, 2, 2, iInit] = model.sconv3.alpha.item()

    # initiate figure
    fig, axs = plt.subplots(3, 3, figsize=(7, 6))
    axs[1, 2].axis('off')
    axs[2, 2].axis('off')

    # plot settings
    color         = plt.cm.get_cmap('tab20c')
    color         = color.colors[16:]

    markersize          = 8
    lw                  = 1

    # visualize
    for iT, current_tempDynamics in enumerate(tempDynamics):

        for iP, current_param in enumerate(param[iT]):

            for iE, epoch in enumerate(epochs):

                # select data
                current_data = param_value[iT][iE, :, iP, :]

                data_mean = current_data.mean(1)
                data_std = current_data.std(1)/math.sqrt(init)

                # select axes
                row = 0
                col = 0
                if current_tempDynamics == 'add_supp':
                    row = 1
                    col = iP
                elif current_tempDynamics == 'lat_recurrence_mult':
                    row = 2
                    col = 0
                elif current_tempDynamics == 'lat_recurrence':
                    row = 2
                    col = 1
                elif current_tempDynamics == 'div_norm':
                    row = 0
                    col = iP

                # plot
                axs[row, col].plot([np.arange(len(layers)), np.arange(len(layers))], [data_mean-data_std, data_mean+data_std], color=color[-1-iE], zorder=-1)  
                axs[row, col].plot(np.arange(len(layers)), data_mean, color=color[-1-iE], label=epoch, marker='o', markeredgewidth=0.5, markersize=markersize, markerfacecolor=color[-1-iE], markeredgecolor='white', lw=lw)

            # adjust axes
            axs[row, col].spines['top'].set_visible(False)
            axs[row, col].spines['right'].set_visible(False)
            axs[row, col].set_title(current_param, size=fontsize_label)
            axs[row, col].set_xlim(-0.5, 2.5)
            axs[row, col].set_xticks(np.arange(len(layers)))
            axs[row, col].set_xticklabels(layers, rotation=45)

    # set labels
    color_label = 'red'
    axs[0, 0].set_ylabel(tempDynamics_label[0], size=fontsize_label, color=color_label)
    axs[1, 0].set_ylabel(tempDynamics_label[1], size=fontsize_label, color=color_label)
    axs[2, 0].set_ylabel(tempDynamics_label[2], size=fontsize_label, color=color_label)
    axs[2, 1].set_ylabel(tempDynamics_label[2], size=fontsize_label, color=color_label)

    # axs[0, 0].legend(fontsize=fontsize_legend, frameon=False)

    # save figure
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig8E', dpi=300)
    plt.savefig(root + 'visualization/Fig8E.svg')
    # plt.close()


if __name__ == '__main__':
    main()

