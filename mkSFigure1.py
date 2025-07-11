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

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

# fontsizes 
global fontsize_title
global fontsize_legend
global fontsize_label
global fontsize_tick

global colors_adapt

global offset
global markersize
global markersize_small
global lw

# fontsizes
fontsize_title          = 12
fontsize_legend         = 9
fontsize_label          = 10
fontsize_tick           = 10

markersize          = 10
markersize_small    = 3
lw                  = 0.5

width               = 0.8

# define root
root                = '/home/amber/Documents/organize_code/nAdaptation_EEG_git/'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():

    # input sequences
    # sequences = ['ABT', 'AAABT', 'AAAAAABTT', 'AAAAAAAAABTTT']
    sequences = ['ABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT']
    # sequences = ['ABT', 'AAABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT', 'AAAAAAAAAAAAAAAAAABTTTTTT']

    seq_length      = [len(seq) for seq in sequences]

    # temporal adaptation mechanisms
    tempDynamics                = ['add_supp', 'div_norm_clamp']
    tempDynamics_label          = [r'$add. $ $supp.$', r'$div.$ $norm.$'] #, r'+ $divisive$ $normalization_{scale}$']

    # network initiliazations
    init            = 5
    layers          = ['conv1', 'conv2', 'conv3']
    n_layer         = len(layers)

    # parameter values for additive adaptation and divisive normalization
    param_AS_lbl    = [r'$\alpha$', r'$\beta$']
    color_AS        = ['crimson', 'navy']
    param_AS        = torch.zeros((len(param_AS_lbl), n_layer, init))

    param_DN_lbl    = [r'$\alpha$', r'$\sigma$', r'DN = $K$']
    color_DN        = ['firebrick', 'teal', 'darkslateblue']
    param_DN        = torch.zeros((len(param_DN_lbl), n_layer, init))

    param_all       = [r'AS - $\alpha$', r'AS - $\beta$', r'DN - $\alpha$', r'DN - $\sigma$', r'DN - $K$']

    color           = [color_AS, color_DN]

    # initiate figure
    _, axs = plt.subplots(len(sequences), len(tempDynamics)+1, figsize=(9, 12))
    sns.despine(offset=10)
    # axs[len(sequences)-1, 2].axis('off')

    # get into
    for iSeq, sequence in enumerate(sequences):

        # retrieve sequence
        t_steps = len(sequence)
        t_steps_label = encode_timesteps(t_steps, sequence)
        print(t_steps_label)

        for iT, current_tempDynamics in enumerate(tempDynamics):

            # initiate model
            model = cnn_feedforward()

            # initiate recurrence
            if tempDynamics != 'none':
                model.initialize_tempDynamics(current_tempDynamics)

            # init timesteps
            model.init_t_steps(t_steps)

            # number of parameters
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Total number of trainable parameters {current_tempDynamics} : {trainable_params}")

            if (current_tempDynamics == 'add_supp') | ('div_norm' in current_tempDynamics):

                for iInit in range(init):

                    # save model
                    model.load_state_dict(torch.load(root + 'models/weights/' + sequence + '/' + current_tempDynamics + '_' + str(iInit+1)))
        
                    # obtain parameters
                    if current_tempDynamics == 'add_supp':

                        # conv1
                        param_AS[0, 0, iInit] = model.sconv1.alpha.item()
                        param_AS[1, 0, iInit] = model.sconv1.beta.item()

                        # conv2
                        param_AS[0, 1, iInit] = model.sconv2.alpha.item()
                        param_AS[1, 1, iInit] = model.sconv2.beta.item()

                        # conv3
                        param_AS[0, 2, iInit] = model.sconv3.alpha.item()
                        param_AS[1, 2, iInit] = model.sconv3.beta.item()

                    # obtain parameters
                    if ('div_norm' in current_tempDynamics):

                        # conv1
                        param_DN[0, 0, iInit] = model.sconv1.alpha.item()
                        param_DN[1, 0, iInit] = model.sconv1.sigma.item()
                        param_DN[2, 0, iInit] = model.sconv1.K.item()

                        # conv2
                        param_DN[0, 1, iInit] = model.sconv2.alpha.item()
                        param_DN[1, 1, iInit] = model.sconv2.sigma.item()
                        param_DN[2, 1, iInit] = model.sconv2.K.item()

                        # conv3
                        param_DN[0, 2, iInit] = model.sconv3.alpha.item()
                        param_DN[1, 2, iInit] = model.sconv3.sigma.item()
                        param_DN[2, 2, iInit] = model.sconv3.K.item()

            # initiate figure
            if current_tempDynamics == 'add_supp':

                # plot parameters
                for iP, param in enumerate(param_AS_lbl):

                    # retrieve values
                    mean        = param_AS[iP, :, :].mean(1)
                    std         = param_AS[iP, :, :].std(1)/math.sqrt(init)

                    # plot
                    axs[iSeq, iT].plot([np.arange(n_layer)+1, np.arange(n_layer)+1], [mean - std, mean + std], color=color[iT][iP], zorder=-1, label=param)
                    axs[iSeq, iT].plot(np.arange(n_layer)+1, mean, color=color[iT][iP], marker='o', markeredgewidth=3, markersize=markersize, markerfacecolor=color[iT][iP], markeredgecolor='white', lw=lw, zorder=1)
            
                    # adjust axes
                    axs[iSeq, iT].set_title('t = ' + str(seq_length[iSeq]), fontsize=fontsize_title)
                    axs[iSeq, iT].tick_params(axis='both', labelsize=fontsize_tick)
                    axs[iSeq, iT].spines['top'].set_visible(False)
                    axs[iSeq, iT].spines['right'].set_visible(False)
                    axs[iSeq, iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                    if iSeq == len(sequences) - 1:
                        axs[iSeq, iT*2].set_xlabel('Layer', fontsize=fontsize_label)
                    axs[iSeq, iT].set_ylabel('Value (a.u.)', fontsize=fontsize_label)
                    # axs[iSeq, iT].set_ylim(0, 2.5)
                    axs[iSeq, iT].set_xticks(np.arange(n_layer)+1)
                    axs[iSeq, iT].set_xlim(0, len(layers)+1)

                    # retrieve values
                    mean        = param_AS[iP, :, :].mean()
                    std         = param_AS[iP, :, :].std()/math.sqrt(init)

                    # plot
                    axs[iP, 2].plot([iSeq, iSeq], [mean - std, mean + std], color='grey', zorder=-1, label=param)
                    # axs[iSeq, iT*2+1].plot(iP, mean, color=color[iT][iP], marker='o', markeredgewidth=3, markersize=markersize, markerfacecolor=color[iT][iP], markeredgecolor='white', lw=lw, zorder=1)
            
                    axs[iP, 2].bar(iSeq, mean, color=color[iT][iP], width=width,  alpha=0.8, zorder=1)

            elif ('div_norm' in current_tempDynamics):

                # plot parameters
                for iP, param in enumerate(param_DN_lbl):

                    # retrieve values
                    mean        = param_DN[iP, :, :].mean(1)
                    std         = param_DN[iP, :, :].std(1)/math.sqrt(init)

                    # plot
                    axs[iSeq, iT].plot([np.arange(n_layer)+1, np.arange(n_layer)+1], [mean - std, mean + std], zorder=-1, color='grey', label=param)
                    axs[iSeq, iT].plot(np.arange(n_layer)+1, mean, color=color[iT][iP], marker='o', markeredgewidth=3, markersize=markersize, markerfacecolor=color[iT][iP], markeredgecolor='white', lw=lw, zorder=1)
            
                    # adjust axes
                    axs[iSeq, iT].set_title('t = ' + str(seq_length[iSeq]), fontsize=fontsize_title)
                    axs[iSeq, iT].tick_params(axis='both', labelsize=fontsize_tick)
                    axs[iSeq, iT].spines['top'].set_visible(False)
                    axs[iSeq, iT].spines['right'].set_visible(False)
                    axs[iSeq, iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                    if iSeq == len(sequences) - 1:
                        axs[iSeq, iT*2].set_xlabel('Layer', fontsize=fontsize_label)
                    # axs[iSeq, iT].set_ylim(0, 2.5)
                    axs[iSeq, iT].set_xticks(np.arange(n_layer)+1)
                    axs[iSeq, iT].set_ylim(-0.1, 1.1)
                    axs[iSeq, iT].set_xlim(0, len(layers)+1)

                    # retrieve values
                    mean        = param_DN[iP, :, :].mean()
                    std         = param_DN[iP, :, :].std()/math.sqrt(init)

                    # plot
                    axs[len(param_AS_lbl)+iP, 2].plot([iSeq, iSeq], [mean - std, mean + std], color='grey', zorder=-1, label=param)
                    # axs[param_count, 2].scatter(iSeq, mean, color=color[iT][iP], marker='o', markeredgewidth=3, markersize=markersize, markerfacecolor=color[iT][iP], markeredgecolor='white', lw=lw, zorder=1)
                    axs[len(param_AS_lbl)+iP, 2].bar(iSeq, mean, color=color[iT][iP], width=width, alpha=0.8, zorder=1)

    # adjust axes for parameter plots
    for iP in range(len(param_all)):
        axs[iP, 2].set_title(param_all[iP], fontsize=fontsize_title)
        axs[iP, 2].tick_params(axis='both', labelsize=fontsize_tick)
        axs[iP, 2].spines['top'].set_visible(False)
        axs[iP, 2].spines['right'].set_visible(False)
        axs[iP, 2].tick_params(axis='both', which='major', labelsize=fontsize_tick)
        if iSeq == len(sequences) - 1:
            axs[iP, 2].set_xlabel('Sequence length', fontsize=fontsize_label)
        axs[iP, 2].set_ylabel('Value (a.u.)', fontsize=fontsize_label)
        axs[iP, 2].set_xticks(np.arange(len(sequences)))
        axs[iP, 2].set_xticklabels(seq_length)

    # save plot
    # axs[0, 0].legend()
    # axs[0, 1].legend()
    plt.tight_layout()
    plt.savefig(root + 'visualization/SFig1', dpi=300)
    plt.savefig(root + 'visualization/SFig1.svg')

if __name__ == '__main__':
    main()

