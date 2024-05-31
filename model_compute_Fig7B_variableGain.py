# %%
# import packages
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
from torch import optim

import torchvision.transforms.functional as F
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns
import pandas as pd
import glob
import scipy
from scipy import stats

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

# fontsizes 
global fontsize_title
global fontsize_legend
global fontsize_label
global fontsize_tick

fontsize_title          = 11
fontsize_legend         = 10
fontsize_label          = 11
fontsize_tick           = 10

# set seed
random.seed(0)

def naka_rushton(C, c50, q, B, **args):

    y = ((args['A']-B)*C**q)/(c50**q + C**q) + B
    
    return y

def main():

    # input specification
    dataset                 = 'mnist'

    # specification of image sequence
    t_steps         = 3
    dur             = [1, 1]
    start           = [0, 2]

    # other settings
    init        = 5
    batch_size  = 100

    # noise
    adapters           = ['same', 'different']

    # define contrast values
    contrasts            = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
    contrast_values      = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])

    # visualize different contrast levels for repeated trials
    cmap = plt.cm.get_cmap('cool')
    contrasts_color = cmap(np.linspace(0, 1, len(contrast_values)))

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start=start, dur=dur)

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), 'div. norm.')
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    ######################################## MODELS

    # colors
    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    adapters_color_beh  = [color_same[-2], color_diff[-2]]

    # mstyle
    markersize  = 7
    lw          = 1

    # set home directory
    root_data           = root + 'models/dataset/'
    if dataset == 'mnist':
        testData        = datasets.MNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())
    elif dataset == 'fmnist':
        testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())

    # data loaders
    ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)

    # compute CRF
    x_min = 0.4
    x_max = 1.05
    x = torch.linspace(x_min, x_max, 100)

    offset              = [-0.3, 0.3]
    offset_iC           = np.array([0, 3, 6, 9, 12])

    # number of epochs used during training
    epoch       = 10

    # iterate over B, q, and C50
    c50         = torch.linspace(0.5, 0.9, 5)
    q           = torch.Tensor([14])
    B           = torch.Tensor([0.05])

    c50_lbls    = ['50', '60', '70', '80', '90']

    # store K-values
    accu        = torch.zeros((len(c50), len(contrasts), len(adapters)))
    param       = torch.zeros((len(c50), len(adapters), 2)) # 2 => slope and intercept

    # save accu
    count = 0
    for ic50, c50_current in enumerate(c50):
        for _, q_current in enumerate(q):
            for _, B_current in enumerate(B):

                # initiate figure
                _, axs = plt.subplots(1, 2, figsize=(4, 2.5))
                sns.despine(offset=10)
            
                # loop and retrieve accuracies
                K_values = torch.Tensor(init, len(contrast_values))

                print('C50: ', c50_current)
                print('q: ', q_current)
                print('B: ', B_current)

                accu_current = torch.zeros((len(contrasts), len(adapters), init))
                for iInit in range(init):
                    for iC, contrast_value in enumerate(contrast_values):

                        # load model
                        model = cnn_feedforward(t_steps, 'div_norm')
                        model.initialize_tempDynamics()
                        model.load_state_dict(torch.load(root + 'models/weights/div_norm_' + str(iInit+1)))

                        # compute K-value
                        K_value = naka_rushton(contrast_values, c50_current, q_current, B_current, A=model.sconv1.K.detach().cpu())
                        K_values[iInit, :] = K_value

                        y = naka_rushton(x, c50_current, q_current, B_current, A=model.sconv1.K.detach().cpu())
                        axs[1].plot(x, y, color='darkgrey', zorder=-1, lw=0.2)
                        axs[1].axvline(c50_current, linestyle='dotted', lw=0.5, color='black', zorder=-10)

                        # visualize
                        axs[1].scatter(contrast_values[iC], K_value[iC], color='grey', zorder=1, label=str(int(contrast_values[iC]*100)) + '%', s=5, alpha=0.5)

                        # adjust axes
                        axs[1].spines['top'].set_visible(False)
                        axs[1].spines['right'].set_visible(False)
                        axs[1].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                        axs[1].set_xticks(contrast_values)
                        axs[1].set_xticklabels([50, 60, 70, 80, 90], rotation=45)
                        axs[1].set_xlabel('Contrast (%)', fontsize=fontsize_label)
                        axs[1].set_ylabel(r'K', fontsize=fontsize_label)

                        # change variable of K
                        model.sconv1.K = nn.Parameter(torch.Tensor(K_value[iC]))

                        # to GPU
                        model.to(device)
                        model.eval()

                        # retrieve test accuracies
                        accu_temp = torch.zeros((len(ldrTest), 2))
                        for a, (imgs, lbls) in enumerate(ldrTest):

                            # create input sequence
                            ax = sequence_train(F.adjust_contrast(imgs, contrast_value), t_steps, t_steps_label, False)

                            # validate
                            testoutp = model.forward(ax.to(device))
                            predicy = torch.argmax(testoutp, dim=1).to('cpu')

                            # compute accuracy for same and different noise
                            accu_temp[a, 0] = (predicy[:int(imgs.shape[0]/2)] == lbls[:int(imgs.shape[0]/2)]).sum().item() / float(lbls.size(0)/2)
                            accu_temp[a, 1] = (predicy[int(imgs.shape[0]/2):] == lbls[int(imgs.shape[0]/2):]).sum().item() / float(lbls.size(0)/2)

                        # add to dataframe
                        accu_current[iC, :, iInit] = accu_temp.mean(0)

                    # save accuracies
                    accu[ic50, :, :] = accu_current.mean(-1)

                # statistical testing
                K_values_mean = torch.mean(K_values, 0)
                for iC, contrast_value in enumerate(contrast_values):

                    # plot contrast values
                    axs[1].scatter(contrast_values[iC], K_values_mean[iC], s=70, color=contrasts_color[iC], edgecolors='white')

                    # select
                    sample1 = accu_current[iC, 0, :]
                    sample2 = accu_current[iC, 1, :]

                    # ttest
                    result = stats.ttest_ind(sample1, sample2)
                    if result[1] < 0.05:
                        print(str(np.round(contrast_value.item(), 1)) + ': ' + str(np.round(result[1].item(), 3)))

                # plot model behavioiur
                for iA, adapter in enumerate(adapters):

                    # select data
                    data_mean = accu[ic50, :, iA]

                    # visualize     
                    axs[0].plot(offset_iC+offset[iA], data_mean, color=adapters_color_beh[iA], label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA], markeredgecolor='white')
                    for iC in range(len(contrast_values)):
                        axs[0].scatter(torch.ones(accu_current.shape[-1])*offset_iC[iC]+offset[iA], accu_current[iC, iA, :], color=adapters_color_beh[iA], s=3, alpha=0.2)

                # adjust axes            
                axs[0].spines['top'].set_visible(False)
                axs[0].spines['right'].set_visible(False)
                axs[0].tick_params(axis='both', which='major')
                axs[0].set_xticks(offset_iC)
                axs[0].set_xticklabels([50, 60, 70, 80, 90])
                axs[0].set_xlabel('Contrast (%)', fontsize=fontsize_label)
                axs[0].set_ylabel('Accuracy', fontsize=fontsize_label)

                # save figure
                plt.tight_layout()
                plt.savefig(root + 'visualization/SFig2_' + c50_lbls[ic50], dpi=300)
                plt.savefig(root + 'visualization/SFig2_' + c50_lbls[ic50] + '.svg')

                # increment count
                count+=1

    # save
    torch.save(accu, root + 'models/variableGain_performance')
    torch.save(param, root + 'models/variableGain_params')


if __name__ == '__main__':
    main()

