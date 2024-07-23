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
# models
from models.module_div_norm import module_div_norm
from models.module_add_supp import module_add_supp

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
fontsize_tick           = 8

# set seed
random.seed(0)

def naka_rushton(C, c50, q, B, **args):

    y = ((args['A']-B)*C**q)/(c50**q + C**q) + B
    
    return y

def main():

    # input specification
    dataset                 = 'mnist'

    # networks
    tempDynamics = ['add_supp', 'div_norm']

    # other settings
    init        = 10

    # noise
    adapters           = ['same', 'different']

    # define parameters
    params_add_supp = ['alpha', 'beta']
    params_div_norm = ['alpha', 'sigma', 'K']
    params = [params_add_supp, params_div_norm]

    params_add_supp_lbls = [r'$\alpha$', r'$\beta$']
    params_div_norm_lbls = [r'$\alpha$', r'$\sigma$', r'$K$']
    params_lbls = [params_add_supp_lbls, params_div_norm_lbls]

    # number of values
    n_value = 10

    # initiate parameter values to showacase behaviour of activations
    params_value_add_supp_alpha     = torch.linspace(0, 0.4, n_value)
    params_value_add_supp_beta      = torch.linspace(0.1, 2, n_value)
    params_value_add_supp = [params_value_add_supp_alpha, params_value_add_supp_beta]

    params_value_div_norm_alpha     = torch.linspace(0.001, 0.1, n_value)
    params_value_div_norm_sigma     = torch.linspace(0.5, 1.5, n_value)
    params_value_div_norm_K         = torch.linspace(0.1, 1, n_value)
    params_value_div_norm = [params_value_div_norm_alpha, params_value_div_norm_sigma, params_value_div_norm_K]

    params_values = [params_value_add_supp, params_value_div_norm]

    # define contrast values
    contrasts_lbls       = ['50', '60', '70', '80', '90']
    contrast_values      = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # iterate over B, q, and C50
    c50         = torch.linspace(0.5, 0.9, 5)

    # colors
    cmap = plt.cm.get_cmap('cool')
    color = cmap(np.linspace(0, 1, n_value))

    cmap                = plt.cm.get_cmap('Blues')
    color_same          = cmap(np.linspace(0.3, 1, len(c50)))

    # color_diff          = plt.cm.get_cmap('tab20b')
    # color_diff          = color_diff.colors[8:11]
    cmap                = plt.cm.get_cmap('YlOrBr')
    color_diff          = cmap(np.linspace(0.3, 1, len(c50)))

    adapters_color_beh  = [color_same, color_diff]

    cmap                = plt.cm.get_cmap('Greys')
    color_CRF           = cmap(np.linspace(0.3, 1, len(c50)))

    # plot settings
    offset_iC           = np.array([0, 3, 6, 9, 12])
    offset_iA           = [-0.3, 0.3]

    # mstyle
    markersize  = 7
    lw          = 1

    # toy models
    t = torch.arange(0, 100)
    t_steps = len(t)

    # set stimulus
    x = torch.zeros(t_steps)
    x[20:65] = 1

    # compute CRF
    x_min = 0.4
    x_max = 1.05
    x_CRF = torch.linspace(x_min, x_max, 100)

    alpha = [1, 0.6]

    # save accu
    for iT, tempDynamics_current in enumerate(tempDynamics):

        # initialize
        p_values_current = np.zeros((len(params[iT]), len(c50), len(contrast_values)))

        # initiate model
        if tempDynamics_current == 'add_supp':
            module = module_add_supp()
        elif tempDynamics_current == 'div_norm':
            module = module_div_norm()

        # initiate figure
        fig, axs = plt.subplots(1, 2*len(params[iT]), figsize=(10, 2))
        sns.despine(offset=10)

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=3, hspace=None)

        # load accuracies without variable gain
        accu = np.load(root + 'models/performance/variableContrast_' + tempDynamics_current + '.npy')

        # plot same withoout variable gain
        data_current = accu[:, 0, :, 0]
        # print(data_current.shape)

        # compute data
        data_mean = np.mean(data_current, 1)
        data_sem = np.std(data_current, 1)/math.sqrt(init)

        # for iP in range(len(params[iT])):
        #     axs[iP, 2].plot(offset_iC+offset_iA[0], data_mean, color='lightsalmon', marker='o', markersize=markersize, lw=lw, markerfacecolor='lightsalmon', markeredgecolor='white')

        # compute parameter
        for iP, param in enumerate(params[iT]):

            # load
            value_accu         = torch.load(root + 'models/variableGain/accu_' + tempDynamics_current + '_' + param)
            value_param        = torch.load(root + 'models/variableGain/param_' + tempDynamics_current + '_' + param)
            value_CRF          = torch.load(root + 'models/variableGain/CRF_' + tempDynamics_current + '_' + param)

            ############################################################### COLUMN 1
            for iV in range(n_value):

                # initiate values
                if tempDynamics_current == 'add_supp':

                    if iP == 0:
                        alpha   = params_values[iT][iP][iV]
                        beta    = torch.mean(params_values[iT][1])
                    elif iP == 1:
                        alpha   = torch.mean(params_values[iT][0])
                        beta    = params_values[iT][iP][iV]

                if tempDynamics_current == 'div_norm':

                    if iP == 0:
                        alpha   = params_values[iT][iP][iV]
                        sigma   = torch.mean(params_values[iT][1])
                        K       = torch.mean(params_values[iT][2])
                    elif iP == 1:
                        alpha   = torch.mean(params_values[iT][0])
                        sigma   = params_values[iT][iP][iV]
                        K       = torch.mean(params_values[iT][2])
                    elif iP == 2:
                        alpha   = torch.mean(params_values[iT][0])
                        sigma   = torch.mean(params_values[iT][1])
                        K       = params_values[iT][iP][iV]

                # set values
                if tempDynamics_current == 'add_supp':
                    module.alpha    = nn.Parameter(alpha)
                    module.beta     = nn.Parameter(beta)
                elif tempDynamics_current == 'div_norm':
                    module.alpha    = nn.Parameter(alpha)
                    module.sigma    = nn.Parameter(sigma)
                    module.K        = nn.Parameter(K)

                # reset dataframes
                r = torch.zeros(t_steps) # activations
                f = torch.zeros(t_steps) # feedback signal

                # compute response
                for tt in range(1, t_steps):
                    if tempDynamics_current == 'add_supp':
                        g, feedback = module.forward(r[tt-1], f[tt-1])
                        r[tt] = torch.nn.functional.relu(x[tt]-feedback)
                        f[tt] = g
                    if tempDynamics_current == 'div_norm':
                        feedback = module.forward(f[tt-1])
                        r[tt] = torch.nn.functional.relu(x[tt])*feedback
                        f[tt] = module.update_g(f[tt-1], r[tt-1])
                    
                # plot activation
                axs[iP*2].plot(r.detach().numpy(), color=color[iV], label=str(params_values[iT][iP][iV]), lw=1.5)
                axs[iP*2].plot(f.detach().numpy(), color=color[iV], linestyle='dashed', label=str(params_values[iT][iP][iV]), lw=0.5)
            
            for ic50, c50_current in enumerate(c50):

                ############################################################### COLUMN 2

                # # select data
                # data_current = value_CRF[ic50, :, :]

                # # compute data
                # data_mean = torch.mean(data_current, 1)
                # data_sem = torch.std(data_current, 1)/math.sqrt(init)

                # # find max value
                # if ic50 == 0:
                #     max_value = torch.max(data_mean)

                # # visualize CRF
                # axs[iP, 1].plot(x_CRF, data_mean, color=color_CRF[ic50], zorder=-1)
                # axs[iP, 1].axvline(c50_current, linestyle='dashed', lw=1, color=color_CRF[ic50], zorder=-10)


                ############################################################### COLUMN 3
                for iA, adapter in enumerate(adapters):

                    # select data
                    data_current = value_accu[ic50, :, iA, :]

                    # compute data
                    data_mean = torch.mean(data_current, 1)
                    data_sem = torch.std(data_current, 1)/math.sqrt(init)

                    # visualize accuracy 
                    if iA == 1:
                        # axs[iP, 2].plot(offset_iC+offset_iA[iA], data_mean, alpha=0.2, color=adapters_color_beh[iA], label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA], markeredgecolor='white')
                        axs[iP*2+1].plot(offset_iC+offset_iA[iA], data_mean, color=adapters_color_beh[iA][ic50], label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA][ic50], markeredgecolor='white', zorder=-10)
                    else:
                        axs[iP*2+1].plot(offset_iC+offset_iA[iA], data_mean, color=adapters_color_beh[iA][ic50], label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA][ic50], markeredgecolor='white')
                

                ############################################################### STATS
                for iC, contrast_value in enumerate(contrast_values):

                    # select
                    sample1 = value_accu[ic50, iC, 0, :]
                    sample2 = value_accu[ic50, iC, 1, :]

                    # ttest
                    result = stats.ttest_ind(sample1, sample2)[1]

                    if result < 0.05:
                        p_values_current[iP, ic50, iC] = result
                        # print('c50: ', c50_current, ',', str(np.round(contrast_value.item(), 1)) + ': ' + str(np.round(result[1].item(), 3)))
                    else:
                        p_values_current[iP, ic50, iC] = None

            ############################################################### ADJUST AXES
            axs[iP*2].set_title('Variable ' + params_lbls[iT][iP], fontsize=fontsize_title, color='dimgrey')
            # axs[iP*2].set_ylabel('activations (a.u.)', fontsize=fontsize_label)
            # axs[iP*2].set_xlabel('model timesteps (a.u.)', fontsize=fontsize_label)
            axs[iP*2].set_yticks([])
            axs[iP*2].set_xticks([])

            # if iP == 0:
            #     axs[iP, 1].set_title('Contrast response function ', fontsize=fontsize_title)
            # axs[iP, 1].set_ylabel(params_lbls[iT][iP], fontsize=fontsize_label)
            # if iP == len(params[iT]) - 1:
            #     axs[iP, 1].set_xlabel('contrast (%)', fontsize=fontsize_label)
            # axs[iP, 1].set_xticks(contrast_values)
            # axs[iP, 1].set_xticklabels(contrasts_lbls)
            # axs[iP, 1].axhline(max_value, color='lightsalmon')

            # if iP == 0:
            #     axs[iP*2+1].set_title('Performance ', fontsize=fontsize_title)
            # axs[iP*2+1].set_ylabel('accuracy', fontsize=fontsize_label)
            # if iP == len(params[iT]) - 1:
                # axs[iP*2+1].set_xlabel('contrast (%)', fontsize=fontsize_label)
            axs[iP*2+1].set_xticks(offset_iC)
            axs[iP*2+1].set_xticklabels(contrasts_lbls)
            if param == 'K':
                axs[iP*2+1].set_ylim(-0.3, 1)
            elif param == 'sigma':
                axs[iP*2+1].set_ylim(0.35, 1)
            else:
                axs[iP*2+1].set_ylim(0.5, 1)
                
            # print('\n')
            # print(tempDynamics_current, ', ', param, ': ', p_values_current[iP, :, :])
            np.savetxt(root + 'models/variableGain/stats_' + tempDynamics_current + '_' + param + '.csv', p_values_current[iP, :, :])

        # save figure
        fig.align_labels()
        plt.tight_layout()
        if tempDynamics_current == 'add_supp':
            plt.savefig(root + 'visualization/Fig7BC', dpi=300)
            plt.savefig(root + 'visualization/Fig7BC.svg')
        else:
            plt.savefig(root + 'visualization/Fig7DE', dpi=300)
            plt.savefig(root + 'visualization/Fig7DE.svg')
        # plt.close()



if __name__ == '__main__':
    main()

