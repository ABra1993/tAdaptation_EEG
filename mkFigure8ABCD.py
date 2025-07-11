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
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

# define root
# root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'
root                = '/home/amber/Documents/organize_code/nAdaptation_EEG_git/'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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
fontsize_tick           = 9

# axes settings
offset1              = np.array([-0.02, 0, 0.02])
offset2              = np.array([-0.25, 0, 0.25])
markersize          = 70
markersize_small    = 3
lw                  = 2

def performance():

    # input sequences
    # sequences = ['ABT', 'AAABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AABTT', 'AAABTTT', 'AAAABTTTT', 'AAAAAAAABTTTTTTTT']
    # sequences = ['ABT', 'AAABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT', 'AAAAAAAAAAAAAAAAAABTTTTTT', 'AAAAAAAAAAAAAAAAAAAAABTTTTTTT']
    # sequences = ['ABT', 'AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT', 'AAAAAAAAAAAAAAAAAABTTTTTT', 'AAAAAAAAAAAAAAAAAAAAABTTTTTTT']

    # all sequences
    sequences       = ['ABT', 'AAAAAAAAAAAAAAABTTTTT']
    sequences_all   = ['ABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT']
    seq_length      = [len(seq) for seq in sequences_all]

    # other settings
    init        = 5

    # EEG paper (investigate contrast gain)
    # tempDynamics                = ['none', 'add_supp', 'l_recurrence_A', 'div_norm_clamp'] #, 'div_norm_scale']
    # tempDynamics_label          = [r'feedforward', r'+ $additive$ $suppression$', r'+ $lateral$ $recurrence_{A}$', r'+ $divisive$ $normalization$'] #, r'+ $divisive$ $normalization_{scale}$']

    tempDynamics                = ['none', 'add_supp', 'l_recurrence_A', 'div_norm_clamp'] #, 'div_norm_scale']
    tempDynamics_label          = [r'$feedforward$', r'$add. $ $supp.$', r'$lat.$ $rec.$', r'$div.$ $norm.$'] #, r'+ $divisive$ $normalization_{scale}$']

    # noise
    adapters                    = ['same', 'different']
    hadapters                   = ['none', 'same', 'different']

    # define contrast values
    contrasts_value_model       = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # avgs
    avgs = np.zeros((len(sequences_all), len(tempDynamics), len(adapters), init))

    ######################################## BEHAVIOUR

    # retrieve files human behaviour
    files = sorted(glob.glob(root + 'data/behaviour/raw/*.txt'))

    # noise
    hadapters                    = ['none', 'same', 'different']

    # define contrast values
    hcontrasts                   = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
    hcontrasts_value             = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    
    # initiate dataframe to store data
    accu_behaviour = np.zeros((len(hcontrasts_value), len(files), len(hadapters)))
    sub_n = len(files)

    # extract data
    for iF, file in enumerate(files):

        # import datafile
        df = pd.read_csv(file)

        for iC, contrast in enumerate(hcontrasts):

            # select trials
            select = df[(df.trial_type == 'single') & (df.contrast == contrast)]
            n_trials = len(select)

            # count number of correct trials
            correct = select[select.response == 1]
            n_trials_correct = len(correct)
            accu_behaviour[iC, iF, 0] = (n_trials_correct/n_trials)

            for iA, adapter in enumerate(hadapters[1:]):
            
                # select trials
                select = df[(df.trial_type == 'repeated') & (df.adapter == adapter) & (df.contrast == contrast)]
                n_trials = len(select)

                # count number of correct trials
                correct = select[select.response == 1]
                n_trials_correct = len(correct)
                accu_behaviour[iC, iF, 1+iA] = (n_trials_correct/n_trials)

    ######################################## VISUALIZE

    # visualize
    fig, axs = plt.subplots(2, len(tempDynamics), figsize=(10, 5))
    sns.despine(offset=10)

    # color
    color_none          = plt.cm.get_cmap('tab20c')
    color_none          = color_none.colors[16:]
    color_none          = 'darkgrey'

    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    colors_adapt_beh    = [color_none, color_same[-1], color_diff[-1]]

    # PLOT HUMAN PERFORMANCE and VINKEN
    for iA, adapter in enumerate(hadapters):

        if adapter == 'none':
            continue

        # select data
        mean   = np.mean(accu_behaviour[:, :, iA], 1)
        std    = np.std(accu_behaviour[:, :, iA], 1)/math.sqrt(sub_n)

        # visualize     
        # axs[0, 0].plot([hcontrasts_value.numpy()+offset1[iA-1], hcontrasts_value.numpy()+offset1[iA-1]], [mean-std, mean+std], color=colors_adapt_beh[iA], zorder=-1)  
        axs[0, 0].plot(hcontrasts_value.numpy()+offset1[iA-1], mean, color=colors_adapt_beh[iA], lw=lw) #, marker='o', markeredgewidth=0.5, markersize=markersize, markerfacecolor=colors_adapt_beh[iA], lw=lw, markeredgecolor='white')

        # adjust axes
        axs[0, 0].tick_params(axis='both', labelsize=fontsize_tick)
        # axs[0, 0].set_xlabel('Contrast target', fontsize=fontsize_label)
        axs[0, 0].set_xticks([0.6, 0.8])
        axs[0, 0].set_xticklabels([60, 80])
        # axs[0, 0].set_ylabel('Classification accuracy', fontsize=fontsize_label)

        ######################################## EXTRACT MODEL BEHAVIOUR
        for iT, current_tempDynamics in enumerate(tempDynamics):

            if current_tempDynamics == 'none':
                continue
            
            # retrieve accuracies
            accu    = np.zeros((len(contrasts_value_model), 2, init)) # 2 = same/different
            for iInit in range(init):
                accu[:, :, iInit] = np.load(root + 'models/performance/' + sequences[0] + '/' + current_tempDynamics + '_' + str(iInit+1) + '.npy') 

            # plot networks with temporal adaptation
            data_mean   = np.mean(accu[:, iA-1, :], 1)/100
            data_std    = np.std(accu[:, iA-1, :], 1)/math.sqrt(init)/100

            # visualize
            axs[0, iT].fill_between(contrasts_value_model+offset1[iA-1], data_mean - data_std, data_mean + data_std, color=colors_adapt_beh[iA], alpha=0.2)
            axs[0, iT].plot(contrasts_value_model+offset1[iA-1], data_mean, color=colors_adapt_beh[iA], lw=lw)

            # adjust axes
            axs[0, iT].tick_params(axis='both', labelsize=fontsize_tick)
            axs[0, iT].set_xticks([0.1, 0.5, 0.9])
            axs[0, iT].set_xticklabels([10, 50, 90])
            # axs[0, iT].set_xlabel('Contrast target (%)', fontsize=fontsize_label)
            # if iT == 1:
            #     axs[0, iT].set_ylabel('Accuracy', fontsize=fontsize_label)
            axs[0, iT].set_title(tempDynamics_label[iT], color='gray', fontsize=fontsize_title)

    # adjust axes
    axs[0, 0].set_title('Human behaviour', color='black', fontsize=fontsize_title)

    # PLOT MODEL PERFORMANCE
    start               = 0.8
    end                 = 0.3
    color_same          = plt.colormaps['Blues'] #cm.get_cmap('tab20c')
    color_same          = [color_same(i) for i in np.linspace(start, end, len(sequences_all))]

    color_diff          = plt.colormaps['YlOrBr'] #cm.get_cmap('tab20b')
    color_diff          = [color_diff(i) for i in np.linspace(start, end, len(sequences_all))]

    colors_adapt        = [color_same, color_diff]

    for iT, current_tempDynamics in enumerate(tempDynamics):

        print(current_tempDynamics)

        # visualize networks
        for iSeq, sequence in enumerate(sequences[1:]):

            print(sequence)

            ######################################## EXTRACT MODEL BEHAVIOUR
            accu    = np.zeros((len(contrasts_value_model), 2, init)) # 2 = same/different
            for iInit in range(init):
                accu[:, :, iInit] = np.load(root + 'models/performance/' + sequence + '/' + current_tempDynamics + '_' + str(iInit+1) + '.npy') 

            # set chance level
            # axs[1, iT+1].axhline(0.1, linestyle='dotted', color='grey', linewidth=1.5)

            # plot networks with temporal adaptation
            for iA, adapter in enumerate(adapters):
                
                # select data and visualize
                data_mean   = np.mean(accu[:, iA, :], 1)/100
                data_std    = np.std(accu[:, iA, :], 1)/math.sqrt(init)/100

                # axs[1, iT].plot(contrasts_value_model[:-4]+offset1[iA], data_mean[:-4], color=colors_adapt_beh[iA+1], label=sequence, lw=lw)
                axs[1, iT].fill_between(contrasts_value_model+offset1[iA], data_mean - data_std, data_mean + data_std, color=colors_adapt_beh[iA+1], alpha=0.2)
                axs[1, iT].plot(contrasts_value_model+offset1[iA], data_mean, color=colors_adapt_beh[iA+1], label=sequence, lw=lw)

                # adjust axes
                axs[1, iT].tick_params(axis='both', labelsize=fontsize_tick)
                axs[1, iT].spines['top'].set_visible(False)
                axs[1, iT].spines['right'].set_visible(False)
                axs[1, iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                axs[1, iT].set_title(tempDynamics_label[iT], color='gray', fontsize=fontsize_title)
                axs[1, iT].set_xticks([0.1, 0.5, 0.9])
                axs[1, iT].set_xticklabels([10, 50, 90])
                # axs[1, iT].tick_params(axis='x', rotation=45)
                # axs[1, iT].set_xlabel('Contrast target (%)', fontsize=fontsize_label)
                # if iT < 2:
                #     axs[1, iT].set_ylabel('Classification accuracy', fontsize=fontsize_label)
                axs[1, iT].set_ylim(0.05, 1.05)

    # save figure
    fig.align_ylabels(axs)
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    # plt.tight_layout()
    plt.savefig(root + 'visualization/Fig8', dpi=300)
    plt.savefig(root + 'visualization/Fig8.svg')
    # plt.show()

    # statistical testing (ONE-WAY ANOVA)
    for iT, current_tempDynamic in enumerate(tempDynamics):

        print('Temp. dyn.: ', current_tempDynamic)

        res = f_oneway(avgs[0, iT, 0, :], avgs[1, iT, 0, :], avgs[2, iT, 0, :], avgs[3, iT, 0, :], avgs[4, iT, 0, :]) #, avgs[5, iT, 0, :])
        print(res)

        if res[1] < 0.05:
            res = tukey_hsd(avgs[0, iT, 0, :], avgs[1, iT, 0, :], avgs[2, iT, 0, :], avgs[3, iT, 0, :], avgs[4, iT, 0, :])# , avgs[5, iT, 0, :])
            print(res)

# plot 7A
performance()

