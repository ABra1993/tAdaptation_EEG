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
    fig, axs = plt.subplots(1, len(tempDynamics), figsize=(10, 2))
    sns.despine(offset=10)

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
        for iSeq, sequence in enumerate(sequences_all):

            print(sequence)

            ######################################## EXTRACT MODEL BEHAVIOUR
            accu    = np.zeros((len(contrasts_value_model), 2, init)) # 2 = same/different
            for iInit in range(init):
                accu[:, :, iInit] = np.load(root + 'models/performance/' + sequence + '/' + current_tempDynamics + '_' + str(iInit+1) + '.npy') 

            # plot networks with temporal adaptation
            for iA, adapter in enumerate(adapters):

                # select data
                data_mean   = np.mean(accu[:, iA, :])/100
                data_std    = np.std(accu[:, iA, :].mean(0))/math.sqrt(init)/100

                # store averages
                avgs[iSeq, iT, iA, :] = accu[:, iA, :].mean(0)

                # visualize
                axs[iT].scatter(iSeq+offset2[iA], data_mean, color=np.array(colors_adapt[iA][iSeq]), edgecolor='white', s=markersize)
                axs[iT].plot([iSeq+offset2[iA], iSeq+offset2[iA]], [data_mean - data_std, data_mean + data_std], color=np.array(colors_adapt[iA][iSeq]), zorder=-1)

                axs[iT].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iT].spines['top'].set_visible(False)
                axs[iT].spines['right'].set_visible(False)
                axs[iT].tick_params(axis='both', which='major', labelsize=fontsize_tick)
                axs[iT].set_title(tempDynamics_label[iT], color='gray', fontsize=fontsize_title)
                axs[iT].set_xticks(np.arange(len(sequences_all)))
                axs[iT].set_xticklabels(seq_length, fontsize=fontsize_label, rotation=45)
                axs[iT].set_xlabel('Sequence length', fontsize=fontsize_label)
                if iT == 0:
                    axs[iT].set_ylabel('Accuracy', fontsize=fontsize_label)
                axs[iT].set_ylim(0.62, 0.95)

    # save figure
    plt.subplots_adjust(hspace=0.8, wspace=0.8)
    # plt.tight_layout()
    plt.savefig(root + 'visualization/SFig3', dpi=300)
    plt.savefig(root + 'visualization/SFig3.svg')
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

