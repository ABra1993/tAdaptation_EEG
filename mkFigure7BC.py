# %%
# import packages
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
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
    # noise
    adapters           = ['same', 'different']

    # define contrast values
    contrasts            = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
    contrast_values      = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    contrasts_value_lbl  = [50, 60, 70, 80, 90]

    # visualize different contrast levels for repeated trials
    cmap = plt.cm.get_cmap('cool')
    contrasts_color = cmap(np.linspace(0, 1, len(contrast_values)))

    # set task and dataset
    task = 'contrast_gain'
    dataset = 'mnist'

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), 'div. norm.')
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # retrieve files human behaviour
    files = sorted(glob.glob(root + 'data/behaviour/raw/*.txt'))

    # initiate dataframe
    accu_grid = torch.load(root + 'models/variableGain_performance')

    # ######################################## BEHAVIOUR
    
    # initiate dataframe to store data
    accu_behaviour = np.zeros((len(contrast_values), len(files), len(adapters)))
    accu_model = np.zeros((len(contrast_values), len(files), len(adapters)))

    # extract data
    c50 = list()
    for iF, _ in enumerate(files):
    # for iF in range(1):

        # import datafile
        df = pd.read_csv(files[iF])

        for iC, contrast in enumerate(contrasts):

            # select trials
            select = df[(df.trial_type == 'single') & (df.contrast == contrast)]
            n_trials = len(select)

            # count number of correct trials
            correct = select[select.response == 1]
            n_trials_correct = len(correct)
            accu_behaviour[iC, iF, 0] = (n_trials_correct/n_trials)

            for iA, adapter in enumerate(adapters):
            
                # select trials
                select = df[(df.trial_type == 'repeated') & (df.adapter == adapter) & (df.contrast == contrast)]
                n_trials = len(select)

                # count number of correct trials
                correct = select[select.response == 1]
                n_trials_correct = len(correct)
                accu_behaviour[iC, iF, iA] = (n_trials_correct/n_trials)

        # compute difference
        idx = np.argwhere(accu_behaviour[:, iF, 0] > accu_behaviour[:, iF, 1])
        n_idx = len(idx)
        accu_model[:, iF, :] = accu_grid[-n_idx, :, :]

        # add c50
        c50.append(np.round(contrast_values[-n_idx].item(), 1))

    # initiate figure
    _, axs = plt.subplots(1, 2, figsize=(4.25, 2.7))
    sns.despine(offset=10)

    # colors
    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    adapters_color_beh  = [color_same[-2], color_diff[-2]]
    adapters_color      = [color_same[-1], color_diff[-1]]

    # mstyle
    markersize  = 7
    lw          = 1

    offset              = np.array([-0.02, 0.02])
    markersize          = 5
    lw                  = 0.5

    # statistical testing
    for iC, contrast_value in enumerate(contrast_values):

        # select
        sample1 = accu_model[iC, :, 0]
        sample2 = accu_model[iC, :, 1]

        # ttest
        result = stats.ttest_ind(sample1, sample2)
        if result[1] < 0.05:
            print(str(np.round(contrast_value.item(), 1)) + ': ' + str(np.round(result[1].item(), 3)))

    # plot accuracies
    for iA, adapter in enumerate(adapters):

        # select data
        data_mean = np.mean(accu_model[:, :, iA], 1)
        data_std = np.std(accu_model[:, :, iA], 1)/math.sqrt(accu_model.shape[1])

        # visualize     
        axs[0].plot(contrast_values+offset[iA], data_mean, color=adapters_color_beh[iA], markeredgewidth=0.5, label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA], markeredgecolor='white')
        axs[0].plot([contrast_values.numpy()+offset[iA], contrast_values.numpy()+offset[iA]], [data_mean - data_std, data_mean + data_std], color=adapters_color_beh[iA], zorder=-1)
        
        # adjust axes
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].tick_params(axis='both', which='major', labelsize=fontsize_tick)
        axs[0].set_xticks(contrast_values)
        axs[0].set_xticklabels(['50', '60', '70', '80', '90'])
        axs[0].set_xlabel('Contrast (%)', fontsize=fontsize_label)
        axs[0].set_ylabel('Accuracy', fontsize=fontsize_label)
        axs[0].set_title('$divisive$ $normalization$ \n $with$ $variable$ $gain$', fontsize=fontsize_title, color='gray')

    # histogram for c50
    axs[1].hist(c50, color='grey', edgecolor='white', bins=[0.5, 0.6, 0.7, 0.8, 0.9, 1])
    axs[1].set_ylabel('Count', fontsize=fontsize_label)
    axs[1].set_xticks(contrast_values.numpy()+0.05)
    axs[1].set_xticklabels(contrasts_value_lbl, fontsize=fontsize_tick)
    axs[1].set_xlabel(r'c$_{50}$ (%)', fontsize=fontsize_label)
    print(c50)

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig7BC', dpi=300)
    plt.savefig(root + 'visualization/Fig7BC.svg')
    # plt.close()


if __name__ == '__main__':
    main()

