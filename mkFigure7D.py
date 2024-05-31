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

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

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

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

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

    # layers with temporal adaptation
    layer_temp      = 'L123'

    # noise
    adapters           = ['same', 'different']
    adapters_color     = ['dodgerblue', 'gold']

    # define contrast values
    contrasts            = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
    contrast_values      = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start=start, dur=dur)

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), 'div. norm.')
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # retrieve files human behaviour
    files = sorted(glob.glob(root + 'data/behaviour/raw/*.txt'))

    ######################################## BEHAVIOUR
    
    # initiate dataframe to store data
    accu_behaviour = np.zeros((len(contrast_values), len(files), len(adapters)))

    # extract data
    count = 0
    for iF, file in enumerate(files):

        # import datafile
        df = pd.read_csv(file)

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

    ######################################## MODELS
                
    # subject-wise
    subjects_idx    = [12, 0]
    subject_c50     = [0.5, 0.8]

    subject_B       = [0.05, 0.05]
    subject_q       = [14, 14]

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

    # set home directory
    root_data           = root + 'models/dataset/'
    if dataset == 'mnist':
        testData        = datasets.MNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())
    elif dataset == 'fmnist':
        testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())

    # data loaders
    ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)

    # initiate frame to save accu
    accu    = torch.zeros((len(contrast_values), init, 2)) # 2 = same/different

    # compute CRF
    x_min = 0.4
    x_max = 1.05
    x = np.linspace(x_min, x_max, 100)

    offset              = [-0.3, 0.3]
    offset_iC           = np.array([0, 3, 6, 9, 12])

    # number of epochs used during training
    epoch       = 10

    # initiate figure
    fig, axs = plt.subplots(1, 4, figsize=(10, 2.5))
    sns.despine(offset=10)

    # search space
    for iS in range(len(subjects_idx)):
        
        # loop and retrieve accuracies
        K_values = torch.Tensor(init, len(contrast_values))
        # print
        print('C50: ', subject_c50[iS])
        print('B: ', subject_B[iS])
        print('q: ', subject_q[iS])

        for iInit in range(init):
            for iC, contrast_value in enumerate(contrast_values):

                # load model
                model = cnn_feedforward(t_steps, 'div_norm')
                model.initialize_tempDynamics()
                model.load_state_dict(torch.load(root + 'models/weights/div_norm_' + str(iInit+1)))

                # compute K-value
                K_value = naka_rushton(contrast_values, subject_c50[iS], subject_q[iS], subject_B[iS], A=model.sconv1.K.detach().cpu())
                K_values[iInit, :] = K_value

                # change variable of K
                model.sconv1.K = nn.Parameter(torch.Tensor(K_value[iC]))

                # to GPU
                model.to(device)
                model.eval()

                # retrieve test accuracies
                accu_current    = torch.zeros((len(ldrTest), 2))
                for a, (imgs, lbls) in enumerate(ldrTest):

                    # create input sequence
                    ax = sequence_train(F.adjust_contrast(imgs, contrast_value), t_steps, t_steps_label, False)

                    # validate
                    testoutp = model.forward(ax.to(device))
                    predicy = torch.argmax(testoutp, dim=1).to('cpu')

                    # compute accuracy for same and different noise
                    accu_current[a, 0] = (predicy[:int(imgs.shape[0]/2)] == lbls[:int(imgs.shape[0]/2)]).sum().item() / float(lbls.size(0)/2)
                    accu_current[a, 1] = (predicy[int(imgs.shape[0]/2):] == lbls[int(imgs.shape[0]/2):]).sum().item() / float(lbls.size(0)/2)

                # save accuracies
                accu[iC, iInit, 0] = torch.mean(accu_current[:, 0])
                accu[iC, iInit, 1] = torch.mean(accu_current[:, 1])

        # statistical testing
        K_values_mean = torch.mean(K_values, 0)
        for iC, contrast_value in enumerate(contrast_values):

            # select
            sample1 = accu[iC, :, 0]
            sample2 = accu[iC, :, 1]

            # ttest
            result = stats.ttest_ind(sample1, sample2)
            if result[1] < 0.05:
                print(str(np.round(contrast_value.item(), 1)) + ': ' + str(np.round(result[1].item(), 3)))

        # plot human behaviour
        for iA, adapter in enumerate(adapters):

            # visualize     
            axs[iS*2].plot(offset_iC+offset[iA], accu_behaviour[:, subjects_idx[iS], iA], color=adapters_color[iA], marker='o', markersize=markersize, markerfacecolor=adapters_color[iA], lw=lw, markeredgecolor='white')
        
        # adjust axes
        axs[iS*2].spines['top'].set_visible(False)
        axs[iS*2].spines['right'].set_visible(False)
        axs[iS*2].tick_params(axis='both', which='major')
        axs[iS*2].set_xticks(offset_iC)
        axs[iS*2].set_xticklabels([50, 60, 70, 80, 90])
        axs[iS*2].set_ylabel('Accuracy', fontsize=fontsize_label)
        axs[iS*2].set_xlabel('Contrast (%)', fontsize=fontsize_label)

        # plot model behavioiur
        for iA, adapter in enumerate(adapters):

            # select data
            data_mean   = torch.mean(accu[:, :, iA], 1)

            # visualize     
            axs[iS*2+1].plot(offset_iC+offset[iA], data_mean, color=adapters_color_beh[iA], label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA], markeredgecolor='white')
            for iC in range(len(contrast_values)):
                axs[iS*2+1].scatter(torch.ones(accu.shape[1])*offset_iC[iC]+offset[iA], accu[iC, :, iA], color=adapters_color_beh[iA], s=3, alpha=0.2)

        # adjust axes
        axs[iS*2+1].spines['top'].set_visible(False)
        axs[iS*2+1].spines['right'].set_visible(False)
        axs[iS*2+1].tick_params(axis='both', which='major', labelsize=fontsize_tick)
        axs[iS*2+1].set_xticks(offset_iC)
        axs[iS*2+1].set_xticklabels([50, 60, 70, 80, 90])
        axs[iS*2+1].set_xticklabels([50, 60, 70, 80, 90])
        axs[iS*2+1].set_xlabel('Contrast (%)', fontsize=fontsize_label)

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig7D', dpi=300)
    plt.savefig(root + 'visualization/Fig7D.svg')
    # plt.close()


if __name__ == '__main__':
    main()

