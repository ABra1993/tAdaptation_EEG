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

fontsize_title          = 11
fontsize_legend         = 10
fontsize_label          = 11
fontsize_tick           = 10

def main():

    # input specification
    dataset                 = 'mnist'

    # specification of image sequence
    t_steps         = 3
    dur             = [1, 1]
    start           = [0, 2]

    # EEG paper (investigate contrast gain)
    tempDynamics                = ['none', 'add_supp', 'div_norm', 'lat_recurrence', 'lat_recurrence_mult']
    tempDynamics_label          = [r'feedforward', r'$additive$ $suppression$', r'$divisive$ $normalization$', r'$lateral$ $recurrence_{A}$', r'$lateral$ $recurrence_{M}$']
    tempDynamics_label_stats    = ['feedforward', 'additive suppression', 'divisive normalization', 'lateral recurrence (add.)', 'lateral recurrence (mult.)']

    # noise
    adapters                    = ['none', 'same', 'different']

    # define contrast values
    contrasts                   = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
    contrasts_value             = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])
    contrasts_value_lbl         = [50, 60, 70, 80, 90]

    # retrieve accuracies or load
    preload = True # if False the accuracies will be shown from the folder /models/performance_variableContrast_*, otherwise accuracies wil be re-computed

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start=start, dur=dur)

    # other settings
    init        = 10
    batch_size  = 100

    epochs      = [10]

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # retrieve files human behaviour
    files = sorted(glob.glob(root + 'data/behaviour/raw/*.txt'))

    ######################################## BEHAVIOUR
    
    # initiate dataframe to store data
    accu_behaviour = np.zeros((len(contrasts_value), len(files), len(adapters)))
    sub_n = len(files)

    # extract data
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

            for iA, adapter in enumerate(adapters[1:]):
            
                # select trials
                select = df[(df.trial_type == 'repeated') & (df.adapter == adapter) & (df.contrast == contrast)]
                n_trials = len(select)

                # count number of correct trials
                correct = select[select.response == 1]
                n_trials_correct = len(correct)
                accu_behaviour[iC, iF, 1+iA] = (n_trials_correct/n_trials)

    ######################################## MODELS

    # set home directory
    root_data           = root + 'models/dataset/'
    if dataset == 'mnist':
        testData        = datasets.MNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())
    elif dataset == 'fmnist':
        testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())

    # Define data loaders with drop_last=True
    ldrTest = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False, drop_last=True)

    if preload == False:

        # loop and retrieve accuracies
        for iT, current_tempDynamics in enumerate(tempDynamics):

            print(current_tempDynamics)

            # initiate frame to save accu
            accu    = torch.zeros((len(contrasts_value), len(epochs), init, 2)) # 2 = same/different

            for iE, epch in enumerate(epochs):

                for iInit in range(init):

                    # load model
                    model = cnn_feedforward(t_steps, current_tempDynamics)
                    model.initialize_tempDynamics()
                    model.load_state_dict(torch.load(root + 'models/weights/' + current_tempDynamics + '_' + str(iInit+1)))
                    model.to(device)
                    model.eval()

                    for iC, contrast_value in enumerate(contrasts_value):
                        
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
                        accu[iC, iE, iInit, 0] = torch.mean(accu_current[:, 0])
                        accu[iC, iE, iInit, 1] = torch.mean(accu_current[:, 1])

            # save accuracies
            np.save(root + 'models/performance/variableContrast_' + current_tempDynamics, accu)

    # import accuracy
    accu    = np.zeros((len(tempDynamics), len(contrasts_value), len(epochs), init, 2)) # 2 = same/different
    for iT, current_tempDynamics in enumerate(tempDynamics):
        accu[iT, :, :, :, :] = np.load(root + 'models/performance/variableContrast' + '_' + current_tempDynamics + '.npy')

    # visualize
    fig, axs = plt.subplots(1, len(tempDynamics), figsize=(10, 2.5))

    # color
    color_none          = plt.cm.get_cmap('tab20c')
    color_none          = color_none.colors[16:]
    color_none          = 'darkgrey'

    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    colors_adapt_beh    = [color_none, color_same[-2], color_diff[-2]]
    colors_adapt        = [color_same[-2], color_diff[-2]]

    sns.despine(offset=10)

    # axes settings
    offset              = np.array([-0.02, 0, 0.02])
    markersize          = 5
    markersize_small    = 2
    lw                  = 0.5

    # PLOT HUMAN PERFORMANCE
    for iA, adapter in enumerate(adapters):
        
        # retrieve accuracies
        mean   = np.mean(accu_behaviour[:, :, iA], 1)
        std    = np.std(accu_behaviour[:, :, iA], 1)/math.sqrt(sub_n)

        # visualize     
        axs[0].plot([contrasts_value.numpy()+offset[iA], contrasts_value.numpy()+offset[iA]], [mean-std, mean+std], color=colors_adapt_beh[iA], zorder=-1)  
        axs[0].plot(contrasts_value.numpy()+offset[iA], mean, color=colors_adapt_beh[iA], label=adapter, marker='o', markeredgewidth=0.5, markersize=markersize, markerfacecolor=colors_adapt_beh[iA], lw=lw, markeredgecolor='white')     

        # for iC in range(len(contrasts)):
            # axs[0].scatter(np.ones(accu_behaviour.shape[1])*contrasts_value[iC].numpy()+offset[iA], accu_behaviour[iC, :, iA], color=colors_adapt_beh[iA], s=markersize_small, alpha=0.5)
            # sns.stripplot(x=np.ones(accu_behaviour.shape[1])*contrasts_value[iC].numpy()+offset[iA], y=accu_behaviour[iC, :, iA], jitter=0.005, ax=axs[0], color=colors_adapt_beh[iA], size=markersize_small, alpha=0.5, native_scale=True, legend=False, zorder=-10)

        # adjust axes
        axs[0].spines['top'].set_visible(False)
        axs[0].spines['right'].set_visible(False)
        axs[0].tick_params(axis='both', which='major', labelsize=fontsize_tick)
        axs[0].set_xticks(contrasts_value)
        axs[0].set_xticklabels(['50', '60', '70', '80', '90'])
        axs[0].set_xlabel('Contrast (%)', fontsize=fontsize_label)
        axs[0].set_ylabel('Accuracy', fontsize=fontsize_label)
        axs[0].set_title('Human behaviour', fontsize=fontsize_title)
        # inset.set_xticks(contrasts_value)


    # color
    color_none          = plt.cm.get_cmap('tab20c')
    color_none          = color_none.colors[16:]
    color_none          = 'lightgrey'

    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    colors_adapt_beh    = [color_none, color_same[-2], color_diff[-2]]
    colors_adapt        = [color_same[-2], color_diff[-2]]


    # PLOT MODEL PERFORMANCE
    y_lim = [0.6, 1]
    for iT, current_tempDynamics in enumerate(tempDynamics[1:]):
        for iE, epch in enumerate(epochs):

            # statistical testing
            for iC, contrast_value in enumerate(contrasts_value):
                sample1 = accu[iT+1, iC, iE, :, 0]
                sample2 = accu[iT+1, iC, iE, :, 1]

                p_value = stats.ttest_ind(sample1, sample2)[1]
                print(tempDynamics_label_stats[iT+1], '-', epch, '-', contrasts_value_lbl[iC], '-', p_value)

            # select data (np.zeros((len(tempDynamics), len(contrasts_value), len(epochs), init, 2)) # 2 = same/different)
            data_mean   = np.mean(accu[0, :, iE, :, 0], 1)
            data_std    = np.std(accu[0, :, iE, : , 0], 1)/math.sqrt(init)

            # visualize
            axs[iT+1].plot([contrasts_value.numpy()+offset[0], contrasts_value.numpy()+offset[0]], [data_mean-data_std, data_mean+data_std], color=color_none, zorder=-1)  
            axs[iT+1].plot(contrasts_value+offset[0], data_mean, color=color_none, label=adapter, marker='o', markeredgewidth=0.5, markersize=markersize, markerfacecolor=color_none, markeredgecolor='white', lw=lw)

            # plot per network
            for iC in range(len(contrasts)):
                # axs[iT+1].scatter(np.ones(init)*contrasts_value[iC].numpy(), accu[0, iC, 0, :, iA], color=color_none, s=markersize_small, alpha=0.5)
                sns.stripplot(x=np.ones(init)*contrasts_value[iC].numpy()+offset[0], y=accu[0, iC, 0, :, 0], jitter=0.008, ax=axs[iT+1], color=color_none, size=markersize_small, alpha=0.5, native_scale=True, legend=False, zorder=-10)

            # plot networks with temporal adaptation
            for iA, adapter in enumerate(adapters[1:]):
                
                # select data
                data_mean   = np.mean(accu[iT+1, :, iE, :, iA], 1)
                data_std    = np.std(accu[iT+1, :, iE, : , iA], 1)/math.sqrt(init)

                # visualize
                axs[iT+1].plot([contrasts_value.numpy()+offset[iA+1], contrasts_value.numpy()+offset[iA+1]], [data_mean-data_std, data_mean+data_std], color=colors_adapt[iA], zorder=-1)  
                axs[iT+1].plot(contrasts_value+offset[iA+1], data_mean, color=colors_adapt[iA], label=adapter, marker='o', markeredgewidth=0.5, markersize=markersize, markerfacecolor=colors_adapt[iA], markeredgecolor='white', lw=lw)

                # plot per network
                for iC in range(len(contrasts)):
                    # axs[iT+1].scatter(np.ones(init)*contrasts_value[iC].numpy()+offset[iA], accu[iT+1, iC, 0, :, iA], color=colors_adapt[iA], s=markersize_small, alpha=0.5)
                    sns.stripplot(x=np.ones(init)*contrasts_value[iC].numpy()+offset[iA+1], y=accu[iT+1, iC, 0, :, iA], jitter=0.008, ax=axs[iT+1], color=colors_adapt[iA], size=markersize_small, alpha=0.5, native_scale=True, legend=False, zorder=-10)

                # adjust axes
                axs[iT+1].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iT+1].spines['top'].set_visible(False)
                axs[iT+1].spines['right'].set_visible(False)
                axs[iT+1].tick_params(axis='both', which='major', labelsize=fontsize_tick)

                axs[iT+1].set_title(tempDynamics_label[iT+1], color='gray', fontweight='bold', fontsize=fontsize_title)
    
                axs[iT+1].set_xticks(contrasts_value)
                axs[iT+1].set_xticklabels(['50', '60', '70', '80', '90'])
                axs[iT+1].set_xlabel('Contrast (%)', fontsize=fontsize_label)
                axs[iT+1].set_ylim(y_lim[0], y_lim[1])
                    
    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig7A', dpi=300)
    plt.savefig(root + 'visualization/Fig7A.svg')
    # plt.show()


if __name__ == '__main__':
    main()

