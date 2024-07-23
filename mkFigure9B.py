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
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd
import matplotlib.pyplot as plt
import random
import neptune.new as neptune
from torch.autograd import Variable as var

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *
from mkFigure9B_utils import *


# define root
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():

    # tempDynamics (exp. decay, div. norm., lat. recurrence, power law)
    tempDynamics                = ['add_supp', 'div_norm', 'lat_recurrence',  'lat_recurrence_mult']
    tempDynamics_label          = [r'$AS$', r'$DN$', r'$LR_{A}$', r'$LR_{M}$']

    # set task and dataset
    dataset = 'mnist'

    # determine manipulation
    values  = np.arange(0, 8, dtype=int)

    # network initliaizations
    init            = 10

    # preload dataframe
    preload = True

    if preload == False:

        # specification of image sequence
        t_steps         = 3
        dur             = [1, 1]
        start           = [0, 2]
    
        # retrieve timesteps
        t_steps_label = encode_timesteps(t_steps, start, dur)

        # hyperparameter specification
        batch_size      = 100

        # set home directory
        root_data           = root + 'models/dataset/'
        if dataset == 'mnist':
            testData        = datasets.MNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())
        elif dataset == 'fmnist':
            testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())

        # data loaders
        ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True, drop_last=True)

        # initiate accuracies
        accu = np.zeros((len(tempDynamics), init, len(values)))

        # loop and retrieve accuracies
        for iT, current_tempDynamics in enumerate(tempDynamics):

            for iInit in range(init):

                # save model
                model = cnn_feedforward(t_steps, current_tempDynamics)
                model.initialize_tempDynamics()
                model.load_state_dict(torch.load(root + 'models/weights/' + current_tempDynamics + '_' + str(iInit+1)))
                model.to(device)
                model.eval()

                # compute accuracies on test set (without shift)
                accu_current = np.zeros(len(ldrTest))
                for a, (imgs, lbls) in enumerate(ldrTest):

                    # initiate input
                    ax = torch.ones(batch_size, t_steps, imgs.shape[1], imgs.shape[2], imgs.shape[3])*0.5
                    ax = sequence_test(ax, imgs, 0, t_steps, t_steps_label)

                    # validate
                    testoutp = model.forward(ax.to(device))
                    predicy = torch.argmax(testoutp, dim=1).to('cpu')
                    accu_current[a] = (predicy == lbls).sum().item() / float(lbls.size(0))

                # compute accuracies on test set (with shift)
                accu_origin = np.mean(accu_current)

                for iV in range(len(values)):
                    accu_current = np.zeros(len(ldrTest))
                    for a, (imgs, lbls) in enumerate(ldrTest):

                        # initiate input
                        ax = torch.ones(batch_size, t_steps, imgs.shape[1], imgs.shape[2], imgs.shape[3])*0.5
                        ax = sequence_test(ax, imgs, values[iV], t_steps, t_steps_label)

                        # validate
                        testoutp = model.forward(ax.to(device))
                        predicy = torch.argmax(testoutp, dim=1).to('cpu')
                        accu_current[a] = (predicy == lbls).sum().item() / float(lbls.size(0))

                    # add to accuracies
                    accu[iT, iInit, iV] = (accu_origin - np.mean(accu_current)) * 100
                    accu[iT, iInit, iV] = (accu_origin - np.mean(accu_current)) * 100

        # save accuracies
        np.save(root + 'models/robustness', accu)

    # import accuracy
    accu = np.load(root + 'models/robustness.npy')

    # initiate figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [4, 1]}, sharey=True)

    sns.despine(offset=10)

    barWidth = 0.5

    fontsize_label      = 15
    fontsize_legend     = 10
    fontsize_title      = 15
    fontsize_ticks      = 15

    # visualize different contrast levels for repeated trials
    # cmap = plt.cm.get_cmap('Set2')
    cmap = ['#ACD39E', '#E49C39', '#225522', '#DF4828']
    # cmap = ['#4477AA', '#228833', '#66CCEE', '#CCBB44']
    start = 0

    sns.despine(offset=10)

    barWidth = 0.5

    fontsize_label      = 15
    fontsize_legend     = 10
    fontsize_title      = 15
    fontsize_ticks      = 15

    # visualize
    for iT, current_tempDynamic in enumerate(tempDynamics):

        # select data
        data_current = accu[iT, :, -1]

        # compute
        data_mean = data_current.mean()
        data_std = data_current.std()/math.sqrt(init)

        # visualize
        axs[1].bar(iT, data_mean, color=cmap[iT], width=barWidth, label=current_tempDynamic)

        # plot accuracies
        axs[1].plot([iT, iT], [data_mean - data_std, data_mean + data_std], color='black')
        axs[1].scatter(np.ones(init)*iT, data_current, color='grey', s=7, alpha=0.7)

    # visualize
    start = 0
    xtick_idx = list()
    for iT, current_tempDynamic in enumerate(tempDynamics):

        # define x-labels
        x = np.arange(len(values))+iT*10
        xtick_idx.append(x.tolist())

        # compute
        data_mean = accu[iT, :, :].mean(0)
        data_std = accu[iT, :, :].std(0)/math.sqrt(init)

        # plot accuracies
        for iV in range(len(values)):
            # if iV == len(values) - 1:
            #     axs[0].bar(x[iV], data_mean[iV], color=cmap[iT], edgecolor='red', width=barWidth, label=current_tempDynamic, alpha=0.45+0.07*iV)
            #     axs[0].plot([x[iV], x[iV]], [data_mean[iV] - data_std[iV], data_mean[iV] + data_std[iV]], color='black', alpha=0.45+0.07*iV)
            #     axs[0].scatter(np.ones(init)*x[iV], accu[iT, :, iV], color='grey', s=3, alpha=0.7)
            # else:
                axs[0].bar(x[iV], data_mean[iV], color=cmap[iT], width=barWidth, label=current_tempDynamic, alpha=0.45+0.07*iV)
                axs[0].plot([x[iV], x[iV]], [data_mean[iV] - data_std[iV], data_mean[iV] + data_std[iV]], color='black', alpha=0.45+0.07*iV)
                axs[0].scatter(np.ones(init)*x[iV], accu[iT, :, iV], color='grey', s=3, alpha=0.7)

    print(np.array(xtick_idx).flatten())

    # adjust axis
    for i in range(2):
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
        if i == 0:
            axs[i].set_xticks(np.array(xtick_idx).flatten())
            axs[i].set_xticklabels(np.tile(values, len(tempDynamics)))
            axs[i].set_ylabel('Drop in accuracy (%)', fontsize=fontsize_label)
        else:
            axs[i].set_xticks(np.arange(len(tempDynamics)))
            axs[i].set_xticklabels(tempDynamics_label, rotation=45, size=fontsize_ticks, ha='right', rotation_mode='anchor')

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig9B', dpi=300)
    plt.savefig(root + 'visualization/Fig9B.svg')

    # statistical testing
    iV = -1
    res = f_oneway(accu[0, :, iV], accu[1, :, iV], accu[2, :, iV], accu[3, :, iV])
    print(res)
    res = tukey_hsd(accu[0, :, iV], accu[1, :, iV], accu[2, :, iV], accu[3, :, iV])
    print(res)

    # res = f_oneway(accu[0, :, :].mean(1), accu[1, :, :].mean(1), accu[2, :, :].mean(1), accu[3, :, :].mean(1))
    # print(res)
    # res = tukey_hsd(accu[0, :, :].mean(1), accu[1, :, :].mean(1), accu[2, :, :].mean(1), accu[3, :, :].mean(1))
    # print(res)


if __name__ == '__main__':
    main()

