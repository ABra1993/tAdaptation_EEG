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
from mkFigure9_utils import *

# define root
# root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'
root                = '/home/amber/Documents/organize_code/nAdaptation_EEG_git/'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():

    # input sequences
    # sequences = ['ABT', 'AAABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AABTT', 'AAABTTT', 'AAAABTTTT', 'AAAAAAAABTTTTTTTT']
    sequences = ['ABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT'] #, 'AAAAAAAAAAAAAAAAAABTTTTTT']
    seq_length = [len(seq) for seq in sequences]

    # other settings
    init        = 5

    # EEG paper (investigate contrast gain)
    # tempDynamics                = ['none', 'add_supp', 'l_recurrence_A', 'div_norm_clamp'] #, 'div_norm_scale']
    # tempDynamics_label          = [r'feedforward', r'+ $additive$ $suppression$', r'+ $lateral$ $recurrence_{A}$', r'+ $divisive$ $normalization$'] #, r'+ $divisive$ $normalization_{scale}$']

    tempDynamics                = ['add_supp', 'l_recurrence_A', 'div_norm_clamp'] #, 'div_norm_scale']
    tempDynamics_label          = [r'$add. $ $supp.$', r'$lat.$ $rec.$', r'$div.$ $norm.$'] #, r'+ $divisive$ $normalization_{scale}$']

    # set task and dataset
    dataset = 'mnist'

    # determine manipulation
    values  = np.arange(0, 8, dtype=int)

    # network initliaizations
    init            = 5

    # preload dataframe
    preload = True

    if preload == False:

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

        for iSeq, sequence in enumerate(sequences):

            # print progress
            print('Sequence: ' + sequence)

            # retrieve sequence
            t_steps = len(sequence)
            t_steps_label = encode_timesteps(t_steps, sequence)
            print(t_steps_label)

            # loop and retrieve accuracies
            for iT, current_tempDynamics in enumerate(tempDynamics):

                # print progress
                print('Current dynamics: ' + current_tempDynamics)

                # initiate accuracies
                accu = np.zeros((init, len(values)))    

                for iInit in range(init):

                    print('Init: ' + str(iInit))

                    # initiate model
                    model = cnn_feedforward()

                    # initiate recurrence
                    if tempDynamics != 'none':
                        model.initialize_tempDynamics(current_tempDynamics)

                    # init timesteps
                    model.init_t_steps(t_steps)

                    # load weights
                    model.load_state_dict(torch.load(root + 'models/weights/' + sequence + '/' + current_tempDynamics + '_' + str(iInit+1)))
                    model.to(device)

                    # # compute accuracies on test set (without shift)
                    # accu_current = np.zeros(len(ldrTest))
                    # for a, (imgs, lbls) in enumerate(ldrTest):

                    #     # initiate input
                    #     ax = torch.ones(batch_size, t_steps, imgs.shape[1], imgs.shape[2], imgs.shape[3])*0.5
                    #     ax = sequence_test(ax, imgs, 0, t_steps, t_steps_label)

                    #     # validate
                    #     testoutp = model.forward(ax.to(device))
                    #     predicy = torch.argmax(testoutp, dim=1).to('cpu')
                    #     accu_current[a] = (predicy == lbls).sum().item() / float(lbls.size(0))

                    # # compute accuracies on test set (with shift)
                    # accu_origin = np.mean(accu_current)

                    for iV in range(len(values)):
                        accu_current = np.zeros(len(ldrTest))
                        for a, (imgs, lbls) in enumerate(ldrTest):

                            # initiate input
                            ax = sequence_test(imgs, values[iV], t_steps, t_steps_label)

                            # validate
                            testoutp = model.forward(ax)
                            predicy = torch.argmax(testoutp, dim=1).to('cpu')
                            accu_current[a] = (predicy == lbls).sum().item() / float(lbls.size(0))

                        # add to accuracies
                        accu[iInit, iV] = np.mean(accu_current)
                        accu[iInit, iV] = np.mean(accu_current)

                # save accuracies
                np.save(root + 'models/robustness/' + sequence + '_' + current_tempDynamics, accu)

    # import accuracy
    accu = np.zeros((len(sequences), len(tempDynamics), init, len(values)))
    for iT, current_tempDynamics in enumerate(tempDynamics):
        for iSeq, sequence in enumerate(sequences):
            accu[iSeq, iT, :, :] = np.load(root + 'models/robustness/' + sequence + '_' + current_tempDynamics + '.npy')

    # FIGURE 9 -----------------------------------------------

    # initiate figure
    fig, axs = plt.subplots(1, len(tempDynamics)+1, figsize=(10, 3), gridspec_kw={'width_ratios': [1, 1, 1, 4]}, sharey=True)
    sns.despine(offset=10)

    barWidth = 0.5
    barWidth_avg = 0.8

    fontsize_label      = 10
    fontsize_legend     = 10
    fontsize_title      = 13
    fontsize_ticks      = 10

    # color maps
    start                       = 0.8
    end                         = 0.3

    cmap_add_supp               = plt.colormaps['Purples'] #cm.get_cmap('tab20c')
    cmap_add_supp               = [cmap_add_supp(i) for i in np.linspace(start, end, len(sequences))]

    cmap_lat_rec                = plt.colormaps['Greens'] #cm.get_cmap('tab20c')
    cmap_lat_rec                = [cmap_lat_rec(i) for i in np.linspace(start, end, len(sequences))]

    cmap_div_norm               = plt.colormaps['Oranges'] #cm.get_cmap('tab20c')
    cmap_div_norm               = [cmap_div_norm(i) for i in np.linspace(start, end, len(sequences))]

    color = [cmap_add_supp, cmap_div_norm, cmap_lat_rec]

    # store avgs
    avgs = np.zeros((len(sequences), len(tempDynamics), init))

    # visualize
    xtick_idx = list()
    for iT, current_tempDynamic in enumerate(tempDynamics):

        print('Temp. dyn.: ', current_tempDynamic)

        # define x-labels
        x = np.arange(len(values))+iT*10
        xtick_idx.append(x.tolist())

        # visualize
        for iSeq, sequence in enumerate(sequences):

            if (iSeq == 0) | (iSeq == len(sequences)-1):

                # select data
                data_current = accu[iSeq, iT, :, :].mean(1)/np.max(accu[iSeq, iT, :, :], 1)

                data_mean = data_current.mean()
                data_std = data_current.std()/math.sqrt(init)

                # save averages
                avgs[iSeq, iT, :] = data_current

                # visualize
                if iSeq == 0:
                    axs[iT].bar(iSeq, data_mean, color=color[iT][-1-iSeq], width=barWidth_avg, label=current_tempDynamic)
                    axs[iT].plot([iSeq, iSeq], [data_mean - data_std, data_mean + data_std], color='black')
                    # axs[1+iT].scatter(np.ones(init)*iSeq, data_current, color='grey', s=7, alpha=0.7)
                else:
                    axs[iT].bar(1, data_mean, color=color[iT][-1-iSeq], width=barWidth_avg, label=current_tempDynamic)
                    axs[iT].plot([1, 1], [data_mean - data_std, data_mean + data_std], color='black')
                    # axs[1+iT].scatter(np.ones(init)*iSeq, data_current, color='grey', s=7, alpha=0.7)

            # select data
            data_current = accu[-1 - iSeq, iT, :, :].transpose()/np.max(accu[-1-iSeq, iT, :, :], 1)

            data_mean = data_current.mean(1)
            data_std = data_current.std(1)/math.sqrt(init)

            # plot accuracies
            for iV in range(len(values)):
                # if iV == len(values) - 1:
                #     axs[0].bar(x[iV], data_mean[iV], color=cmap[iT], edgecolor='red', width=barWidth, label=current_tempDynamic, alpha=0.45+0.07*iV)
                #     axs[0].plot([x[iV], x[iV]], [data_mean[iV] - data_std[iV], data_mean[iV] + data_std[iV]], color='black', alpha=0.45+0.07*iV)
                #     axs[0].scatter(np.ones(init)*x[iV], accu[iT, :, iV], color='grey', s=3, alpha=0.7)
                # else:
                    axs[len(tempDynamics)].bar(x[iV], data_mean[iV], color=color[iT][iSeq], width=barWidth, label=current_tempDynamic) #, alpha=0.45+0.07*iV)
                    # axs[0].plot([x[iV], x[iV]], [data_mean[iV] - data_std[iV], data_mean[iV] + data_std[iV]], color='black', alpha=0.45+0.07*iV)
                    # axs[0].scatter(np.ones(init)*x[iV], data_current[iV, :], color='grey', s=3, alpha=0.7)

        # update axes
        if iT == 0:
            axs[iT].set_ylabel('Accuracy (normalized)', fontsize=fontsize_label)
        axs[iT].tick_params(axis='both', labelsize=fontsize_ticks)
        axs[iT].set_xticks([0, 1])
        axs[iT].set_xticklabels(['Short', 'Long'], ha='right', fontsize=fontsize_ticks, rotation=45)
        axs[iT].set_title(tempDynamics_label[iT], color='gray', fontsize=fontsize_title)
        if iT == 1:
            axs[iT].set_xlabel('Sequence length', fontsize=fontsize_label)

    # adjust axes
    axs[len(tempDynamics)].tick_params(axis='both', labelsize=fontsize_ticks)
    axs[len(tempDynamics)].set_xticks(np.array(xtick_idx).flatten())
    axs[len(tempDynamics)].set_xticklabels(np.tile(values, len(tempDynamics)))
    axs[len(tempDynamics)].set_ylabel('Accuracy (normalized)', fontsize=fontsize_label)
    axs[len(tempDynamics)].set_xlabel('Spatial shift noise (pxl)', fontsize=fontsize_label)
    # axs[0].set_xticklabels(tempDynamics_label, rotation=45, size=fontsize_ticks, ha='right', rotation_mode='anchor')

    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig9', dpi=300)
    plt.savefig(root + 'visualization/Fig9.svg')

    # # statistical testing
    # for iT, current_tempDynamic in enumerate(tempDynamics):

    #     print('Temp. dyn.: ', current_tempDynamic)

    #     res = f_oneway(avgs[0, iT, :], avgs[1, iT, :], avgs[2, iT, :], avgs[3, iT, :], avgs[4, iT, :])
    #     print(res)

    #     if res[1] < 0.05:
    #         res = tukey_hsd(avgs[0, iT, :], avgs[1, iT, :], avgs[2, iT, :], avgs[3, iT, :], avgs[4, iT, :])
    #         print(res)

    # statistical testing
    for iT, current_tempDynamic in enumerate(tempDynamics):

        print('Temp. dyn.: ', current_tempDynamic)

        res = stats.ttest_ind(avgs[0, iT, :], avgs[4, iT, :])
        print(res)

    # statistical testing
    res = f_oneway(avgs[0, 0, :], avgs[0, 1, :], avgs[0, 2, :])
    print(res)

    if res[1] < 0.05:
        res = tukey_hsd(avgs[0, 0, :], avgs[0, 1, :], avgs[0, 2, :])
        print(res)

    # statistical testing
    res = f_oneway(avgs[4, 0, :], avgs[4, 1, :], avgs[4, 2, :])
    print(res)

    if res[1] < 0.05:
        res = tukey_hsd(avgs[4, 0, :], avgs[4, 1, :], avgs[4, 2, :])
        print(res)

    # # closest pixel shift
    # res = f_oneway(avgs[0, 0, :], avgs[0, 1, :], avgs[0, 2, :])
    # print(res)
    # res = tukey_hsd(avgs[0, 0, :], avgs[0, 1, :], avgs[0, 2, :])
    # print(res)

    # # furthest pixel shift
    # res = f_oneway(avgs[3, 0, :], avgs[3, 1, :], avgs[3, 2, :])
    # print(res)
    # res = tukey_hsd(avgs[3, 0, :], avgs[3, 1, :], avgs[3, 2, :])
    # print(res)

if __name__ == '__main__':
    main()