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

fontsize_title          = 13
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

    # set temporal dynamics
    tempDynamics            = ['add_supp', 'div_norm', 'lat_recurrence',  'lat_recurrence_mult']

    # noise
    adapters           = ['same', 'different']

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start=start, dur=dur)

    # other settings
    init        = 1
    batch_size  = 1000

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # set home directory
    root_data           = root + 'models/dataset/'
    if dataset == 'mnist':
        testData        = datasets.MNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())
    elif dataset == 'fmnist':
        testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())

    # data loaders
    ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)

    # define input shape (channels * height * width)
    input_shape = testData[0][0].shape

    # define contrast values
    # contrast_values      = torch.Tensor([0.5, 0.7, 0.9])
    contrast_values      = np.array([0.9])

    # initiate frame to save accu
    actvs   = torch.zeros((len(tempDynamics), len(contrast_values), init, input_shape[0], 24, 24, len(adapters)))

    # color
    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    colors_adapt        = [color_same, color_diff]

    # loop and retrieve accuracies
    epch        = 10

    # initiate figure
    fig, axs = plt.subplots(2, 2, figsize=(5, 4), sharey=False)
    sns.despine(offset=5)

    # visualize
    for a, (imgs, lbls) in enumerate(ldrTest):

        if a == 0:

            iT = 0
            for i in range(2):
                for j in range(2):

                    for iInit in range(init):

                        # save model
                        model = cnn_feedforward(t_steps, tempDynamics[iT])
                        model.initialize_tempDynamics()
                        model.load_state_dict(torch.load(root + 'models/weights/' + tempDynamics[iT] + '_' + str(iInit+1)))
                        model.to(device)
                        model.eval()

                        for iC, contrast_value in enumerate(contrast_values):

                            # create input sequence
                            imgs = F.adjust_contrast(imgs, contrast_value)
                            ax = sequence_train(imgs, t_steps, t_steps_label, False)

                            # validate
                            _ = model.forward(ax.to(device))

                            # get activations for last timestep
                            ax_conv1 = model.actvs[0][2].detach().cpu().mean(1) # average over featuremaps
                            ax_conv1 = ax_conv1.reshape(ax_conv1.shape[0], ax_conv1.shape[1]*ax_conv1.shape[2])

                            # # creae bins
                            bin_width = 0.001  # Adjust bin width as needed
                            bins = np.arange(torch.min(ax_conv1), torch.max(ax_conv1) + bin_width, bin_width)  # Define bin edges

                            # extract values
                            histograms = np.apply_along_axis(lambda x: np.histogram(x, bins=bins)[0], axis=1, arr=ax_conv1)
                            print(histograms.shape)

                            # compute distributions per adapter
                            histograms_per_adapter = np.zeros((len(adapters), histograms.shape[1]))
                            histograms_per_adapter[0, :] = np.mean(histograms[:int(batch_size/2), :], 0)
                            histograms_per_adapter[1, :] = np.mean(histograms[int(batch_size/2):, :], 0)
                            # histograms_per_adapter = histograms_per_adapter/np.max(histograms_per_adapter)

                            # Plot the histograms
                            n_samples = histograms_per_adapter.shape[0]

                            for iA, adapter in enumerate(adapters):
                                if iA < n_samples:
                                    axs[i, j].bar(bins[:-1], histograms_per_adapter[iA], width=bin_width, align='edge', color=colors_adapt[iA][0], alpha=0.5)
                                    
                                    # adjust axes
                                    axs[i, j].spines['top'].set_visible(False)
                                    axs[i, j].spines['right'].set_visible(False)
                                    if i == 1:
                                        axs[i, j].set_xlabel('Pixel value', fontsize=fontsize_label)
                                    if j == 0:
                                        axs[i, j].set_ylabel('Frequency', fontsize=fontsize_label)

                            # increment count
                            iT = iT + 1

    # save plot
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig8D', dpi=300)
    plt.savefig(root + 'visualization/Fig8D.svg')
    # plt.close()

if __name__ == '__main__':
    main()

