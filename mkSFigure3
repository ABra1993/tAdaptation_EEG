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
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.manifold import TSNE

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

fontsize_title          = 14
fontsize_legend         = 12
fontsize_label          = 12
fontsize_tick           = 10

def main():

    # input specification
    dataset                 = 'mnist'

    # specification of image sequence
    t_steps         = 3
    dur             = [1, 1]
    start           = [0, 2]

    # EEG paper (investigate contrast gain)
    tempDynamics            = ['add_supp', 'div_norm', 'lat_recurrence',  'lat_recurrence_mult']
    tempDynamics_lbl        = ['AS', 'DN', r'LR$_{A}$', r'LR$_{M}$']

    # noise
    adapters           = ['same', 'different']

    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]

    adapters_color      = [color_same[-1], color_diff[-1]]

    # define contrast values
    contrasts            = ['0.9']
    contrasts_value      = torch.Tensor([0.9])

    # retrieve accuracies or load
    preload = False

    # set task and dataset
    dataset = 'mnist'

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start=start, dur=dur)

    # other settings
    init        = 1

    epoch       = 10

    batch_size  = 1000

    layers = ['conv1', 'conv2', 'conv3']

    n_pca = 2

    ######################################## MODELS

    # set home directory
    root_data           = root + 'models/dataset/'
    if dataset == 'mnist':
        testData        = datasets.MNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())
    elif dataset == 'fmnist':
        testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform = transforms.ToTensor())

    # data loaders
    ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)

    # store pc
    layerwise_pca = np.zeros((len(tempDynamics), len(adapters), len(contrasts), len(layers), int(batch_size/2), n_pca))
    
    if preload == False:

        # loop and retrieve accuracies
        for iT, current_tempDynamics in enumerate(tempDynamics):

            for iInit in range(init):

                # save model
                model = cnn_feedforward(t_steps, current_tempDynamics)
                model.initialize_tempDynamics()
                model.load_state_dict(torch.load(root + 'models/weights/' + current_tempDynamics + '_' + str(iInit+1)))
                model.to(device)
                model.eval()

                for iC, contrast_value in enumerate(contrasts_value):
                    for a, (imgs, lbls) in enumerate(ldrTest):

                        # only first batch
                        if a != 0:
                            continue

                        # create input sequence
                        ax = sequence_train(F.adjust_contrast(imgs, contrast_value), t_steps, t_steps_label, False)

                        # forward
                        model.forward(ax.to(device))

                        # layer-wise PCA
                        for iL in range(len(layers)):
                            for iA in range(len(adapters)):
                                
                                # retrieve activations
                                if iA == 0:
                                    current_actvs = model.actvs[iL][2][:int(batch_size/2), :, :, :].detach().cpu().numpy()
                                elif iA == 1:
                                    current_actvs = model.actvs[iL][2][int(batch_size/2):, :, :, :].detach().cpu().numpy()
                                current_actvs = current_actvs.reshape(int(batch_size/2), current_actvs.shape[1]*current_actvs.shape[2]*current_actvs.shape[3])

                                # PCA
                                # pca = PCA(n_components=n_pca)
                                # current_actvs_transformed = pca.fit_transform(current_actvs)
                                tsne = TSNE(n_components=n_pca, perplexity=50, n_iter=500)
                                current_actvs_transformed = tsne.fit_transform(current_actvs)
                                layerwise_pca[iT, iA, iC, iL, :, :] = current_actvs_transformed

        # save accuracies
        np.save(root + 'models/tSNE', layerwise_pca)

    # import accuracy
    layerwise_pca = np.load(root + 'models/tSNE.npy')

    # initiate figure
    _, axs = plt.subplots(3, len(tempDynamics), figsize=(8, 4))

    # plot pca, torch.Tensor(t_steps, len(adapters), len(contrasts), len(layers), int(batch_size/2), n_pca)
    color_axes = 'lightgrey'
    for iT, current_tempDynamics in enumerate(tempDynamics):
        for iL, layer in enumerate(layers):
            for iA, adapter in enumerate(adapters):
                
                # select
                current_actvs = layerwise_pca[iT, iA, :, iL, :, :].squeeze()

                # visualize
                axs[iL, iT].scatter(current_actvs[:, 0], current_actvs[:, 1], s=20, facecolor=adapters_color[iA], edgecolors='white', alpha=0.35)

            # adjust axes
            axs[iL, iT].set_xticks([])
            axs[iL, iT].set_yticks([])
            axs[iL, iT].spines['bottom'].set_color(color_axes)
            axs[iL, iT].spines['top'].set_color(color_axes)
            axs[iL, iT].spines['right'].set_color(color_axes)
            axs[iL, iT].spines['left'].set_color(color_axes)
            if iT == 0:
                axs[iL, iT].set_ylabel(layer)

        # adjust axes
        axs[0, iT].set_title(tempDynamics_lbl[iT])
            
    # save figure
    plt.tight_layout()
    plt.savefig(root + 'visualization/SFig3', dpi=300)
    plt.savefig(root + 'visualization/SFig3.svg')
    # plt.close()


if __name__ == '__main__':
    main()

