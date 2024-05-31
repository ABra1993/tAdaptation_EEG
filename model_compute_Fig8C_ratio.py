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
from scipy import stats
import matplotlib.pyplot as plt
import random
import seaborn as sns
from matplotlib.legend_handler import HandlerLineCollection, HandlerTuple

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

# define root
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def main():

    # specification of image sequence
    t_steps         = 3
    dur             = [1, 1]
    start           = [0, 2]

    color_same          = plt.cm.get_cmap('tab20c')
    color_same          = color_same.colors[:3]

    color_diff          = plt.cm.get_cmap('tab20b')
    color_diff          = color_diff.colors[8:11]   

    # # EEG paper (investigate contrast gain)
    tempDynamics            = ['none', 'add_supp', 'div_norm', 'lat_recurrence', 'lat_recurrence_mult']

    # set task and dataset
    dataset         = 'mnist'
    adapters        = ['same', 'different']

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start, dur)

    # other settings
    init        = 5
    batch_size  = 100

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # preload activations
    preload = False

    # set home directory
    root_data           = root + 'models/dataset/'
    if dataset == 'mnist':
        testData        = datasets.MNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())
    elif dataset == 'fmnist':
        testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())

    # data loaders
    ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)
    
    # define input shape (channels * height * width)
    input_shape = testData[0][0].shape
    print('Input shape: ', input_shape)

    # define contrast values
    contrast_values      = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    # training length
    epch = 10

    # compute
    if preload == False:

        # initiate frame to save accu
        actvs   = torch.zeros((len(tempDynamics), len(contrast_values), init, len(adapters), 2)) #  0: figure/ 1: ground

        # loop and retrieve accuracies
        for iT, current_tempDynamics in enumerate(tempDynamics):

            for iC, contrast_value in enumerate(contrast_values):

                for iInit in range(init):

                    # save model
                    model = cnn_feedforward(t_steps, current_tempDynamics)
                    model.initialize_tempDynamics()
                    model.load_state_dict(torch.load(root + 'models/weights/' + current_tempDynamics + '_' + str(iInit+1)))
                    model.to(device)
                    model.eval()
                    
                    # evaluate
                    actvs_current    = torch.zeros((len(ldrTest), 2, 2))
                    for a, (imgs, lbls) in enumerate(ldrTest):

                        # extract activations from first convolutional layer to do figure/ground segmentation
                        conv1_imgs = model.conv1(imgs.to(device))
                        conv1_imgs = conv1_imgs.detach().cpu().mean(1).unsqueeze(1)

                        # compute most frequent pixel value
                        conv1_imgs_flatten = conv1_imgs.view(conv1_imgs.size(0), -1)
                        mode_values, _ = torch.mode(conv1_imgs_flatten, dim=1)

                        # create masks
                        mask_figure = (conv1_imgs == mode_values[:, None, None, None])
                        mask_ground = (conv1_imgs != mode_values[:, None, None, None])

                        # segment pixels
                        img_mask_figure         = torch.where(mask_figure, torch.tensor(0.0, device=conv1_imgs.device), torch.tensor(1.0, device=conv1_imgs.device))
                        img_mask_ground         = torch.where(mask_ground, torch.tensor(0.0, device=conv1_imgs.device), torch.tensor(1.0, device=conv1_imgs.device))
                        # print(img_mask_figure.shape)

                        # create input sequence
                        imgs = F.adjust_contrast(imgs, contrast_value)
                        ax = sequence_train(imgs, t_steps, t_steps_label, False)
                        
                        # validate
                        _ = model.forward(ax.to(device))

                        # get activations for last timestep
                        ax_conv1 = model.actvs[0][2].detach().cpu().mean(1).unsqueeze(1) # first layer last timestep (containing target digit)
                        # print(ax_conv1.shape)

                        # compute figure and ground
                        ax_conv1_figure     = torch.mul(ax_conv1, img_mask_figure)
                        ax_conv1_ground     = torch.mul(ax_conv1, img_mask_ground)

                        # save activations
                        actvs_current[a, 0, 0] = torch.mean(ax_conv1_figure[:int(imgs.shape[0]/2):, :, :, :])
                        actvs_current[a, 0, 1] = torch.mean(ax_conv1_ground[:int(imgs.shape[0]/2):, :, :, :])

                        actvs_current[a, 1, 0] = torch.mean(ax_conv1_figure[int(imgs.shape[0]/2):, :, :, :])
                        actvs_current[a, 1, 1] = torch.mean(ax_conv1_ground[int(imgs.shape[0]/2):, :, :, :])


                    # save activations
                    actvs[iT, iC, iInit, 0, 0] = torch.mean(actvs_current[a, 0, 0])
                    actvs[iT, iC, iInit, 0, 1] = torch.mean(actvs_current[a, 0, 1])

                    actvs[iT, iC, iInit, 1, 0] = torch.mean(actvs_current[a, 1, 0])
                    actvs[iT, iC, iInit, 1, 1] = torch.mean(actvs_current[a, 1, 1])

        # save activations
        torch.save(actvs, root + 'models/ratio_actvs')

    # import activations
    actvs = torch.load(root + 'models/ratio_actvs')
    print(actvs.shape)

if __name__ == '__main__':
    main()

