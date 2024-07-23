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

def main():

    # input specification
    dataset                 = 'mnist'

    # specification of image sequence
    t_steps         = 3
    dur             = [1, 1]
    start           = [0, 2]

    # set temporal dynamics
    tempDynamics       = ['add_supp', 'div_norm', 'lat_recurrence',  'lat_recurrence_mult']

    # noise
    adapters           = ['same', 'different']

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start, dur)

    # other settings
    init        = 1
    batch_size  = 2

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print(30*'--')
    print('\n')

    # set home directory
    root_data           = root + 'models/dataset/'
    if dataset == 'mnist':
        testData        = datasets.MNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())
    elif dataset == 'fmnist':
        testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())

    # data loaders
    # ldrTrain            = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True)
    ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)
    
    # define input shape (channels * height * width)
    input_shape = testData[0][0].shape
    test_img    = testData[0][0]

    # define contrast values
    contrast_values      = np.array([0.9])

    # initiate frame to save accu
    actvs   = torch.zeros((len(tempDynamics), len(contrast_values), init, input_shape[0], 24, 24, len(adapters)))

    for a, (imgs, _) in enumerate(ldrTest):

        if a == 0:

            # set image
            imgs[0, :, :, :] = test_img
            imgs[1, :, :, :] = test_img

            for iT, current_tempDynamics in enumerate(tempDynamics):

                for iInit in range(init):

                    # save model
                    model = cnn_feedforward(t_steps, current_tempDynamics)
                    model.initialize_tempDynamics()
                    model.load_state_dict(torch.load(root + 'models/weights/' + current_tempDynamics + '_' + str(iInit+1)))
                    model.to(device)
                    model.eval()

                    for iC, contrast_value in enumerate(contrast_values):

                        # create input sequence
                        imgs = F.adjust_contrast(imgs, contrast_value)
                        ax = sequence_train(imgs, t_steps, t_steps_label, False)

                        # validate
                        _ = model.forward(ax.to(device))
                        # print(pred)

                        # get activations for last timestep
                        ax_conv1 = model.actvs[0][2].detach().cpu().mean(1).unsqueeze(1) # first layer last timestep (containing target digit)

                        # plot units
                        actvs[iT, iC, iInit, :, :, :, 0] = ax_conv1[0, :, :, :]/torch.max(ax_conv1[0, :, :, :])
                        actvs[iT, iC, iInit, :, :, :, 1] = ax_conv1[-1, :, :, :]/torch.max(ax_conv1[-1, :, :, :])

    # initiate figure
    _, axs = plt.subplots(len(contrast_values), len(tempDynamics)*2, figsize=(10, 4))

    # plot activations
    size = 24
    vmin = 0
    vmax = 1
    for iT, current_tempDynamics in enumerate(tempDynamics):                    
        for iC, contrast_value in enumerate(contrast_values):

            # plot activations
            axs[iT*2].imshow(actvs[iT, iC, :, :, :, :, 0].mean(0).reshape(size, size, 1), vmin=vmin, vmax=vmax, cmap='plasma')
            axs[iT*2+1].imshow(actvs[iT, iC, :, :, :, :, 1].mean(0).reshape(size, size, 1), vmin=vmin, vmax=vmax, cmap='plasma')

            # adjust axes
            axs[iT*2].axis('off')
            axs[iT*2+1].axis('off')

    # save plot
    plt.tight_layout()
    plt.savefig(root + 'visualization/Fig9A', dpi=300)
    plt.savefig(root + 'visualization/Fig9A.svg')

if __name__ == '__main__':
    main()

