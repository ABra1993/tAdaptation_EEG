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
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import f_oneway
from scipy.stats import tukey_hsd

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

def cross_validate_svm(activations, labels, cv):
    scaler = StandardScaler()
    clf = SVC(kernel='linear')
    pipeline = make_pipeline(scaler, clf)
    scores = cross_val_score(pipeline, activations, labels, cv=cv)
    return scores

# define root
# root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'
root                = '/home/amber/Documents/organize_code/nAdaptation_EEG_git/'

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# set the seed
seed = 32
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# fontsizes 
global fontsize_title
global fontsize_legend
global fontsize_label
global fontsize_tick

fontsize_title          = 11
fontsize_legend         = 10
fontsize_label          = 10
fontsize_tick           = 8

def main():

    # input sequences
    # sequences       = ['ABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT']
    sequences       = ['AAAAAAAAAAAAAAABTTTTT']
    seq_length      = [len(seq) for seq in sequences]
    print(seq_length)
    
    # EEG paper (investigate contrast gain)
    tempDynamics                = ['none']
    tempDynamics_label          = [r'feedforward']

    tempDynamics                = ['none', 'add_supp', 'l_recurrence_A', 'div_norm_clamp']#, 'div_norm_scale']
    tempDynamics_label          = [r'feedforward', r'$add.$ $supp.$', r'$lat.$ $rec.$', r'$div.$ $norm.$'] #, r'+ $divisive$ $normalization_{scale}$']

    # input specification
    dataset                 = 'mnist'

    # number of layers
    layers      = ['conv1', 'conv2', 'conv3', 'fc1']
    n_layer     = len(layers)
    
    # noise
    adapters           = ['same', 'different']

    # other settings
    init        = 5
    batch_size  = 600

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # define contrast values
    contrast_values      = torch.Tensor([0.1, 0.5, 0.9, 1])
    # contrast_values      = torch.Tensor([1])

    # PLOT MODEL PERFORMANCE
    start               = 0.8
    end                 = 0.3
    color_same          = plt.colormaps['Blues'] #cm.get_cmap('tab20c')
    color_same          = [color_same(i) for i in np.linspace(end, start, len(contrast_values))]

    color_diff          = plt.colormaps['YlOrBr'] #cm.get_cmap('tab20b')
    color_diff          = [color_diff(i) for i in np.linspace(end, start, len(contrast_values))]

    colors_adapt        = [color_same, color_diff]

    # retrieve accuracies or load
    preload = False # if False the accuracies will be shown from the folder /models/performance_variableContrast_*, otherwise accuracies wil be re-computed

    # layer index
    layer_idx = 0

    if preload == False:

        # set home directory
        root_data           = root + 'models/dataset/'
        if dataset == 'mnist':
            testData        = datasets.MNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())
        elif dataset == 'fmnist':
            testData        = datasets.FashionMNIST(root=root_data, download=False, train=False, transform=transforms.ToTensor())

        # data loaders
        ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)

        # perform cross-validation
        for a, (imgs, lbls) in enumerate(ldrTest):

            # only perform decoding over first batch
            if a != 0:
                break

            for iSeq, sequence in enumerate(sequences):

                # print progress
                print('Sequence: ' + sequence)

                # retrieve sequence
                t_steps = len(sequence)
                t_steps_label = encode_timesteps(t_steps, sequence)
                print(t_steps_label)

                for iInit in range(init):

                    # initiat figure
                    _, axs = plt.subplots(4, 1, figsize=(5, 7))

                    for iT, current_tempDynamics in enumerate(tempDynamics):

                        # set
                        axs[iT].tick_params(axis='both', labelsize=fontsize_tick)
                        axs[iT].spines['top'].set_visible(False)
                        axs[iT].spines['right'].set_visible(False)
                        axs[iT].set_xticks(np.arange(t_steps))
                        axs[iT].set_xticklabels(np.arange(t_steps)+1, rotation=45)
                        axs[iT].set_title(tempDynamics_label[iT], color='gray', fontsize=fontsize_title)
                        axs[iT].axvspan(0, 14, alpha=0.5, color='lightgray')
                        axs[iT].axvspan(15, 20, alpha=0.5, color='lightgray')

                        # print progress
                        print('Current temp. dynamics: ', current_tempDynamics)
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
                        model.eval()

                        # iterate over contrast values and obtain decoding accuracy
                        for iC, contrast_value in enumerate(contrast_values):

                            # create input sequence
                            _, ax = create_sequence(imgs=imgs, 
                                                t_steps=t_steps, 
                                                t_steps_label=t_steps_label, 
                                                create_sample=False, 
                                                contrast=contrast_value, 
                                                contrast_low=None, 
                                                contrast_high=None)
                            # validate
                            _ = model.forward(ax)
                            actvs = model.actvs

                            # extract activations
                            actvs_avgs = np.zeros((len(adapters), t_steps))
                            for t in range(t_steps):
                                actvs_avgs[0, t] = actvs[layer_idx][t][:int(batch_size/2), :, :, :].mean()
                                actvs_avgs[1, t] = actvs[layer_idx][t][int(batch_size/2):, :, :, :].mean()
                                # actvs_avgs[0, t] = actvs[layer_idx][t][0, 3, 3, 3]
                                # actvs_avgs[1, t] = actvs[layer_idx][t][-1, 3, 3, 3]

                            # plot
                            for iA, _ in enumerate(adapters):
                                axs[iT].plot(actvs_avgs[iA, :], color=colors_adapt[iA][iC])

                    # save figure
                    plt.tight_layout()
                    plt.savefig(root + 'visualization/SFig4-7_init' + str(iInit+1), dpi=300)
                    plt.savefig(root + 'visualization/SFig4-7_init' + str(iInit+1) + '.svg')


if __name__ == "__main__":
    main()
