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

# fontsizes
fontsize_title          = 12
fontsize_legend         = 9
fontsize_label          = 10
fontsize_tick           = 9

def main():

    # input sequences
    sequences       = ['ABT', 'AAAAAAAAAAAAAAABTTTTT']
    seq_length      = [len(seq) for seq in sequences]
    
    # EEG paper (investigate contrast gain)
    tempDynamics                = ['add_supp', 'l_recurrence_A', 'div_norm_clamp']#, 'div_norm_scale']
    tempDynamics_label          = [r'$add.$ $supp.$', r'$lat.$ $rec.$', r'$div.$ $norm.$'] #, r'+ $divisive$ $normalization_{scale}$']

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
    contrast_values      = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # decoding accuracy
    cv = 3  # cross-validations

    # retrieve accuracies or load
    preload = True # if False the accuracies will be shown from the folder /models/performance_variableContrast_*, otherwise accuracies wil be re-computed

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

                for iT, current_tempDynamics in enumerate(tempDynamics):
                    
                    # store decoding accuracies
                    dec_acc = np.zeros((len(contrast_values), len(adapters), len(layers), init))

                    # print progress
                    print('Current temp. dynamics: ', current_tempDynamics)

                    for iInit in range(init):

                        if iInit != 2:
                            continue

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

                            for iA, _ in enumerate(adapters):
                                for iL in range(n_layer):

                                    # select data
                                    if iA == 0:
                                        if iL == 3:
                                            X = actvs[iL][len(sequence)-1][:int(batch_size/2), :]
                                        else:
                                            X = actvs[iL][len(sequence)-1][:int(batch_size/2), :, :, :]
                                            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
                                        y = lbls[:int(batch_size/2)]
                                    else:
                                        if iL == 3:
                                            X = actvs[iL][len(sequence)-1][int(batch_size/2):, :]
                                        else:
                                            X = actvs[iL][len(sequence)-1][int(batch_size/2):, :, :, :]
                                            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
                                        y = lbls[int(batch_size/2):]

                                    # perform decoding analysis np.zeros((len(tempDynamics), len(contrast_values), len(adapters), len(layers), init))
                                    temp = cross_validate_svm(X.detach().cpu().numpy(), y.detach().cpu().numpy(), cv)
                                    dec_acc[iC, iA, iL, iInit] = temp.mean()
                                    # print(temp.mean())

                    # save accuracies
                    np.save(root + 'models/decoding/' + sequence + '_' + current_tempDynamics, dec_acc)

    # visualize
    fig, axs = plt.subplots(1, len(tempDynamics), figsize=(10, 2), sharey=True)
    sns.despine(offset=10)

    contrast_values      = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # PLOT MODEL PERFORMANCE
    start               = 0.8
    end                 = 0.3
    color_same          = plt.colormaps['Blues'] #cm.get_cmap('tab20c')
    color_same          = [color_same(i) for i in np.linspace(start, end, len(sequences))]

    color_diff          = plt.colormaps['YlOrBr'] #cm.get_cmap('tab20b')
    color_diff          = [color_diff(i) for i in np.linspace(start, end, len(sequences))]

    colors_adapt        = [color_same, color_diff]

    # layer index
    layer_idx = 3
 
    # axes settings
    linewidth = 2

    # # PLOT MODEL PERFORMANCE
    for iT, current_tempDynamics in enumerate(tempDynamics):
        for iSeq, sequence in enumerate(sequences):
                
            ######################################## EXTRACT MODEL BEHAVIOUR
            accu = np.load(root + 'models/decoding/' + sequence + '_' + current_tempDynamics + '.npy') 

            # plot networks with temporal adaptation
            for iA, adapter in enumerate(adapters):

                # select data
                data_mean   = accu[:, iA, layer_idx,  :].mean(1)
                # data_std    = np.std(accu[:, iA, layer_idx,  :].mean(0))/math.sqrt(init)

                axs[iT].plot(contrast_values, data_mean, color=colors_adapt[iA][iSeq], linewidth=linewidth, label=adapter, zorder=10)
                # axs[iT].plot([iA+1, iA+1], [data_mean - data_std, data_mean + data_std], color=adapters_color[iA], zorder=-1)
                # sns.stripplot(x=np.ones(init)*iA+1, y=accu[:, iA, layer_idx,  :].mean(0), jitter=True, ax=axs[iT+1], color=adapters_color[iA], size=markersize_small, alpha=0.5, native_scale=True, legend=False, zorder=-10)

        # adjust axes
        axs[iT].tick_params(axis='both', labelsize=fontsize_tick)
        axs[iT].spines['top'].set_visible(False)
        axs[iT].spines['right'].set_visible(False)
        # axs[iT].set_xlabel('Contrast target (%)', fontsize=fontsize_label)
        axs[iT].set_xticks(contrast_values)
        axs[iT].set_xticklabels([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], rotation=45)
        axs[iT].set_ylabel('Decoding accuracy', fontsize=fontsize_label)
        axs[iT].set_title(tempDynamics_label[iT], color='gray', fontsize=fontsize_title)


    # save figure
    if layer_idx < 3:
        plt.savefig(root + 'visualization/Fig8E_conv' + str(layer_idx+1), dpi=300)
        plt.savefig(root + 'visualization/Fig8E_conv' + str(layer_idx+1) + '.svg')
    else:
        plt.savefig(root + 'visualization/Fig8E_fc1', dpi=300)
        plt.savefig(root + 'visualization/Fig8E_fc1.svg')
    # plt.show()

if __name__ == "__main__":
    main()
