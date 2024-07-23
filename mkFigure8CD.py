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

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

def naka_rushton(C, c50, q, **args):

    y = ((args['A'])*C**q)/(c50**q + C**q)
    
    return y

def cross_validate_svm(activations, labels, cv):
    scaler = StandardScaler()
    clf = SVC(kernel='linear')
    pipeline = make_pipeline(scaler, clf)
    scores = cross_val_score(pipeline, activations, labels, cv=cv)
    return scores

# define root
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

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

    # number of layers
    layers      = ['conv1', 'conv2', 'conv3', 'fc1']
    n_layer     = len(layers)

    # set temporal dynamics
    # tempDynamics       = ['none', 'add_supp', 'div_norm', 'lat_recurrence',  'lat_recurrence_mult', 'add_supp_vGain', 'div_norm_vGain']
    # tempDynamics_lbls   = ['FF', 'AS', 'DN', r'LR$_{A}$', r'LR$_{M}$', r'AS$_{vGain}$', r'DN$_{vGain}$']

    tempDynamics        = ['add_supp_vGain', 'div_norm_vGain']
    params              = [['alpha', 'beta'], ['alpha']]

    # tempDynamics        = ['div_norm_vGain']
    # params              = [['alpha']]
    
    # noise
    adapters           = ['none', 'same', 'different']

    # retrieve timesteps
    t_steps_label = encode_timesteps(t_steps, start=start, dur=dur)
    print(t_steps_label)

    # other settings
    init        = 10
    batch_size  = 600

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # define contrast values
    contrast_values      = torch.Tensor([0.5, 0.6, 0.7, 0.8, 0.9])

    # decoding accuracy
    cv = 3  # cross-validations

    # values CRF
    q           = torch.Tensor([14])
    c50         = torch.linspace(0.5, 0.9, 5)

    # retrieve accuracies or load
    preload = True # if False the accuracies will be shown from the folder /models/performance_variableContrast_*, otherwise accuracies wil be re-computed

    if preload == False:

        # set home directory
        root_data           = root + 'models/dataset/'
        if dataset == 'mnist':
            testData        = datasets.MNIST(root=root_data, download=True, train=False, transform=transforms.ToTensor())
        elif dataset == 'fmnist':
            testData        = datasets.FashionMNIST(root=root_data, download=True, train=False, transform=transforms.ToTensor())

        # data loaders
        ldrTest             = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=True)

        # perform cross-validation
        for a, (imgs, lbls) in enumerate(ldrTest):

            # only perform decoding over first batch
            if a != 0:
                break

            for iT, current_tempDynamics in enumerate(tempDynamics):

                for _, param in enumerate(params[iT]):

                    # store decoding accuracies
                    dec_acc = np.zeros((len(c50), len(contrast_values), len(adapters), len(layers), init))

                    # print progress
                    print('Current temp. dynamics: ', current_tempDynamics, ', param: ', param)

                    for iInit in range(init):

                        print('Initialization: ', iInit+1)

                        # load model
                        if current_tempDynamics == 'div_norm_vGain':
                            model = cnn_feedforward(t_steps, 'div_norm')
                        elif current_tempDynamics == 'add_supp_vGain':
                            model = cnn_feedforward(t_steps, 'add_supp')
                        else:
                            model = cnn_feedforward(t_steps, current_tempDynamics)

                        # initialize temporal dynamics
                        model.initialize_tempDynamics()

                        # load weights
                        if current_tempDynamics == 'div_norm_vGain':
                            model.load_state_dict(torch.load(root + 'models/weights/div_norm_' + str(iInit+1)))
                        elif current_tempDynamics == 'add_supp_vGain':
                            model.load_state_dict(torch.load(root + 'models/weights/add_supp_' + str(iInit+1)))
                        else:
                            model.load_state_dict(torch.load(root + 'models/weights/' + current_tempDynamics + '_' + str(iInit+1)))
                        model.to(device)
                        model.eval()

                        for ic50, c50_current in enumerate(c50):

                            # print progress
                            print('c50: ', c50_current)

                            # initiate gain
                            if current_tempDynamics == 'div_norm_vGain':
                                if param == 'alpha':
                                    value = naka_rushton(contrast_values, c50_current, q[0], A=model.sconv1.alpha.detach().cpu())
                                elif param == 'sigma':
                                    value = naka_rushton(contrast_values, c50_current, q[0], A=model.sconv1.sigma.detach().cpu())
                                elif param == 'K':
                                    value = naka_rushton(contrast_values, c50_current, q[0], A=model.sconv1.K.detach().cpu())

                            elif current_tempDynamics == 'add_supp_vGain':
                                if param == 'alpha':
                                    value = naka_rushton(contrast_values, c50_current, q[0], A=model.sconv1.alpha.detach().cpu())
                                elif param == 'beta':
                                    value = naka_rushton(contrast_values, c50_current, q[0], A=model.sconv1.beta.detach().cpu())

                            # iterate over contrast values and obtain decoding accuracy
                            for iC, contrast_value in enumerate(contrast_values):

                                if current_tempDynamics == 'none':

                                    # create input sequence
                                    ax = sequence_train(F.adjust_contrast(imgs, contrast_value), t_steps, t_steps_label, False)

                                    # validate
                                    _ = model.forward(ax[:int(batch_size/2), :, :, :, :].to(device))
                                    actvs = model.actvs

                                    # select data
                                    for iL in range(n_layer):
                                        X = actvs[iL][2]
                                        if iL != 3:
                                            X = actvs[iL][2]
                                            X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
                                        y = lbls[:int(batch_size/2)]

                                        # perform decoding analysis np.zeros((len(tempDynamics), len(contrast_values), len(adapters), len(layers), init))
                                        temp = cross_validate_svm(X.detach().cpu().numpy(), y.detach().cpu().numpy(), cv)
                                        dec_acc[ic50, iC, 0, iL, iInit] = temp.mean()
                                        
                                else:

                                    # change variable of gain
                                    if current_tempDynamics == 'div_norm_vGain':
                                        if param == 'alpha':
                                            model.sconv1.alpha = nn.Parameter(torch.Tensor(value[iC]))
                                        elif param == 'sigma':
                                            model.sconv1.sigma = nn.Parameter(torch.Tensor(value[iC]))
                                        elif param == 'K':
                                            model.sconv1.K = nn.Parameter(torch.Tensor(value[iC]))

                                    elif current_tempDynamics == 'add_supp_vGain':
                                        if param == 'alpha':
                                            model.sconv1.alpha = nn.Parameter(torch.Tensor(value[iC]))
                                        elif param == 'beta':
                                            model.sconv1.beta = nn.Parameter(torch.Tensor(value[iC]))
                                    
                                    # create input sequence
                                    ax = sequence_train(F.adjust_contrast(imgs, contrast_value), t_steps, t_steps_label, False)

                                    # validate
                                    _ = model.forward(ax.to(device))
                                    actvs = model.actvs

                                    for iA, _ in enumerate(adapters[1:]):
                                        for iL in range(n_layer):

                                            # select data
                                            if iA == 0:
                                                if iL == 3:
                                                    X = actvs[iL][2][:int(batch_size/2), :]
                                                else:
                                                    X = actvs[iL][2][:int(batch_size/2), :, :, :]
                                                    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
                                                y = lbls[:int(batch_size/2)]
                                            else:
                                                if iL == 3:
                                                    X = actvs[iL][2][int(batch_size/2):, :]
                                                else:
                                                    X = actvs[iL][2][int(batch_size/2):, :, :, :]
                                                    X = X.reshape(X.shape[0], X.shape[1]*X.shape[2]*X.shape[3])
                                                y = lbls[int(batch_size/2):]

                                            # perform decoding analysis np.zeros((len(tempDynamics), len(contrast_values), len(adapters), len(layers), init))
                                            temp = cross_validate_svm(X.detach().cpu().numpy(), y.detach().cpu().numpy(), cv)
                                            dec_acc[ic50, iC, iA+1, iL, iInit] = temp.mean()

                    # save accuracies
                    np.save(root + 'models/decoding/accuracies_' + current_tempDynamics + '_' + param , dec_acc)

    # colors
    cmap                = plt.cm.get_cmap('Blues')
    color_same          = cmap(np.linspace(0.3, 1, len(c50)))

    cmap                = plt.cm.get_cmap('YlOrBr')
    color_diff          = cmap(np.linspace(0.3, 1, len(c50)))

    adapters_color_beh  = [color_same, color_diff]

    # # visualize
    # for iT, current_tempDynamics in enumerate(tempDynamics):

    #     # initiate figure
    #     fig, axs = plt.subplots(len(params[iT]), n_layer, figsize=(6, 3.5), sharey=True)
    #     sns.despine(offset=5)

    #     for iL in range(n_layer):
    #         for iP, param in enumerate(params[iT]):

    #             # load data
    #             dec_acc = np.load(root + 'models/decoding/accuracies_' + current_tempDynamics + '_' + param + '.npy')
                
    #             for iA, adapter in enumerate(adapters[1:]):
    #                 for ic50, c50_current in enumerate(c50):

    #                     # select data
    #                     current_data = dec_acc[ic50, :, iA+1, iL, :]

    #                     # compute averages
    #                     data_mean = np.mean(current_data, 1)
    #                     data_sem = np.std(current_data,  1)/math.sqrt(init)

    #                     # plot
    #                     if len(params[iT]) == 1:
    #                         axs[iL].plot(np.arange(len(contrast_values)), data_mean, color=adapters_color_beh[iA][ic50])
    #                         axs[iL].fill_between(np.arange(len(contrast_values)), data_mean - data_sem, data_mean + data_sem, color=adapters_color_beh[iA][ic50], alpha=0.5)
    #                     else:
    #                         axs[iP, iL].plot(np.arange(len(contrast_values)), data_mean, color=adapters_color_beh[iA][ic50])
    #                         axs[iP, iL].fill_between(np.arange(len(contrast_values)), data_mean - data_sem, data_mean + data_sem, color=adapters_color_beh[iA][ic50], alpha=0.5)

    #                     # adjust axes
    #                     # if iL == 0:
    #                     #     if len(params[iT]) == 1:
    #                     #         axs[iL].set_ylabel('Decoding accuracy', fontsize=fontsize_label)
    #                     #     else:
    #                     #         axs[iP, iL].set_ylabel('Decoding accuracy', fontsize=fontsize_label)

    #             # adjust axes
    #             if len(params[iT]) == 1:
    #                 axs[iL].set_xticks(np.arange(len(contrast_values)))
    #                 axs[iL].set_xticklabels(['50', '60', '70', '80', '90'], fontsize=fontsize_label)
    #                 # axs[iL].set_title(layers[iL], fontsize=fontsize_title)
    #             else:
    #                 axs[iP, iL].set_xticks(np.arange(len(contrast_values)))
    #                 if iP == len(tempDynamics) - 1:
    #                     axs[iP, iL].set_xticklabels(['50', '60', '70', '80', '90'], fontsize=fontsize_label)
    #                 else:
    #                     axs[iP, iL].set_xticklabels(['', '', '', '', ''], fontsize=fontsize_label)
    #                 if iP == 0:
    #                     axs[iP, iL].set_title(layers[iL], fontsize=fontsize_title)

    #     # save plot
    #     fig.align_ylabels()
    #     plt.tight_layout()
    #     plt.savefig(root + 'visualization/Fig8C_' + current_tempDynamics, dpi=300)
    #     plt.savefig(root + 'visualization/Fig8C_' + current_tempDynamics + '.svg')
    #     # plt.close()

    # colors
    cmap                = plt.cm.get_cmap('Blues')
    color_same          = cmap(np.linspace(0.3, 1, 4))

    cmap                = plt.cm.get_cmap('YlOrBr')
    color_diff          = cmap(np.linspace(0.3, 1, 4))

    adapters_color_beh  = [color_same[1], color_diff[1]]

    # indices
    layer_idx   = 3
    ic50_idx    = [[3, 3], [4]]
    # ic50_idx    = [[4]]

    # mstyle
    markersize          = 7
    markersize_small    = 5
    lw                  = 1

    # visualize
    offset_iC = np.array([0, 2.5, 5, 7.5, 10])
    offset_iA = [-0.1, 0.1]
    for iT, current_tempDynamics in enumerate(tempDynamics):

        print(current_tempDynamics)

        # initiate figure
        if len(params[iT]) == 1:
            fig = plt.figure(figsize=(3, 2))
            axs = plt.gca()
        else:
            fig, axs = plt.subplots(1, len(params[iT]), figsize=(3*len(params[iT]), 2))

        sns.despine(offset=5)

        for iP, param in enumerate(params[iT]):
            for iA, adapter in enumerate(adapters[1:]):

                # load data
                dec_acc = np.load(root + 'models/decoding/accuracies_' + current_tempDynamics + '_' + param + '.npy')

                # select data
                current_data = dec_acc[ic50_idx[iT][iP], :, iA+1, layer_idx, :]

                # compute averages
                data_mean = np.mean(current_data, 1)
                data_sem = np.std(current_data, 1)/math.sqrt(init)

                # plot
                if len(params[iT]) == 1:
                    axs.plot(offset_iC+offset_iA[iA], data_mean, color=adapters_color_beh[iA], label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA], markeredgecolor='white')
                else:
                    axs[iP].plot(offset_iC+offset_iA[iA], data_mean, color=adapters_color_beh[iA], label=adapter, marker='o', markersize=markersize, lw=lw, markerfacecolor=adapters_color_beh[iA], markeredgecolor='white')

                for iC, contrast in enumerate(contrast_values):
                
                    # select data
                    current_data = dec_acc[ic50_idx[iT][iP], iC, iA+1, layer_idx, :]

                    # compute averages
                    data_mean = np.mean(current_data)
                    data_sem = np.std(current_data)/math.sqrt(init)

                    # plot
                    if len(params[iT]) == 1:
                        axs.fill_between([offset_iC[iC]+offset_iA[iA], offset_iC[iC]+offset_iA[iA]], data_mean - data_sem, data_mean + data_sem, color='black')
                        sns.stripplot(x=np.ones(init)*offset_iC[iC]+offset_iA[iA], y=dec_acc[ic50_idx[iT][iP], iC, iA+1, layer_idx, :], jitter=True, ax=axs, color=adapters_color_beh[iA], size=markersize_small, alpha=0.5, native_scale=True, legend=False, zorder=-10)
                    else:
                        axs[iP].fill_between([offset_iC[iC]+offset_iA[iA], offset_iC[iC]+offset_iA[iA]], data_mean - data_sem, data_mean + data_sem, color='black')
                        sns.stripplot(x=np.ones(init)*offset_iC[iC]+offset_iA[iA], y=dec_acc[ic50_idx[iT][iP], iC, iA+1, layer_idx, :], jitter=True, ax=axs[iP], color=adapters_color_beh[iA], size=markersize_small, alpha=0.5, native_scale=True, legend=False, zorder=-10)

            for iC, contrast in enumerate(contrast_values):

                ############################################################### STATS
                sample1 = dec_acc[ic50_idx[iT][iP], iC, 1, layer_idx, :]
                sample2 = dec_acc[ic50_idx[iT][iP], iC, 2, layer_idx, :]

                # ttest
                result = stats.ttest_ind(sample1, sample2)[1]
                print(result)
                if result < (0.05/len(contrast_values)):
                    print(current_tempDynamics, ' , contrast: ', contrast, ': ', np.round(result.item(), 3))

            if len(params[iT]) == 1:
                axs.tick_params(axis='both', labelsize=fontsize_tick)
                axs.spines['top'].set_visible(False)
                axs.spines['right'].set_visible(False)
                axs.set_xticks(offset_iC)
                axs.set_xticklabels(['50', '60', '70', '80', '90'], fontsize=fontsize_label)
                axs.set_ylim(0.5, 1)
            else:
                axs[iP].tick_params(axis='both', labelsize=fontsize_tick)
                axs[iP].spines['top'].set_visible(False)
                axs[iP].spines['right'].set_visible(False)
                axs[iP].set_xticks(offset_iC)
                axs[iP].set_xticklabels(['50', '60', '70', '80', '90'], fontsize=fontsize_label)
                axs[iP].set_ylim(0.5, 1)

        # save plot
        fig.align_ylabels()
        plt.tight_layout()
        plt.savefig(root + 'visualization/Fig8C_' + current_tempDynamics, dpi=300)
        plt.savefig(root + 'visualization/Fig8C_' + current_tempDynamics + '.svg')
        # plt.close()

if __name__ == '__main__':
    main()

