# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import json
import sys
from preprocessing_epochSelection_utils import *

# set root directory
dir        = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

# set timecourse
tmin                    = -0.1
tmax                    = 0.5
sample_rate             = 2048
down_sample_rate        = 256

n_timepoints            = 154 #  (abs(tmin) + tmax)/(1/sample_rate)
t                       = np.arange(n_timepoints)*(1/down_sample_rate)+tmin

# channels
channelNames = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']
n_channel = len(channelNames)

# import data
data_clean          = np.load(dir + 'data/EEG/data_clean.npy')
data_single         = np.load(dir + 'data/EEG/data_single.npy')
data_repeated       = np.load(dir + 'data/EEG/data_repeated.npy')

print('Shape data struct for clean trials: '.ljust(50), data_clean.shape)
print('Shape data struct for single trials: '.ljust(50), data_single.shape)
print('Shape data struct for repeated trials: '.ljust(50), data_repeated.shape)

n_sub = data_clean.shape[0]

# flatten
data_clean_flatten = data_clean.reshape(data_clean.shape[0], int(data_clean.shape[1]*data_clean.shape[2]*data_clean.shape[3]), data_clean.shape[4], data_clean.shape[5])
data_single_flatten = data_single.reshape(data_single.shape[0], int(data_single.shape[1]*data_single.shape[2]*data_single.shape[3]), data_single.shape[4], data_single.shape[5])
data_repeated_flatten = data_repeated.reshape(data_repeated.shape[0], int(data_repeated.shape[1]*data_repeated.shape[2]*data_repeated.shape[3]), data_repeated.shape[4], data_repeated.shape[5])

print('Shape data struct for clean trials: '.ljust(50), data_clean_flatten.shape)
print('Shape data struct for single trials: '.ljust(50), data_single_flatten.shape)
print('Shape data struct for repeated trials: '.ljust(50), data_repeated_flatten.shape)

data_flatten                            = np.zeros((21, 880, 64, 154))
data_flatten[:, :400, :, :]             = data_clean_flatten
data_flatten[:, 400:560, :, :]          = data_single_flatten
data_flatten[:, 560:, :, :]             = data_repeated_flatten
print('Shape all: ', data_flatten.shape)

data_flatten_epoch = np.zeros_like(data_flatten)

# store rejection percentages
rejection_percentages = np.zeros((n_sub, n_channel))

# select epochs
for iS in range(n_sub):
# for iS in range(2):

    # create directory
    try:
        os.mkdir(dir + 'visualization/preprocessing_epochSelection/sub' + str(iS+1))
    except:
        print('directory already exists')
    
    # select epcohs
    data_temp, rejection = select_epochs(iS, channelNames, t, data_flatten[iS, :, :, :], dir)
    data_flatten_epoch[iS, :, :, :]     = data_temp
    rejection_percentages[iS, :]        = rejection

# reshape to original
data_clean_epoch            = data_flatten[:, :400, :, :].reshape(data_clean.shape)
data_single_epoch           = data_flatten[:, 400:560, :, :].reshape(data_single.shape)
data_repeated_epoch         = data_flatten[:, 560:, :, :].reshape(data_repeated.shape)

# save dataframes
np.save(dir + 'data/EEG/data_clean_epochSelection', data_clean_epoch)
np.save(dir + 'data/EEG/data_single_epochSelection', data_single_epoch)
np.save(dir + 'data/EEG/data_repeated_epochSelection', data_repeated_epoch)

# save rejection criteria
np.save(dir + 'data/EEG/rejecionPercentages', rejection_percentages)

# save rejection criteria
rejection_percentages               = np.load(dir + 'data/EEG/rejecionPercentages.npy')
rejection_percentages_mean          = rejection_percentages.mean(1)
print(rejection_percentages_mean.shape)
print(rejection_percentages[2, :])
print('avg: ', np.mean(rejection_percentages_mean))
print('min: ', np.min(rejection_percentages_mean))
print('max: ', np.max(rejection_percentages_mean))

