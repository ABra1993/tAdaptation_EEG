import os
import numpy as np
import matplotlib.pyplot as plt
import json
import pandas as pd

def select_epochs(current_subject, channelNames, t, data, dir):

    # initiate array
    data_epoch = np.zeros_like(data)

    rejection = np.zeros(len(channelNames))

    # number of standard deviations (threshold)
    nstd = 2

    # timecourse to display
    t_start     = np.argwhere(t > 0)[0][0] # tmin = -0.2
    
    for iCh, ch in enumerate(channelNames):

        print('Current electrode: ', ch + ' ( subject', current_subject+1, ')')

        # initiate figure
        _, axs = plt.subplots(2, 2, figsize=(12, 7))
        axs[0, 1].axis('off')

        # current data
        data_current = data[:, iCh, :]

        # compute max values
        max_values = np.max(np.abs(data_current[:, t_start:]), 1)
        max_values_mean = np.nanmean(max_values)
        max_values_std = np.nanstd(max_values)
        print('Mean of max. values: ', max_values_mean)
        print('Standard deviation of max. values: ', max_values_std)

        ################################################
        ######################### BEFORE EPOCH SELECTION
        ################################################

        axs[0, 0].set_title('Max. values computed per epoch')
        axs[0, 0].hist(max_values, color='grey', bins=100)
        axs[0, 0].axvline(max_values_mean, color='mediumblue', label='mean')
        axs[0, 0].axvline(max_values_mean + nstd * max_values_std, color='dodgerblue', label=str(nstd) + ' x std')
        axs[0, 0].set_ylabel('Count')
        axs[0, 0].set_xlabel(r'Maximum value ($\mu$V)')
        axs[0, 0].legend(fontsize=8, frameon=False)

        # plot all trials (before epochs election)
        axs[1, 0].set_title('Before epoch selection')
        axs[1, 0].axvline(0, color='grey', linestyle='dotted')
        axs[1, 0].set_ylabel(r'$\mu$V')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].plot(t, data_current.T, lw=0.2)

        ################################################
        ################################################
        
        # remove epochs
        index = np.argwhere(max_values > max_values_mean + nstd * max_values_std)
        data_current[index.squeeze().tolist()] = np.NaN

        # plot all trials (before epochs election)
        excludedEpoch   =  (len(index) + len(index))/len(max_values)
        rejection[iCh]  = np.round(excludedEpoch*100, 2)

        # ################################################
        # ########################## AFTER EPOCH SELECTION
        # ################################################
        
        axs[1, 1].set_title('After epoch selection \n(' + str(np.round(excludedEpoch*100, 2)) + '%, ' + str(len(index)) + ' out of ' + str(len(max_values)) + ' epochs)')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].axvline(0, color='grey', linestyle='dotted')
        axs[1, 1].plot(t, data_current.T, lw=0.2)

        # save data
        data_epoch[:, iCh, :] = data_current

        # save figure
        plt.tight_layout()
        plt.savefig(dir + 'visualization/preprocessing_epochSelection/sub' + str(current_subject+1) + '/' + ch)
        plt.close()

    return data_epoch, rejection