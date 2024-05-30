import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import math
import sys
import scipy

import scipy.stats

# fontsizes 
global fontsize_title
global fontsize_legend
global fontsize_label
global fontsize_tick

fontsize_title          = 15
fontsize_legend         = 10
fontsize_label          = 12
fontsize_tick           = 10

##### SET directories
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'
root_data           = '/home/amber/OneDrive/data/EEG/'

# initiate figure
fig, axs = plt.subplots(1, 2, figsize=(10, 3))

# settings for plotting
sns.despine(offset=10)

# foloder
folder = 'dataCollection2024/pilot4'

# subjects
exclude = []

# retrieve files
files = sorted(glob.glob(root_data + 'behaviour/' + folder +'/*.txt'))
print(root_data + 'behaviour/' + folder)

# info experimental settings
adapters            = ['blank', 'same', 'different']
adapters_col        = ['gray', 'dodgerblue', np.array([212, 170, 0])/255]
contrasts           = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
contrasts_values    = [50, 60, 70, 80, 90]

target              = [3, 6, 8, 9]
target_n            = len(target)

# initiate dataframe to store data
accu_clean          = np.zeros(len(files)-len(exclude))
accu                = np.zeros((len(files)-len(exclude), len(contrasts), len(adapters)))

rTime_clean         = np.zeros(len(files)-len(exclude))
rTime               = np.zeros((len(files)-len(exclude), len(contrasts), len(adapters)))
print(rTime.shape)

# initiate counts
subject_n = len(files)
print(subject_n)

# extract data
count = 0
for i, file in enumerate(files):

    # import datafile
    df = pd.read_csv(file)

    # extract performance of clean trials
    select = df[(df.trial_type == 'clean')]
    n_trials = len(select)

    # count number of correct trials
    correct = select[select.response == 1]
    n_trials_correct = len(correct)
    accu_clean[count] = (n_trials_correct/n_trials)*100

    # compute average reaction time
    correct = select[select.response == 1]
    rTime_clean[count] = np.mean(correct.loc[:, 'RT'])

    # extract performance of noisy trials
    for iA in range(len(adapters)):
        for iC in range(len(contrasts)):

            if adapters[iA] == 'blank':

                # select trials
                select = df[(df.trial_type == 'single') & (df.adapter == 'none') & (df.contrast == contrasts[iC])]
                n_trials = len(select)
            
            else:

                # select trials
                select = df[(df.trial_type == 'repeated') & (df.adapter == adapters[iA]) & (df.contrast == contrasts[iC])]
                n_trials = len(select)

            # count number of correct trials
            correct = select[select.response == 1]
            n_trials_correct = len(correct)
            accu[count, iC, iA] = (n_trials_correct/n_trials)*100

            # compute average reaction time
            correct = select[select.response == 1]
            rTime[count, iC, iA] = np.mean(correct.loc[:, 'RT'])

    # increment count
    count+=1

# offset adapter wrt x-axis
offset_iC = [0, 3, 6, 9, 12]
offset = [-0.5, 0, 0.5]

# create strings for subject
subject_str = []
for i in range(count):
    subject_str.append('sub' + str(i+1))

# plot 
for i in range(2): # accuracy (i == 0) and reaction time (i == 1)

    # initiate dataframe
    df_stats = pd.DataFrame()
    df_stats['subject'] = np.tile(subject_str, len(adapters)*len(contrasts))
    df_stats['contrast'] = np.repeat(contrasts, len(adapters)*(len(files)-len(exclude)))
    df_stats['adapter'] = np.tile(np.repeat(adapters, len(files)-len(exclude)), len(contrasts_values))
    df_stats['dependentVar'] = 0
    print(df_stats)

    # adjust axes
    if i == 0:
        axs[i].set_ylabel('Performance (%)', fontsize=fontsize_label)
    else:
        axs[i].set_ylabel('Reaction time (s)', fontsize=fontsize_label)
    
    axs[i].tick_params(axis='both', labelsize=fontsize_tick)
    axs[i].set_xlabel('Contrast target image (%)', fontsize=fontsize_label)
    axs[i].set_xticks(offset_iC)
    axs[i].set_xticklabels(contrasts_values, fontsize=fontsize_label)
    axs[i].spines['top'].set_visible(False)
    axs[i].spines['right'].set_visible(False)
    axs[i].set_xlim(offset_iC[0]-1, offset_iC[-1]+1)
    if i == 0:
        axs[i].set_ylim(-5, 105)
        axs[i].axhline(100/target_n, linestyle=(0, (5, 10)), zorder=-10, color='black', linewidth=1)

    # plot accuracy for clean trials
    if i == 0:
        data_mean = np.mean(accu_clean)
        data_std = np.std(accu_clean)/math.sqrt(accu.shape[0])
    else:
        data_mean = np.mean(rTime_clean[:])
        data_std = np.std(rTime_clean[:])/math.sqrt(accu.shape[0])    

    # plot
    axs[i].fill_between(np.arange(offset[0], offset[-1]), data_mean - data_std, data_mean + data_std, color='lightgrey', zorder=-1, alpha=0.5)
    axs[i].axhline(data_mean, color='lightgrey', zorder=1, lw=1.5)
    print('Mean: ', data_mean)
    print('SD: ', data_std)

    # statistical testing
    for iC, contrast in enumerate(contrasts):
        for iA, adapter in enumerate(adapters):

            if i == 0:
                idx = df_stats[(df_stats.contrast == contrast) & (df_stats.adapter == adapter)].index
                df_stats.loc[idx, 'dependentVar'] = accu[:, iC, iA]

            elif i == 1:
                idx = df_stats[(df_stats.contrast == contrast) & (df_stats.adapter == adapter)].index
                df_stats.loc[idx, 'dependentVar'] = rTime[:, iC, iA]
    
    # plot accuracies
    for iC, contrast in enumerate(contrasts):
        for iA, adapter in enumerate(adapters):

            # extrac data
            idx = df_stats[(df_stats.adapter == adapter) & (df_stats.contrast == contrast)].index
            data_current = np.array(df_stats.loc[idx, 'dependentVar'])

            # compute average and spread
            data_mean = np.nanmean(data_current)
            data_std = np.nanstd(data_current)/math.sqrt(subject_n)

            # plot
            axs[i].plot([offset_iC[iC]+offset[iA], offset_iC[iC]+offset[iA]], [data_mean - data_std, data_mean + data_std], color='grey', zorder=-1, lw=0.5)
            axs[i].scatter(offset_iC[iC]+offset[iA], data_mean, color=adapters_col[iA], edgecolor='white', s=50, zorder=1)

            # plot per subject
            if i == 0:
                sns.stripplot(x=np.ones(accu.shape[0])*(offset_iC[iC]+offset[iA]), y=accu[:, iC, iA], jitter=True, ax=axs[i], color=adapters_col[iA], size=3, alpha=0.3, native_scale=True)
            else:
                sns.stripplot(x=np.ones(accu.shape[0])*(offset_iC[iC]+offset[iA]), y=rTime[:, iC, iA], jitter=True, ax=axs[i], color=adapters_col[iA], size=3, alpha=0.3, native_scale=True)

    # retrieve data
    slope           = np.zeros((len(adapter), subject_n))
    intercept       = np.zeros((len(adapter), subject_n))
    for iA, adapter in enumerate(adapters):

        # select data
        if i == 0:
            data_current = accu[:, :, iA]
        else:
            data_current = rTime[:, :, iA]

        # fit regression
        pred = np.zeros((subject_n, len(contrasts)))
        for iS in range(subject_n):
            model = scipy.stats.linregress(np.arange(len(contrasts)), data_current[iS, :])
            slope[iA, iS] = model.slope
            intercept[iA, iS] = model.intercept
            pred[iS, :] = model.intercept + model.slope * np.arange(len(contrasts))
        axs[i].plot(np.array(offset_iC)+offset[iA], np.mean(pred, 0), color=adapters_col[iA], lw=0.5, alpha=0.5, zorder=-10)

    # save dataframe
    if i == 0:
        df_stats.to_csv(root+'data/behaviour/behaviour_performance.csv', sep=',', index=0)
    elif i == 1:
        df_stats.to_csv(root+'data/behaviour/behaviour_reactiontime.csv', sep=',', index=0)

# save
plt.tight_layout()
plt.savefig(root + 'visualization/Fig3')
plt.savefig(root + 'visualization/Fig3.svg')
# plt.show()
