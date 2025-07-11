# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression
import mne
import matplotlib as mpl
from statsmodels.stats.multitest import multipletests  
from scipy.ndimage import gaussian_filter1d
from scipy import stats

# bootstrapping
global subNum
global CI_low
global CI_high
global B_repetitions

# determine confidence interval for plotting
subNum              = 27
CI                  = 68
CI_low              = 50 - (0.5*CI)
CI_high             = 50 + (0.5*CI)
B_repetitions       = 1000

# labels
global label_img
global label_img_long
global label_diff
global label_ISI

label_img           = ['A', 'T']
label_img_long      = ['Adapter', 'Test']
label_diff          = r'$\Delta$ amplitude A-T ($\mu$V)'
label_ISI           =  ['67', '134', '267', '533']

# miscalleneous
global s            # markersize
global alpha        # transparacy SE

s                   = 100
alpha               = 0.15

# fontsizes 
global fontsize_title
global fontsize_legend
global fontsize_label
global fontsize_tick

fontsize_title          = 15
fontsize_legend         = 10
fontsize_label          = 12
fontsize_tick           = 10

def visualizeAdapter_topomap(data, montage, info, dict, adapters, fig_name, dir):

    # create evoked responses
    times = np.array([0.125, 0.3])
    adapters_plot = [0, 2, 4]

    # initiate figure
    fig, axs = plt.subplots(len(times), len(adapters_plot), figsize=(9, 4))

    # min and maximal voltages
    vmin = -4000000
    vmax = 4000000

    print(data[0].shape)
    print(data[1].shape)

    # average dataframes
    data_concat = np.zeros((21, 5, 3, 32, 64, 154))
    data_concat[:, :, :1, :, :, :]         = data[0]
    data_concat[:, :, 1:, :, :, :]         = data[1]
    data_concat = np.nanmean(data_concat, 0)

    for iA in range(len(adapters)):

        # select data
        data_current = data_concat[:, iA, :, :, :]
        
        # print(data_current.shape)
        data_current = data_current.reshape(data_current.shape[0]*data_current.shape[1], data_current.shape[2], data_current.shape[3]).mean(0)
        # data_current = data_current/np.max(data_current)

        # initatie mne
        evoked = mne.EvokedArray(
            data_current, info, tmin=-0.1, nave=data_current.shape[0])

        # set info
        evoked.set_channel_types(dict)
        evoked.set_montage(montage=montage)

        # plot topomap
        evoked.plot_topomap(times, ch_type='eeg',  average=0.05, vlim=(vmin, vmax), contours=0, axes=axs[:, iA], cmap='BrBG', colorbar=False, sensors='k.')
        #  time_format='%0.2f'

    # save figure
    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name, dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '.svg')

    # create colorbar
    fig, ax = plt.subplots(figsize=(1, 3))
    fig.subplots_adjust(right=0.4)

    cmap = mpl.cm.BrBG
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(r'$\mu$V', fontsize=fontsize_label)
    cb1.outline.set_visible(False)
    cb1.ax.tick_params(labelsize=fontsize_tick)

    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name + '_colorbar', dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '_colorbar.svg')

def visualizeAdapter_topomap_diff(data, montage, info, dict, adapters, fig_name, dir):

    # create evoked responses
    times = np.array([0.125, 0.3])

    # initiate figure
    fig, axs = plt.subplots(len(times), len(adapters)-1, figsize=(6, 4))

    # time points
    vmin = -1500000
    vmax = 1500000

    print(data[0].shape)
    print(data[1].shape)

    # compute single trials
    data_none = data[0][:, :, 0, :, :, :]

    for iA in range(len(adapters)-1):

        # select data
        data_current = data[1][:, :, iA, :, :, :] - data_none
        
        # print(data_current.shape)
        data_current = np.nanmean(data_current.reshape(data_current.shape[0]*data_current.shape[1]*data_current.shape[2], data_current.shape[3], data_current.shape[4]), 0)

        # initatie mne
        evoked = mne.EvokedArray(
            data_current, info, tmin=-0.1, nave=data_current.shape[0])

        # set info
        evoked.set_channel_types(dict)
        evoked.set_montage(montage=montage)

        # plot topomap
        evoked.plot_topomap(times, ch_type='eeg', time_format='%0.2f', average=0.05, vlim=(vmin, vmax), contours=0, axes=axs[:, iA], cmap='Spectral', colorbar=False, sensors='k.')

    # save figure
    plt.savefig(dir + 'visualization/' + fig_name, dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '.svg')

    # create colorbar
    fig, ax = plt.subplots(figsize=(1, 3))
    fig.subplots_adjust(right=0.4)

    cmap = mpl.cm.Spectral
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
    cb1.set_label(r'$\Delta\mu$V', fontsize=fontsize_label)
    cb1.outline.set_visible(False)
    cb1.ax.tick_params(labelsize=fontsize_tick)

    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name + '_colorbar', dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '_colorbar.svg')

def visualizeAdapter(data, t, n_sub, channelNames_current, electrode_idx, adapters, fig_name, dir):

    # initiate figure
    _, axs = plt.subplots(1, 3, figsize=(12, 3))

    # concatenate data
    data_concat = np.zeros((21, 5, 3, 32, 64, 154))
    data_concat[:, :, :1, :, :, :]         = data[0]
    data_concat[:, :, 1:, :, :, :]         = data[1]
    # print(data_concat.shape)

    # visualize different contrast levels for repeated trials
    color = ['gray', 'dodgerblue', np.array([212, 170, 0])/255]

    # visualize absolute values
    for iA, adapter in enumerate(adapters):

        # select data
        data_current = data_concat[:, :, iA, :, electrode_idx, :]
        data_current = np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1)

        # reshape and average
        data_mean = np.nanmean(data_current, 0)
        data_std = np.nanstd(data_current, 0)/math.sqrt(n_sub)

        # plot
        axs[0].plot(t, data_mean, label=adapter, color=color[iA], lw=2)
        axs[0].fill_between(t, data_mean - data_std, data_mean + data_std, alpha=0.2, color=color[iA])

    # compute data
    data_current_temp = np.nanmean(data[1], 2).squeeze() - data[0].squeeze()
    data_current_temp = data_current_temp[:, :, :, electrode_idx, :]
    data_current_temp = np.nanmean(np.nanmean(np.nanmean(data_current_temp, 1), 1), 1)

    # compute timepoints where responses are significant
    p_values = np.zeros(len(t))
    for tmp in range(len(t)):

        # parametric
        p_values[tmp] = stats.ttest_1samp(data_current_temp[:, tmp], 0)[1]

    # MC correction
    # _, p_values, _, _ = multipletests(p_values, method='fdr_bh')
    pvals_corrected_sign = np.argwhere(p_values < 0.05).flatten()

    # define time windows
    idx_time_windows = []
    start_idx = None

    for i in range(len(pvals_corrected_sign) - 1):
        if pvals_corrected_sign[i + 1] - pvals_corrected_sign[i] == 1:
            if start_idx is None:
                start_idx = pvals_corrected_sign[i]
        else:
            if start_idx is not None:
                idx_time_windows.append([start_idx, pvals_corrected_sign[i]])
                start_idx = None

    # Check if there are consecutive numbers at the end of the array
    if start_idx is not None:
        idx_time_windows.append([start_idx, pvals_corrected_sign[-1]])

    # compute response magnitude differences
    time_windows    = len(idx_time_windows)
    timepoints      = np.zeros((time_windows,  2))

    # plot time ranges
    count = 0
    for _, idx in enumerate(idx_time_windows):

        if (t[idx[0]] < 0):
            continue
        else:

            # retrieve timepoints
            start = t[idx[0]]
            end = t[idx[1]]
            timepoints[count, 0] = start
            timepoints[count, 1] = end

            # visualize
            axs[1].axvspan(start, end, facecolor='lightsalmon', alpha=0.3, edgecolor='white')
            # axs[1].axvline(start, color='grey', linestyle='dotted')
            # axs[1].axvline(end, color='grey', linestyle='dotted')

        # increment count
        count = count + 1

    # visualize relative values
    data_none = data_concat[:, :, 0, :, electrode_idx, :]

    # save response magnitudes
    AUC             = np.zeros((len(adapters)-1, time_windows, n_sub))

    # save x-ticks
    offset = [-0.1, 0.1]
    for iA, adapter in enumerate(adapters[1:]):

        # select data
        data_current = data_concat[:, :, iA+1, :, electrode_idx, :] - data_none
        # print(data_current.shape)

        # reshape and average
        data_mean = np.mean(np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1), 0)
        data_std = np.nanstd(np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1), 0)/math.sqrt(n_sub)

        # plot
        # print(data_mean)
        axs[1].plot(t, data_mean, label=adapter, color=color[iA+1], lw=2)
        axs[1].fill_between(t, data_mean - data_std, data_mean + data_std, alpha=0.2, color=color[iA+1])

        # visualize time windows and response magnitudes
        count = 0
        for _, idx in enumerate(idx_time_windows):

            if (t[idx[0]] < 0):
                continue
            else:

                # compute AUC
                for iS in range(n_sub):
                    # data_temp = data_current[:, iS, :, :, idx[0]:idx[1]].mean(0).mean(0).mean(0)
                    data_temp = data_current[:, iS, :, :, :]
                    data_temp = data_temp.reshape(data_temp.shape[0]*data_temp.shape[1]*data_temp.shape[2], data_temp.shape[3])
                    data_temp = np.nanmean(data_temp[:, idx[0]:idx[1]], 0)
                    AUC[iA, count, iS] = np.nanmean(data_temp)

                # compute metrics
                data_mean = np.nanmean(AUC[iA, count, :])
                data_std = np.nanstd(AUC[iA, count, :])/math.sqrt(n_sub)

                # axs[2].scatter(np.ones(n_sub)*iIdx+offset[iA], AUC_current, color=color[iA+1], s=5, alpha=0.1)
                axs[2].plot([count+offset[iA], count+offset[iA]], [data_mean - data_std, data_mean + data_std], color=color[iA+1], zorder=-10)
                axs[2].scatter(count+offset[iA], data_mean, facecolor=color[iA+1], edgecolor='white', s=80)

                # increment count
                count = count + 1

    # statistics
    print('---------- same vs. different')
    for iT in range(count):
        print('Time window ', iT+1)
        result = stats.ttest_rel(AUC[0, iT, :], AUC[1, iT, :])
        print(result)

    axs[1].plot(t, np.nanmean(data_current_temp, 0), color='grey', linestyle='dashed')

    # create labels for x ticks
    x_ticklabels = []
    for iT in range(count):
        x_ticklabels.append(str(np.round(timepoints[iT, 0], 2)) + ' s - ' + str(np.round(timepoints[iT, 1], 2)) + ' s')
    # print(x_ticklabels)
    print(timepoints*1000)

    # adjust axes
    for i in range(3):

        axs[i].tick_params(axis='both', labelsize=fontsize_tick)
        axs[i].axhline(0, color='lightgrey', lw=0.5, zorder=-10)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

        if i == 0:
            axs[i].axhline(0, color='lightgrey', lw=0.5, zorder=-10)
            axs[i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
            axs[i].set_ylabel(r'$\mu$V', fontsize=fontsize_label)
            axs[i].set_xlabel('Time (s)', fontsize=fontsize_label)
            axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        elif i == 1:
            axs[i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
            axs[i].set_ylabel('adapter - blank \n ERP amplitude ($\mu$V)', fontsize=fontsize_label)
            axs[i].set_xlabel('Time (s)', fontsize=fontsize_label)
            axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        elif i == 2:
            axs[i].set_ylabel(r'Mean amplitude ($\mu$V)', fontsize=fontsize_label)
            axs[i].set_xticks(np.arange(count))
            axs[i].set_xticklabels(x_ticklabels, rotation=45, fontsize=fontsize_tick, ha='right', rotation_mode='anchor')
            axs[i].set_xlim(-0.5, count-1+0.5)

    # save figure
    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name, dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '.svg')

def visualizeContrast_topomap(data, montage, info, dict, contrasts, fig_name, dir):

    # create evoked responses
    times = np.array([0.125, 0.3])
    contrasts_plot = [0, 2, 4]

    # min and maximal voltages
    vmin = -3500000
    vmax = 3500000

    # initiate figure
    fig, axs = plt.subplots(len(times), len(contrasts_plot), figsize=(9, 4))

    print(data[0].shape)
    print(data[1].shape)

    # average dataframes
    data_concat = np.zeros((21, 5, 3, 32, 64, 154))
    data_concat[:, :, :1, :, :, :]         = data[0]
    data_concat[:, :, 1:, :, :, :]         = data[1]
    data_concat = np.nanmean(data_concat, 0)

    for iC in range(len(contrasts_plot)):

        # select data
        data_current = data_concat[contrasts_plot[iC], :, :, :, :]
        
        # print(data_current.shape)
        data_current = data_current.reshape(data_current.shape[0]*data_current.shape[1], data_current.shape[2], data_current.shape[3]).mean(0)
        # data_current = data_current/np.max(data_current)

        # initatie mne
        evoked = mne.EvokedArray(
            data_current, info, tmin=-0.1, nave=data_current.shape[0])

        # set info
        evoked.set_channel_types(dict)
        evoked.set_montage(montage=montage)

        # plot topomap
        evoked.plot_topomap(times, ch_type='eeg', average=0.05, contours=0, vlim=(vmin, vmax), cmap='BrBG', axes=axs[:, iC], colorbar=False, sensors='k.')

    # save figure
    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name, dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '.svg')

    # create colorbar
    fig, ax = plt.subplots(figsize=(8, 1))
    fig.subplots_adjust(bottom=0.7)

    cmap = mpl.cm.BrBG
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
                                norm=norm,
                                orientation='horizontal')
    cb1.set_label(r'$\mu$V', fontsize=fontsize_label)
    cb1.outline.set_visible(False)
    cb1.ax.tick_params(labelsize=fontsize_tick)

    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name + '_colorbar', dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '_colorbar.svg')

def visualizeContrast(data, t, n_sub, channelNames_current, electrode_idx, contrasts, contrasts_value, fig_name, dir):

    # initiate figure
    fig, axs = plt.subplots(1, 2, figsize=(5, 2))

    # concatenate data
    data_concat = np.zeros((21, 5, 3, 32, 64, 154))
    data_concat[:, :, :1, :, :, :]         = data[0]
    data_concat[:, :, 1:, :, :, :]         = data[1]

    t_start     = np.argwhere(t > 0.280)[0][0] # tmin = -0.2
    t_end       = np.argwhere(t > 0.320)[0][0] # tmin = -0.2

    # data to store response magnitude
    AUC = np.zeros((n_sub, len(contrasts)))

    # visualize different contrast levels for repeated trials
    cmap = plt.cm.get_cmap('cool')
    color = cmap(np.linspace(0, 1, len(contrasts)))

    offset_iC = [0, 3, 6, 9, 12]
    offset_iC = [0, 1, 2, 3, 4]
    
    # initiate slope
    slope = np.zeros(n_sub)

    for iC, contrast in enumerate(contrasts):

        # select data
        data_current = data_concat[:, iC, :, :, electrode_idx, :]
        # print(data_current.shape)
        data_current = np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1)
        # print(data_current.shape)

        # reshape and average
        data_mean = np.nanmean(data_current, 0)
        data_std = np.nanstd(data_current, 0)/math.sqrt(n_sub)

        # plot
        axs[0].plot(t, data_mean, label=contrasts_value[iC], color=color[iC])
        axs[0].fill_between(t, data_mean - data_std, data_mean + data_std, alpha=0.2, color=color[iC])

        # plot response magnitude
        for iS in range(n_sub):
            data_current = np.nanmean(np.nanmean(np.nanmean(data_concat[iS, iC, :, :, electrode_idx, :], 0), 0), 0)
            AUC[iS, iC] = np.nanmean(data_current[t_start:t_end])
        mean            = np.nanmean(AUC[:, iC])
        std             = np.nanstd(AUC[:, iC])/math.sqrt(n_sub)

        axs[1].plot([offset_iC[iC], offset_iC[iC]], [mean-std, mean+std], color='grey', zorder=-1)       
        axs[1].scatter(offset_iC[iC], mean, edgecolors='white', color=color[iC], s=40)
        sns.stripplot(x=np.ones(n_sub)*offset_iC[iC], y=AUC[:, iC], jitter=True, ax=axs[1], color=color[iC], size=4, alpha=0.3, native_scale=True)

    # regression analysis
    pred = np.zeros((n_sub, len(contrasts)))
    for iS in range(n_sub):
        lm = LinearRegression()
        model = LinearRegression().fit(np.arange(len(contrasts)).reshape(-1, 1), AUC[iS, :])
        slope[iS] = model.coef_
        pred[iS, :] = model.predict(np.arange(len(contrasts)).reshape(-1, 1))
    axs[1].plot(np.arange(len(contrasts)), np.mean(pred, 0), color='grey', alpha=0.5, zorder=-10)
    
    # statistics
    results = stats.ttest_1samp(slope, 0)
    print(channelNames_current)
    print('Slope 1-sample T-test:'.ljust(25), results)
    print('\n')

    # adjust axes
    for i in range(2):

        axs[i].tick_params(axis='both', labelsize=fontsize_tick)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].axhline(0, color='lightgrey', lw=0.5, zorder=-10)
        axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        if i == 0:
            # axs[i].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                # mode="expand", borderaxespad=0, title='Contrast level', fontsize=fontsize_legend, frameon=False, ncol=len(contrasts))
            axs[i].set_ylabel(r'$\mu$V', fontsize=fontsize_label)
            axs[i].set_xlabel('Time (s)', fontsize=fontsize_label)
            axs[i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
            # axs[i].axvspan(t[t_start], t[t_end], facecolor='lightgrey', edgecolor='white', zorder=-1., alpha=0.5)
            
        elif i == 1:
            axs[i].set_ylabel(r'Mean amplitude ($\mu$V)', fontsize=fontsize_label)
            axs[i].set_xlabel('Target contrast (%)', fontsize=fontsize_label)
            axs[i].set_xticks(offset_iC)
            axs[i].set_xticklabels(['50', '60', '70', '80', '90'])

    # save figure
    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name, dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '.svg')

def visualize_AdapterContrast(data, t, n_sub, channelNames_current, electrode_idx, adapters, contrasts, contrasts_value, fig_name, dir):

    # timecourse to display
    t_start     = np.argwhere(t > 0.280)[0][0] # tmin = -0.2
    t_end       = np.argwhere(t > 0.320)[0][0] # tmin = -0.2
    # print(t_start)

    # initiate figure
    fig, axs = plt.subplots(2, 3, figsize=(10, 4))

    # concatenate data
    data_concat = np.zeros((21, 5, 3, 32, 64, 154))
    data_concat[:, :, :1, :, :, :]         = data[0]
    data_concat[:, :, 1:, :, :, :]         = data[1]
    # print(data_concat.shape)

    # visualize different contrast levels for repeated trials
    cmap = plt.cm.get_cmap('cool')
    color_contrasts = cmap(np.linspace(0, 1, len(contrasts)))

    # save response magnitude levels
    AUC_abs         = np.zeros((len(adapters), len(contrasts), n_sub))
    slope           = np.zeros((len(adapters), n_sub))

    # visualize different contrast levels for repeated trials
    cmap = plt.cm.get_cmap('cool')
    color = cmap(np.linspace(0, 1, len(contrasts)))

    offset_iC = [0, 3, 6, 9, 12]
    offset_iC = [0, 1, 2, 3, 4]

    # create strings for subject
    subject_str = []
    for i in range(n_sub):
        subject_str.append('sub' + str(i+1))

    # initiate dataframe for lmer
    df_stats = pd.DataFrame()
    df_stats['subject'] = np.tile(subject_str, len(adapters)*len(contrasts))
    df_stats['contrast'] = np.repeat(contrasts, len(adapters)*(n_sub))
    df_stats['adapter'] = np.tile(np.repeat(adapters, n_sub), len(contrasts))
    df_stats['dependentVar'] = 0
    print(df_stats)

    # visualize absolute
    for iA, adapter in enumerate(adapters):

        for iC, contrast in enumerate(contrasts):

            # select data
            data_current = data_concat[:, iC, iA, :, electrode_idx, :]
            # print(data_current.shape)
            data_current = data_current.mean(0)
            data_current = np.nanmean(data_current, 1)

            # reshape and average
            data_mean = np.nanmean(data_current, 0)
            data_std = np.nanstd(data_current, 0)/math.sqrt(n_sub)

            # plot
            axs[0, iA].plot(t, data_mean, label=contrasts_value[iC], color=color_contrasts[iC])
            axs[0, iA].fill_between(t, data_mean - data_std, data_mean + data_std, alpha=0.1, color=color_contrasts[iC])

            # compute respone magnitude
            for iS in range(n_sub):

                # compute
                AUC_abs[iA, iC, iS] = np.mean(abs(data_current[iS, t_start:t_end]))

                # add to dataframe
                idx = df_stats[(df_stats.subject == subject_str[iS]) & (df_stats.contrast == contrast) & (df_stats.adapter == adapter)].index
                df_stats.loc[idx, 'dependentVar'] = AUC_abs[iA, iC, iS]

            # AUC[:, iA, iC] = np.abs(AUC[:, iA, iC])
            mean            = np.nanmean(AUC_abs[iA, iC, :])
            std             = np.nanstd(AUC_abs[iA, iC, :])/math.sqrt(n_sub)

            axs[1, iA].plot([offset_iC[iC], offset_iC[iC]], [mean-std, mean+std], color=color_contrasts[iC], zorder=-1)       
            axs[1, iA].scatter(offset_iC[iC], mean, edgecolors='white', color=color_contrasts[iC], s=60)
            sns.stripplot(x=np.ones(n_sub)*offset_iC[iC], y=AUC_abs[iA, iC, :], jitter=True, ax=axs[1, iA], color=color_contrasts[iC], size=4, zorder=-10, alpha=0.3, native_scale=True)

        # compute slope
        pred = np.zeros((n_sub, len(contrasts)))
        for iS in range(n_sub):
            model = LinearRegression().fit(np.arange(len(contrasts)).reshape(-1, 1), AUC_abs[iA, :, iS])
            slope[iA, iS] = model.coef_
            pred[iS, :] = model.predict(np.arange(len(contrasts)).reshape(-1, 1))

        # visualize
        axs[1, iA].plot(offset_iC, np.mean(pred, 0), color='grey', lw=0.75, alpha=0.5, zorder=-10)

        # statistical testing
        print('----------' + adapter)
        results = stats.ttest_1samp(slope[iA, :], 0)
        print(results)


    # adjust labels
    for i in range(len(adapters)):
        axs[0, i].tick_params(axis='both', labelsize=fontsize_tick)
        axs[0, i].spines['top'].set_visible(False)
        axs[0, i].spines['right'].set_visible(False)
        axs[0, i].axhline(0, color='lightgrey', lw=0.5, zorder=-10) 
        axs[0, i].set_xlabel('Time (s)', fontsize=fontsize_label)
        axs[0, i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
        axs[0, i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[0, i].set_ylim(-4, 5)
        if i == 0:
            axs[0, i].set_ylabel(r'$\mu$V', fontsize=fontsize_label)

    for i in range(len(adapters)):
        axs[1, i].tick_params(axis='both', labelsize=fontsize_tick)
        axs[1, i].spines['top'].set_visible(False)
        axs[1, i].spines['right'].set_visible(False)
        # axs[1, i].axhline(0, color='lightgrey', lw=0.5, zorder=-10)
        # axs[1, i].set_xlabel('Time (s)', fontsize=fontsize_label)
        # axs[1, i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
        axs[1, i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        axs[1, i].set_xticks(offset_iC)
        axs[1, i].set_xticklabels(['50', '60', '70', '80', '90'])
        axs[1, i].set_xlabel('Target contrast (%)', fontsize=fontsize_label)
        if i == 0:
            axs[1, i].set_ylabel(r'Mean amplitude ($\mu$V)', fontsize=fontsize_label)

    # save figure
    fig.align_ylabels()
    plt.tight_layout()
    plt.savefig(dir + 'visualization/' + fig_name, dpi=600)
    plt.savefig(dir + 'visualization/' + fig_name + '.svg')

    return df_stats


def visualizeContrastAdapter_CCN24(data, t, n_sub, channelNames_current, electrode_idx, adapters, contrasts, contrasts_value, fig_name, dir):

    # initiate figure
    fig, axs = plt.subplots(1, 4, figsize=(10, 2))
    fig.subplots_adjust(wspace=1.5)

    ################################################################## ADAPTER
    ##################################################################
    ##################################################################

    # concatenate data
    data_concat = np.zeros((21, 5, 3, 32, 64, 154))
    data_concat[:, :, :1, :, :, :]         = data[0]
    data_concat[:, :, 1:, :, :, :]         = data[1]
    # print(data_concat.shape)

    # visualize different contrast levels for repeated trials
    color = ['gray', 'dodgerblue', np.array([212, 170, 0])/255]

    # visualize absolute values
    for iA, adapter in enumerate(adapters):

        # select data
        data_current = data_concat[:, :, iA, :, electrode_idx, :]
        data_current = np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1)

        # reshape and average
        data_mean = np.nanmean(data_current, 0)
        data_std = np.nanstd(data_current, 0)/math.sqrt(n_sub)

        # plot
        axs[0].plot(t, data_mean, label=adapter, color=color[iA], lw=2)
        axs[0].fill_between(t, data_mean - data_std, data_mean + data_std, alpha=0.2, color=color[iA])

    # compute data
    data_current_temp = np.nanmean(data[1], 2).squeeze() - data[0].squeeze()
    data_current_temp = data_current_temp[:, :, :, electrode_idx, :]
    data_current_temp = np.nanmean(np.nanmean(np.nanmean(data_current_temp, 1), 1), 1)

    # compute timepoints where responses are significant
    p_values = np.zeros(len(t))
    for tmp in range(len(t)):

        # parametric
        p_values[tmp] = stats.ttest_1samp(data_current_temp[:, tmp], 0)[1]

    # MC correction
    # _, p_values, _, _ = multipletests(p_values, method='fdr_bh')
    pvals_corrected_sign = np.argwhere(p_values < 0.05).flatten()

    # define time windows
    idx_time_windows = []
    start_idx = None

    for i in range(len(pvals_corrected_sign) - 1):
        if pvals_corrected_sign[i + 1] - pvals_corrected_sign[i] == 1:
            if start_idx is None:
                start_idx = pvals_corrected_sign[i]
        else:
            if start_idx is not None:
                idx_time_windows.append([start_idx, pvals_corrected_sign[i]])
                start_idx = None

    # Check if there are consecutive numbers at the end of the array
    if start_idx is not None:
        idx_time_windows.append([start_idx, pvals_corrected_sign[-1]])

    # compute response magnitude differences
    time_windows    = len(idx_time_windows)
    timepoints      = np.zeros((time_windows,  2))

    # plot time ranges
    count = 0
    for _, idx in enumerate(idx_time_windows):

        if (t[idx[0]] < 0):
            continue
        else:

            # retrieve timepoints
            start = t[idx[0]]
            end = t[idx[1]]
            timepoints[count, 0] = start
            timepoints[count, 1] = end

            # visualize 
            axs[0].axvspan(start, end, facecolor='lightsalmon', alpha=0.3, edgecolor='white')

        # increment count
        count = count + 1

    # visualize relative values
    data_none = data_concat[:, :, 0, :, electrode_idx, :]

    # save response magnitudes
    AUC             = np.zeros((len(adapters)-1, time_windows, n_sub))

    # save x-ticks
    offset = [-0.1, 0.1]
    for iA, adapter in enumerate(adapters[1:]):

        # select data
        data_current = data_concat[:, :, iA+1, :, electrode_idx, :] - data_none
        # print(data_current.shape)

        # reshape and average
        data_mean = np.mean(np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1), 0)
        data_std = np.nanstd(np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1), 0)/math.sqrt(n_sub)

        # visualize time windows and response magnitudes
        count = 0
        for _, idx in enumerate(idx_time_windows):

            if (t[idx[0]] < 0):
                continue
            else:

                # compute AUC
                for iS in range(n_sub):
                    # data_temp = data_current[:, iS, :, :, idx[0]:idx[1]].mean(0).mean(0).mean(0)
                    data_temp = data_current[:, iS, :, :, :]
                    data_temp = data_temp.reshape(data_temp.shape[0]*data_temp.shape[1]*data_temp.shape[2], data_temp.shape[3])
                    data_temp = np.nanmean(data_temp[:, idx[0]:idx[1]], 0)
                    AUC[iA, count, iS] = np.nanmean(data_temp)

                # compute metrics
                data_mean = np.nanmean(AUC[iA, count, :])
                data_std = np.nanstd(AUC[iA, count, :])/math.sqrt(n_sub)

                # axs[2].scatter(np.ones(n_sub)*iIdx+offset[iA], AUC_current, color=color[iA+1], s=5, alpha=0.1)
                axs[1].plot([count+offset[iA], count+offset[iA]], [data_mean - data_std, data_mean + data_std], color=color[iA+1], zorder=-10)
                axs[1].scatter(count+offset[iA], data_mean, facecolor=color[iA+1], edgecolor='white', s=80)

                # increment count
                count = count + 1

    # statistics
    print('---------- same vs. different')
    for iT in range(count):
        print('Time window ', iT+1)
        result = stats.ttest_rel(AUC[0, iT, :], AUC[1, iT, :])
        print(result)

    # create labels for x ticks
    x_ticklabels = []
    for iT in range(count):
        x_ticklabels.append(str(np.round(timepoints[iT, 0], 2)) + ' s - ' + str(np.round(timepoints[iT, 1], 2)) + ' s')
    # print(x_ticklabels)
    print(timepoints*1000)

    # adjust axes
    for i in range(2):

        axs[i].tick_params(axis='both', labelsize=fontsize_tick)
        axs[i].axhline(0, color='lightgrey', lw=0.5, zorder=-10)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)

        if i == 0:
            axs[i].axhline(0, color='lightgrey', lw=0.5, zorder=-10)
            axs[i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
            # axs[i].set_ylabel(r'$\mu$V', fontsize=fontsize_label)
            # axs[i].set_xlabel('Time (s)', fontsize=fontsize_label)
            axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        elif i == 1:
            axs[i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
            # axs[i].set_ylabel('adapter - blank \n ERP amplitude ($\mu$V)', fontsize=fontsize_label)
            # axs[i].set_xlabel('Time (s)', fontsize=fontsize_label)
            axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            axs[i].set_xticklabels([' ', ' ', ' '])

        
    ################################################################## CONTRAST
    ##################################################################
    ##################################################################

    t_start     = np.argwhere(t > 0.260)[0][0] # tmin = -0.2
    t_end       = np.argwhere(t > 0.350)[0][0] # tmin = -0.2

    # visualize 
    axs[2].axvspan(t[t_start], t[t_end], facecolor='lightsalmon', alpha=0.3, edgecolor='white')

    # concatenate data
    data_concat = np.zeros((21, 5, 3, 32, 64, 154))
    data_concat[:, :, :1, :, :, :]         = data[0]
    data_concat[:, :, 1:, :, :, :]         = data[1]

    t_start     = np.argwhere(t > 0.280)[0][0] # tmin = -0.2
    t_end       = np.argwhere(t > 0.320)[0][0] # tmin = -0.2

    # data to store response magnitude
    AUC = np.zeros((n_sub, len(contrasts)))

    # visualize different contrast levels for repeated trials
    cmap = plt.cm.get_cmap('cool')
    color = cmap(np.linspace(0, 1, len(contrasts)))

    offset_iC = [0, 3, 6, 9, 12]
    offset_iC = [0, 1, 2, 3, 4]
    
    # initiate slope
    slope = np.zeros(n_sub)

    for iC, contrast in enumerate(contrasts):
        print(contrasts)

        # select data
        data_current = data_concat[:, iC, :, :, electrode_idx, :]
        # print(data_current.shape)
        data_current = np.nanmean(np.nanmean(np.nanmean(data_current, 0), 1), 1)
        # print(data_current.shape)

        # reshape and average
        data_mean = np.nanmean(data_current, 0)
        data_std = np.nanstd(data_current, 0)/math.sqrt(n_sub)

        # plot
        axs[2].plot(t, data_mean, label=contrasts_value[iC], color=color[iC])
        axs[2].fill_between(t, data_mean - data_std, data_mean + data_std, alpha=0.2, color=color[iC])

        # plot response magnitude
        for iS in range(n_sub):
            data_current = np.nanmean(np.nanmean(np.nanmean(data_concat[iS, iC, :, :, electrode_idx, :], 0), 0), 0)
            AUC[iS, iC] = np.nanmean(data_current[t_start:t_end])
        mean            = np.nanmean(AUC[:, iC])
        std             = np.nanstd(AUC[:, iC])/math.sqrt(n_sub)

        axs[3].plot([offset_iC[iC], offset_iC[iC]], [mean-std, mean+std], color='grey', zorder=-1)       
        axs[3].scatter(offset_iC[iC], mean, edgecolors='white', color=color[iC], s=40)
        sns.stripplot(x=np.ones(n_sub)*offset_iC[iC], y=AUC[:, iC], jitter=True, ax=axs[3], color=color[iC], size=4, alpha=0.3, native_scale=True)

    # regression analysis
    pred = np.zeros((n_sub, len(contrasts)))
    for iS in range(n_sub):
        lm = LinearRegression()
        model = LinearRegression().fit(np.arange(len(contrasts)).reshape(-1, 1), AUC[iS, :])
        slope[iS] = model.coef_
        pred[iS, :] = model.predict(np.arange(len(contrasts)).reshape(-1, 1))
    axs[3].plot(np.arange(len(contrasts)), np.mean(pred, 0), color='grey', alpha=0.5, zorder=-10)
    
    # statistics
    results = stats.ttest_1samp(slope, 0)
    print(channelNames_current)
    print('Slope 1-sample T-test:'.ljust(25), results)
    print('\n')

    # adjust axes
    for i in range(2, 4):

        axs[i].tick_params(axis='both', labelsize=fontsize_tick)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
        axs[i].axhline(0, color='lightgrey', lw=0.5, zorder=-10)
        axs[i].ticklabel_format(axis='y', style='sci', scilimits=(0,0))

        if i == 0:
            # axs[i].legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                # mode="expand", borderaxespad=0, title='Contrast level', fontsize=fontsize_legend, frameon=False, ncol=len(contrasts))
            axs[i].set_ylabel(r'$\mu$V', fontsize=fontsize_label)
            axs[i].set_xlabel('Time (s)', fontsize=fontsize_label)
            axs[i].axvline(0, color='lightgrey', lw=0.5, zorder=-10)
            # axs[i].axvspan(t[t_start], t[t_end], facecolor='lightgrey', edgecolor='white', zorder=-1., alpha=0.5)
            
        elif i == 3:
            # axs[i].set_ylabel(r'Mean amplitude ($\mu$V)', fontsize=fontsize_label)
            # axs[i].set_xlabel('Contrast', fontsize=fontsize_label)
            axs[i].set_xticks(offset_iC)
            axs[i].set_xticklabels([' ', ' ', ' ', ' ', ' '])

    # save figure
    plt.tight_layout()
    plt.savefig(dir + 'visualization/adapterContrast_CCN24', dpi=600)
    plt.savefig(dir + 'visualization/adapterContrast_CCN24.svg')


