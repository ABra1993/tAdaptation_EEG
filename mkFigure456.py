import numpy as np
from mne.defaults import HEAD_SIZE_DEFAULT
from mne.channels._standard_montage_utils import _read_theta_phi_in_degrees

from mkFigure456_utils import *

# set root directory
root        = '/home/amber/Documents/organize_code/nAdaptation_EEG_git/'

# set figure: ['Fig4A', 'Fig4B', 'Fig4C', 'Fig4D', 'Fig5A', 'Fig5B', 'Fig5C', 'Fig5D', 'Fig6AB']
fig_names = ['Fig4A', 'Fig4B', 'Fig4C', 'Fig4D', 'Fig5A', 'Fig5B', 'Fig5C', 'Fig5D', 'Fig6AB']

fig_names = ['Fig5B', 'Fig5C', 'Fig5D']  # for testings
fig_names = ['Fig6AB']  # for testings

# montage
fname = root + 'config/chs.tsv'
montage = _read_theta_phi_in_degrees(fname=fname, head_size=HEAD_SIZE_DEFAULT,
                                    fid_names=['Nz', 'LPA', 'RPA'],
                                    add_fiducials=False)

# define targets
targets                         = [3, 6, 8, 9]
n_targets                       = len(targets)

# trial types
trial_type_clean                = 'clean'
trial_type_single               = 'single'
trial_type_repeated             = 'repeated'

# number of repetitions
repetitions_clean               = 100 * n_targets
repetitions_single              = 8 * n_targets
repetitions_repeated            = 8 * n_targets

# contrast
contrast_clean                  = ['f_contrast']
contrast_clean_values           = [1]

contrast_single                 = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
contrast_single_values          = [0.50, 0.60, 0.70, 0.80, 0.90]

contrast_repeated               = ['l_contrast', 'lm_contrast', 'm_contrast', 'mh_contrast', 'h_contrast']
contrast_repeated_values        = [0.50, 0.60, 0.70, 0.80, 0.90]

# adapter
adapter_clean                   = ['none']
adapter_single                  = ['none']
adapter_repeated                = ['same', 'different']

# image types for which we are going to extract the trigger numbers
img_type_clean                  = ['test']
img_type_single                 = ['test']
img_type_repeated               = ['test']

# fontsize
fontsize_title          = 15
fontsize_legend         = 10
fontsize_label          = 12
fontsize_tick           = 10

# concatenate information for the different trial types
trial_types                     = [trial_type_clean, trial_type_single, trial_type_repeated]
repetitions                     = [repetitions_clean, repetitions_single, repetitions_repeated]
contrasts                       = [contrast_clean, contrast_single, contrast_repeated]
contrasts_value                 = [contrast_clean_values, contrast_single_values, contrast_repeated_values]
adapters                        = [adapter_clean, adapter_single, adapter_repeated]
img_types                       = [img_type_clean, img_type_single, img_type_repeated]

# import data
data_clean          = np.load(root + 'data/EEG/data_clean_epochSelection.npy')
data_single         = np.load(root + 'data/EEG/data_single_epochSelection.npy')
data_repeated       = np.load(root + 'data/EEG/data_repeated_epochSelection.npy')
data                = [data_clean*1000, data_single*1000, data_repeated*1000] # convert from V to microVolt

num_nan = np.sum(np.isnan(data_clean))
print(num_nan)

n_sub               = data[0].shape[0]

# set timecourse
tmin                    = -0.1
tmax                    = 0.5
sample_rate             = 2048
down_sample_rate        = 256

n_timepoints            = math.ceil((abs(tmin) + tmax)/(1/down_sample_rate))
t                       = np.arange(n_timepoints)*(1/down_sample_rate)+tmin

# all channels names
channelNames = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'AFz', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2']

# add channel i
dict = {}
for i in range(len(channelNames)):
    dict[channelNames[i]]  = 'eeg'
print(dict)

# create info
n_channels = len(channelNames)
sampling_freq = down_sample_rate  # in Hertz
info = mne.create_info(channelNames, sfreq=sampling_freq)

# shape (n_sub x contrast x adapter x channels x timepoints)
print('Shape data struct for clean trials: '.ljust(50), data_clean.shape)
print('Shape data struct for single trials: '.ljust(50), data_single.shape)
print('Shape data struct for repeated trials: '.ljust(50), data_repeated.shape)

# create figures
for fig in fig_names:

    # print progress
    print('Creating figure: ', fig)

    ################# electrode selection Fig. 4 (effect of adapter)
    channelNames_current = []
    if (fig == 'Fig4C') | (fig == 'Fig5D'):
        channelNames_current = ['Iz', 'Oz', 'O1', 'O2']
    elif (fig == 'Fig4D') | (fig == 'Fig5B') | (fig == 'Fig6AB'):
        channelNames_current = ['P9', 'P10']
    elif (fig == 'Fig5C'):
        channelNames_current = ['Pz', 'P1', 'P2', 'P3', 'P4']

    # extract indices
    if len(channelNames_current) != 0:
        electrode_idx = []
        for i in range(len(channelNames_current)):
            idx = np.argwhere(np.array(channelNames) == np.array(channelNames_current[i]))[0][0]
            electrode_idx.append(idx)
        print('Electrode indices:', electrode_idx)

    # visualize
    if fig == 'Fig4A':
        visualizeAdapter_topomap(data=[data[1], data[2]], montage=montage, info=info, dict=dict, adapters=adapter_single+adapter_repeated, fig_name=fig, dir=root)
    elif fig == 'Fig4B':
        visualizeAdapter_topomap_diff(data=[data[1], data[2]], montage=montage, info=info, dict=dict, adapters=adapter_single+adapter_repeated, fig_name=fig, dir=root)
    elif (fig == 'Fig4C') | (fig == 'Fig4D'):
        visualizeAdapter(data=[data[1], data[2]], t=t, n_sub=n_sub, channelNames_current=channelNames_current, electrode_idx=electrode_idx, adapters=adapter_single+adapter_repeated, fig_name=fig, dir=root)
    elif (fig == 'Fig5A'):
        visualizeContrast_topomap(data=[data[1], data[2]], montage=montage, info=info, dict=dict, contrasts=contrasts[1], fig_name=fig, dir=root)
    elif (fig == 'Fig5B') | (fig == 'Fig5C') | (fig == 'Fig5D'):
        visualizeContrast(data=[data[1], data[2]], t=t, n_sub=n_sub, channelNames_current=channelNames_current, electrode_idx=electrode_idx, contrasts=contrasts[1], contrasts_value=contrasts_value[1], fig_name=fig, dir=root)
    elif (fig == 'Fig6AB'):

        # create figure
        df_stats = visualize_AdapterContrast(data=[data[1], data[2]], t=t, n_sub=n_sub, channelNames_current=channelNames_current, electrode_idx=electrode_idx, adapters=adapter_single+adapter_repeated, contrasts=contrast_repeated, contrasts_value=contrast_repeated_values, fig_name=fig, dir=root)
    
        # save stats
        df_stats.to_csv(root+'data/EEG/meanAmplitude_adapterContrast.csv', sep=',', index=0)

    else:
        print('Figure name does not exist...')

print('Figures created!')