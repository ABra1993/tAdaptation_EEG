import torch as torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

def compute_rms_contrast_time_series(batch_sequence):
    """
    Compute RMS contrast for each image at each timestep in a batch.
    Input:
        batch_sequence: Tensor of shape (B, T, 1, H, W)
    Output:
        rms_contrast: Tensor of shape (B, T)
    """
    B, T, C, H, W = batch_sequence.shape
    x = batch_sequence.view(B * T, C, H, W).squeeze(1)  # shape (B*T, H, W)

    mean = x.mean(dim=(1, 2), keepdim=True)  # shape (B*T, 1, 1)
    std = ((x - mean) ** 2).mean(dim=(1, 2)).sqrt()     # shape (B*T,)
    rms = std / (mean.squeeze() + 1e-8)

    return rms.view(B, T)  # reshape back to (B, T)

def encode_timesteps(t_steps, sequence):

    # initiate timesteps
    t_steps_label = torch.zeros(t_steps)

    for idx, s in enumerate(sequence):
        if s == 'A':
            t_steps_label[idx] = 1
        elif s == 'T':
            t_steps_label[idx] = 2

    return t_steps_label

def create_sequence(imgs, t_steps, t_steps_label, create_sample, contrast, contrast_low, contrast_high, root=None, sequence=None):
    
    # create list
    ax = []

    # add random noise
    if contrast == None:
        img_adjusted = imgs.clone()
        contrast = np.random.uniform(contrast_low, contrast_high, imgs.shape[0])
        contrast = torch.from_numpy(contrast)
        for iB in range(imgs.shape[0]):
            img_adjusted[iB, :, :, :] = F.adjust_contrast(imgs[iB, :, :, :], contrast[iB])
    else:
        img_adjusted = F.adjust_contrast(imgs, contrast)

    # define noise
    noise = [create_adapter('pink', [imgs.shape[2], imgs.shape[3]]) for _ in range(imgs.shape[0])] # pink noise
    noise = torch.stack(noise)

    noise2 = [create_adapter('pink', [imgs.shape[2], imgs.shape[3]]) for _ in range(int(imgs.shape[0]/2))] # pink noise
    noise2 = torch.stack(noise2)
    
    # initiate figure
    if create_sample:

        fig, axs = plt.subplots(2, t_steps+1) #, figsize=(10, 4))

        # plot ground truth
        axs[0, 0].imshow(imgs[0, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
        axs[0, 0].set_ylabel('Same')
        axs[0, 3].set_title('Contrast: ' + str(np.round(contrast[0].item(), 2)), fontsize=8)

        axs[1, 0].imshow(imgs[-1, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
        axs[1, 0].set_ylabel('Different')
        axs[1, 3].set_title('Contrast: ' + str(np.round(contrast[-1].item(), 2)), fontsize=8)

    for t in range(t_steps):

        if create_sample:
            axs[0, 1+t].axis('off')
            axs[1, 1+t].axis('off')

        if t_steps_label[t] == 1:               # adapter

            # clamp
            noise_clamp = torch.clamp(noise, 0, 1)
            ax.append(noise_clamp)

            # plot
            if create_sample:
                axs[0, 1+t].imshow(ax[t][0, :, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)
                axs[1, 1+t].imshow(ax[t][-1,:, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)
            
        elif t_steps_label[t] == 0:

            # append
            input_shape = torch.ones(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])*.2
            ax.append(input_shape)

            # plot
            if create_sample:
                axs[0, 1+t].imshow(ax[t][0,:, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
                axs[1, 1+t].imshow(ax[t][-1,:, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
    
        elif t_steps_label[t] == 2:             # test

            # create input sequence
            input_shape = torch.ones(imgs.shape[0], imgs.shape[1], imgs.shape[2], imgs.shape[3])
            
            # clamp and create
            input_shape[:int(imgs.shape[0]/2), :, : , :] = noise[:int(imgs.shape[0]/2)] + img_adjusted[:int(imgs.shape[0]/2)]
            input_shape[int(imgs.shape[0]/2):, :, : , :] = noise2 + img_adjusted[int(imgs.shape[0]/2):]
            input_shape = torch.clamp(input_shape, 0, 1)
            ax.append(input_shape)
            
            # plot
            if create_sample:
                axs[0, 1+t].imshow(ax[t][0, :, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)
                axs[1, 1+t].imshow(ax[t][-1,:, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)

    if create_sample:

        # save figure
        plt.savefig(root + 'visualization/input_sequences/' + sequence, dpi=300)
        plt.savefig(root + 'visualization/input_sequences/' + sequence + '.svg')

    return contrast, ax


def noise_psd(N, psd = lambda f: 1):
        X_white = np.fft.rfft2(torch.rand(N[0], N[1]).numpy())
        S = psd(np.fft.rfftfreq(N[1]))
        S = S / np.sqrt(np.mean(S**2))      # normalize S
        X_shaped = X_white * S
        return np.fft.irfft2(X_shaped)

def PSDGenerator(f):
    return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
    return 1

@PSDGenerator
def blue_noise(f):
    return np.sqrt(f)

@PSDGenerator
def violet_noise(f):
    return f

@PSDGenerator
def brownian_noise(f):
    return 1/np.where(f == 0, float('inf'), f)

@PSDGenerator
def pink_noise(f):
    return 1/np.where(f == 0, float('inf'), np.sqrt(f))

def create_adapter(noise, input_shape):

    if noise == 'white':
        noise_x = white_noise(input_shape)
        noise_y = white_noise(input_shape).transpose()
        return torch.Tensor(noise_x + noise_y).unsqueeze(0)
    elif noise == 'blue':
        noise_x = blue_noise(input_shape)
        noise_y = blue_noise(input_shape).transpose()
        return torch.Tensor(noise_x + noise_y).unsqueeze(0)
    elif noise == 'violet':
        noise_x = violet_noise(input_shape)
        noise_y = violet_noise(input_shape).transpose()
        return torch.Tensor(noise_x + noise_y).unsqueeze(0)
    elif noise == 'brownian':
        noise_x = brownian_noise(input_shape)
        noise_y = brownian_noise(input_shape).transpose()
        return torch.Tensor(noise_x + noise_y).unsqueeze(0)
    elif noise == 'pink':
        noise_x = pink_noise(input_shape)
        noise_y = pink_noise(input_shape).transpose()
        return torch.Tensor(noise_x + noise_y).unsqueeze(0)


def sequence_train(imgs, *args):

    # # set the seed
    # seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True

    # initiate input
    ax = []

    # set contrasts
    contrasts       = torch.Tensor([0.2, 1])
    batch_size      = int(imgs.shape[0]/len(contrasts))

    # create noise pattern
    noise   = torch.randn_like(imgs)*.2
    noise2  = torch.randn_like(imgs)*.2

    # number of timesteps
    n_timestep = 3

    # compute mean of test images
    mean_imgs = torch.mean(imgs, dim=[2, 3], keepdim=True).expand(imgs.shape)

    # adjust contrast
    imgs_adjusted = imgs.clone()
    for iC, contrast in enumerate(contrasts):
        imgs_adjusted[iC*batch_size:(iC+1)*batch_size, :, :, :] = (imgs[iC*batch_size:(iC+1)*batch_size, :, :, :] - mean_imgs[iC*batch_size:(iC+1)*batch_size, :, :, :]) * contrast # + mean_imgs[iC*batch_size:(iC+1)*batch_size, :, :, :]

    # add mean to noise
    noise_adjusted = (noise + mean_imgs)

    # create sequence
    for t in range(n_timestep):
        if t == 0:
            ax.append(noise_adjusted.clamp(0, 1))
        elif t == 1:
            blank = torch.ones_like(imgs) * mean_imgs
            ax.append(blank)
        elif t == 2:
            test = noise_adjusted + imgs_adjusted
            ax.append(test.clamp(0, 1))

    return ax