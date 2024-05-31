import torch as torch
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import numpy as np

def encode_timesteps(t_steps, start, dur):

    # initiate timesteps
    t_steps_label = torch.zeros(t_steps)

    # # adapter
    t_steps_label[start[0]:start[0]+dur[0]] = 1
    
    # # test
    t_steps_label[start[1]:start[1]+dur[1]] = 2

    return t_steps_label

def sequence_train(imgs, t_steps, t_steps_label, create_sample):
    
    # create placeholrder input sequence
    ax = torch.ones(imgs.shape[0], t_steps, imgs.shape[1], imgs.shape[2], imgs.shape[3])

    # convert to greyscale
    ax = torch.multiply(ax, 0.5)

    # define noise
    noise = [create_adapter('pink', [imgs.shape[2], imgs.shape[3]]) for _ in range(imgs.shape[0])] # pink noise
    noise = torch.stack(noise)

    noise2 = [create_adapter('pink', [imgs.shape[2], imgs.shape[3]]) for _ in range(int(imgs.shape[0]/2))] # pink noise
    noise2 = torch.stack(noise2)
    
    # initiate figure
    if create_sample:

        fig, axs = plt.subplots(2, t_steps+1) #, figsize=(10, 4))

        # plot ground truth
        axs[0, 0].axis('off')
        axs[1, 0].axis('off')
        axs[0, 0].imshow(imgs[0, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
        axs[0, 0].set_ylabel('Same')
        axs[1, 0].imshow(imgs[-1, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
        axs[1, 0].set_ylabel('Different')

    for t in range(t_steps):

        if create_sample:
            axs[0, 1+t].axis('off')
            axs[1, 1+t].axis('off')

        if t_steps_label[t] == 1:               # adapter

            # clamp
            noise_clamp = torch.clamp(noise, 0, 1)
            ax[:, t, :, : , :] = noise_clamp

            # plot
            if create_sample:
                axs[0, 1+t].imshow(ax[0, t, :, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)
                axs[1, 1+t].imshow(ax[-1, t, :, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)
            
        elif t_steps_label[t] == 0:

            # plot
            if create_sample:
                axs[0, 1+t].imshow(ax[0, t, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
                axs[1, 1+t].imshow(ax[-1, t, :, :, :].reshape(28, 28, 1), cmap='gray', vmin=0, vmax=1)
    
        elif t_steps_label[t] == 2:             # test
            
            # clamp and create
            ax[:int(imgs.shape[0]/2), t, :, : , :] = noise[:int(imgs.shape[0]/2)] + imgs[:int(imgs.shape[0]/2)]
            ax[int(imgs.shape[0]/2):, t, :, : , :] = noise2 + imgs[int(imgs.shape[0]/2):]
            ax = torch.clamp(ax, 0, 1)
            
            # plot
            if create_sample:
                axs[0, 1+t].imshow(ax[0, t, :, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)
                axs[1, 1+t].imshow(ax[-1, t, :, :, :].reshape(28, 28, 1), cmap='gray') #, vmin=0, vmax=1)

    if create_sample: # return figure depicting a training sample
        return ax, fig
    else:
        return ax


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

