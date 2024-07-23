# %%
# import packages
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torch.nn as nn
from torch import optim

import torchvision.transforms.functional as F
import torchvision.transforms as transforms

import numpy as np

import matplotlib.pyplot as plt

import random
import neptune.new as neptune

# models
from models.cnn_feedforward import cnn_feedforward
from model_train_utils import *

# set the seed
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

def main():

    # track with neptune
    track_with_neptune = False

    # input specification
    dataset                 = 'mnist'

    # specification of image sequence
    t_steps         = 3
    dur             = [1, 1]
    start           = [0, 2]

    # hyperparameter specification
    init            = 10
    numepchs        = 10
    batch_size      = 100
    lr              = 0.001

    # tempDynamics ['none', 'add_supp', 'div_norm', 'lat_recurrence', 'lat_recurrence_mult']
    # tempDynamics            = ['none', 'add_supp', 'div_norm', 'lat_recurrence', 'lat_recurrence_mult']
    tempDynamics = ['add_supp']

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # set home directory
    download            = False # can be set to False after download
    root_data           = root + 'models/dataset/'
    print(root_data)
    if dataset == 'mnist':
        trainData       = datasets.MNIST(root=root_data, download=download, train=True, transform = transforms.ToTensor())
        testData        = datasets.MNIST(root=root_data, download=download, train=False, transform = transforms.ToTensor())
    elif dataset == 'fmnist':
        trainData       = datasets.FashionMNIST(root=root_data, download=download, train=True, transform = transforms.ToTensor())
        testData        = datasets.FashionMNIST(root=root_data, download=download, train=False, transform = transforms.ToTensor())

    # Filter train set to include only digits 3, 6, 8, and 9 (if needed)

    # Define data loaders with drop_last=True
    ldrTrain = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    ldrTest = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False, drop_last=True)

    print('Size dataset during training: ', len(trainData))
    print('Size dataset during inference: ', len(testData))

    # training
    for _, current_tempDynamics in enumerate(tempDynamics):

        # initiate dataframe to store accuracies
        accu = torch.zeros((2, init)) # same/different or low/high contrast

        for iInit in range(init):

            # initiate neptune monitoring
            if track_with_neptune:
                run = neptune.init_run(
                    project="abra1993/adapt-dnn",
                    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODkxNGY3NS05NGJlLTQzZDEtOGU5Yy0xMjJlYzI0YzE2YWUifQ==",
                )  # your credentials

                run['name']         = current_tempDynamics
                run['dataset']      = dataset
                run['numepchs']     = numepchs

            # retrieve sequence
            t_steps_label = encode_timesteps(t_steps, start, dur)
            print(t_steps_label)

            # initiate model
            model = cnn_feedforward(t_steps, current_tempDynamics)

            # initiate recurrence
            if tempDynamics != 'none':
                model.initialize_tempDynamics()
            model.to(device)

            # loss function and optimizer
            lossfunct = nn.CrossEntropyLoss()   
            optimizer = optim.Adam(model.parameters(), lr=lr)

            # train
            print('\nTraining in progress...')
            model.train()

            for epoch in range(numepchs): # images and labels
                for a, (imgs, lbls) in enumerate(ldrTrain):

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # create input sequence
                    ax = sequence_train(imgs, t_steps, t_steps_label, False)

                    # compute step
                    outp = model.forward(ax.to(device))
                    losses = lossfunct(outp, lbls.to(device))

                    # backprop and optimization
                    losses.backward() 
                    optimizer.step()

                    # monitor training 
                    if track_with_neptune:
                        run['train/loss'].log(losses.item())
                        if (current_tempDynamics == 'add_supp') | (current_tempDynamics == 'int_supp_mult'):
                            run['param/alpha'].log(model.sconv1.alpha)
                            run['param/beta'].log(model.sconv1.beta)
                        elif (current_tempDynamics == 'div_norm') | (current_tempDynamics == 'div_norm_add'):
                            run['param/alpha'].log(model.sconv1.alpha)
                            run['param/sigma'].log(model.sconv1.sigma)
                            run['param/K'].log(model.sconv1.K)
                        elif (current_tempDynamics == 'lat_recurrence') | (current_tempDynamics == 'lat_recurrence_mult'):
                            run['param/lweight'].log(torch.mean(model.sconv1.weight.grad))
                            run['param/lbias'].log(torch.mean(model.sconv1.bias.grad))

                    # save losses and print progress
                    if (a+1) % 50 == 0:
                        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                            .format(epoch+1, numepchs, a+1, len(ldrTrain), losses.item()))
            
            # save model
            torch.save(model.state_dict(), root + '/models/weights/' + current_tempDynamics + '_' + str(iInit+1))

            # inference
            model.eval()
            accu_current = torch.zeros((2, len(ldrTest)))

            # loop and retrieve accuracies
            for a, (imgs, lbls) in enumerate(ldrTest):

                # create input sequence
                ax = sequence_train(imgs, t_steps, t_steps_label, False)
                
                # validate
                testoutp = model.forward(ax.to(device))
                predicy = torch.argmax(testoutp, dim=1).to('cpu')

                # compute accuracy for same and different noise
                accu_current[0, a] = (predicy[:int(imgs.shape[0]/2)] == lbls[:int(imgs.shape[0]/2)]).sum().item() / float(lbls.size(0)/imgs.shape[0]/2)
                accu_current[1, a] = (predicy[int(imgs.shape[0]/2):] == lbls[int(imgs.shape[0]/2):]).sum().item() / float(lbls.size(0)/imgs.shape[0]/2)

                # monitoring
                if track_with_neptune:
                    run['test/accu_same'].log(accu_current[0, a])
                    run['test/accu_different'].log(accu_current[1, a])

            # close run
            if track_with_neptune:
                run.stop()

            # save and print accuracy
            accu[0, iInit] = torch.mean(accu_current[0])
            accu[1, iInit] = torch.mean(accu_current[1])
            print('Accuracy ' + current_tempDynamics + ': ', torch.mean(accu_current, 1))

        # save accuracies
        np.save(root + 'models/performance/' + current_tempDynamics, accu)


if __name__ == '__main__':
    main()
