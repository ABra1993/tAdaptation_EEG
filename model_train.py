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

# # set the seed
# seed = 42
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.backends.cudnn.deterministic = True

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# define root
root                = '/home/amber/Documents/organize_code/nAdaptation_EEG_git/'

def custom_regularization_loss(model, lambda_reg=0):
    reg_loss = 0
    for param in model.parameters():
        # Penalize small weights using 1 / (|w| + epsilon)
        reg_loss += torch.sum(1.0 / (torch.abs(param) + 1e-6))
    return lambda_reg * reg_loss

def main():

    # input sequences
    # sequences = ['ABT', 'AAABT', 'AAAAAABTT', 'AAAAAAAAABTTT', 'AABTT', 'AAABTTT', 'AAAABTTTT', 'AAAAAAAABTTTTTTTT']
    # sequences = ['AAABTTT']
    sequences = ['ABT', 'AAABT', 'AAAAAABTT', 'AAAAAAAAABTTT']
    # sequences = ['AAAAAAAAAAAABTTTT', 'AAAAAAAAAAAAAAABTTTTT', 'AAAAAAAAAAAAAAAAAABTTTTTT']
    sequences =  ['AAAAAAAAAAAAAAAAAAAAABTTTTTTT']

    # track with neptune
    track_with_neptune = False

    # input specification
    dataset                 = 'mnist'

    # hyperparameter specification
    init            = 5
    numepchs        = 5
    batch_size      = 100
    lr              = 0.001

    # temporal dynamics
    # tempDynamics = ['div_norm_scale', 'div_norm_clamp', 'none', 'add_supp', 'l_recurrence_A']
    tempDynamics = ['none', 'add_supp', 'l_recurrence_A', 'div_norm_clamp']
    # tempDynamics = ['div_norm']

    # define contrast values
    contrast_values                 = torch.Tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    contrast_low                    = torch.min(contrast_values)
    contrast_high                   = torch.max(contrast_values)

    # print summary 
    print(30*'--')
    print('Model: '.ljust(25), tempDynamics)
    print('Dataset: '.ljust(25), dataset)
    print(30*'--')
    print('\n')

    # set home directory
    download            = False # can be set to False after download
    root_data           = root + 'models/dataset/'

    if dataset == 'mnist':
        trainData       = datasets.MNIST(root=root_data, download=download, train=True, transform = transforms.ToTensor())
        testData        = datasets.MNIST(root=root_data, download=download, train=False, transform = transforms.ToTensor())
    elif dataset == 'fmnist':
        trainData       = datasets.FashionMNIST(root=root_data, download=download, train=True, transform = transforms.ToTensor())
        testData        = datasets.FashionMNIST(root=root_data, download=download, train=False, transform = transforms.ToTensor())

    # Define data loaders with drop_last=True
    ldrTrain = torch.utils.data.DataLoader(trainData, batch_size=batch_size, shuffle=True, drop_last=True)
    ldrTest = torch.utils.data.DataLoader(testData, batch_size=batch_size, shuffle=False, drop_last=True)

    print('Size dataset during training: ', len(trainData))
    print('Size dataset during inference: ', len(testData))

    # train networks
    for sequence in sequences:

        # retrieve sequence
        t_steps = len(sequence)
        t_steps_label = encode_timesteps(t_steps, sequence)
        print(t_steps_label)

        # training
        for _, current_tempDynamics in enumerate(tempDynamics):

            for iInit in range(init):

                # initiate dataframe to store accuracies
                accu = np.zeros((len(contrast_values), 2)) # 2 = same/different

                # initiate neptune monitoring
                if track_with_neptune:
                    run = neptune.init_run(
                        project="abra1993/adapt-dnn",
                        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5ODkxNGY3NS05NGJlLTQzZDEtOGU5Yy0xMjJlYzI0YzE2YWUifQ==",
                    )  # your credentials

                    run['name']         = current_tempDynamics
                    run['dataset']      = dataset
                    run['numepchs']     = numepchs

                # initiate model
                model = cnn_feedforward()

                # initiate recurrence
                if tempDynamics != 'none':
                    model.initialize_tempDynamics(current_tempDynamics)
                model.to(device)

                # init timesteps
                model.init_t_steps(t_steps)

                # loss function and optimizer
                lossfunct = nn.CrossEntropyLoss()   
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

                # train
                print('\nTraining in progress...')
                model.train()

                for epoch in range(numepchs): # images and labels
                    for a, (imgs, lbls) in enumerate(ldrTrain):

                        # create sample sequence for first batch sample
                        create_sample = False
                        if (a == 0) & (epoch == 0):
                            create_sample = True

                        # create input sequence
                        _, ax = create_sequence(imgs=imgs, 
                                                t_steps=t_steps, 
                                                t_steps_label=t_steps_label, 
                                                create_sample=create_sample, 
                                                contrast=None, 
                                                contrast_low=contrast_low, 
                                                contrast_high=contrast_high,
                                                sequence=sequence,
                                                root=root)
                        create_sample = False

                        # # create input sequence
                        # ax = sequence_train(imgs)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # compute step
                        outp = model.forward(ax)
                        total_loss = lossfunct(outp, lbls.to(device))

                        # backprop and optimization
                        total_loss.backward() 
                        optimizer.step()

                        # monitor training 
                        if track_with_neptune:
                            run['train/loss'].log(total_loss.item())
                            if (current_tempDynamics == 'add_supp') | (current_tempDynamics == 'int_supp_mult'):
                                run['param/alpha'].log(model.sconv1.alpha)
                                run['param/beta'].log(model.sconv1.beta)
                            elif (current_tempDynamics == 'div_norm') | (current_tempDynamics == 'div_norm_add'):
                                run['param/alpha'].log(model.sconv1.alpha)
                                run['param/sigma'].log(model.sconv1.sigma)
                                run['param/K'].log(model.sconv1.K)
                            elif (current_tempDynamics == 'l_recurrence_A') | (current_tempDynamics == 'l_recurrence_M'):
                                run['param/lweight'].log(torch.mean(model.sconv1.weight.grad))
                                run['param/lbias'].log(torch.mean(model.sconv1.bias.grad))

                        # save losses and print progresstmux a
                        if (a+1) % 50 == 0:
                            print ('Sequence: {}, Temp. dynamics: {}, init: [{}/{}] Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                                .format(sequence, current_tempDynamics, iInit+1, init, epoch+1, numepchs, a+1, len(ldrTrain), total_loss.item()))
                
                # save model
                torch.save(model.state_dict(), root + '/models/weights/' + sequence + '/' + current_tempDynamics + '_' + str(iInit+1))

                # testing
                print('\nInference in progress...')
                model.train()

                # inference
                model.eval()
                accu_current = torch.zeros((2, len(ldrTest)))
                
                # loop and retrieve accuracies
                for iC, contrast in enumerate(contrast_values):
                    for a, (imgs, lbls) in enumerate(ldrTest):

                        # if (a != 0) | (iC != 0):
                        #     break

                        # create input sequence
                        _, ax = create_sequence(imgs=imgs, 
                                            t_steps=t_steps, 
                                            t_steps_label=t_steps_label, 
                                            create_sample=False, 
                                            contrast=contrast, 
                                            contrast_low=None, 
                                            contrast_high=None)

                        # # create input sequence
                        # ax = sequence_train(imgs)

                        # compute step
                        outp = model.forward(ax)
                        total_loss = lossfunct(outp, lbls.to(device))

                        # compute
                        testoutp = model.forward(ax)
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
                    accu[iC, 0] = torch.mean(accu_current[0])
                    accu[iC, 1] = torch.mean(accu_current[1])
                    print('Accuracy ' + current_tempDynamics + ': ', torch.mean(accu_current, 1))

                # save accuracies
                # if 'variable_gain' in current_tempDynamics:
                #     np.save(root + 'models/performance_all_contrasts/' + current_tempDynamics + str(c50) + '_' + str(iInit+1), accu)
                # else:
                    
                np.save(root + 'models/performance/' + sequence + '/' + current_tempDynamics + '_' + str(iInit+1), accu)

if __name__ == '__main__':
    main()
