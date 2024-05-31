
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module_div_norm import module_div_norm
from models.module_add_supp import module_add_supp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class cnn_feedforward(nn.Module):
    def __init__(self, t_steps=None, tempDynamics=None):
        super(cnn_feedforward, self).__init__()

        # training variables
        self.t_steps = t_steps
        self.actvs = {}

        # temporal dynamics
        self.tempDynamics = tempDynamics

        # activation functions, pooling and dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.pool = nn.MaxPool2d(2, 2)

        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # torch.nn.init.xavier_normal_(self.conv1.weight)
        # torch.nn.init.zeros_(self.conv1.bias)

        # conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        # torch.nn.init.zeros_(self.conv2.bias)  

        # conv3
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        # torch.nn.init.xavier_normal_(self.conv3.weight)
        # torch.nn.init.zeros_(self.conv3.bias)   

        # fc 1
        # self.fc1 = nn.Linear(in_features=10368, out_features=1024)
        self.fc1 = nn.Linear(in_features=128, out_features=1024)
        # torch.nn.init.xavier_normal_(self.fc1.weight)
        # torch.nn.init.zeros_(self.fc1.bias)    

        # decoder
        # self.decoder = nn.Linear(in_features=1024*self.t_steps, out_features=10)
        self.decoder = nn.Linear(in_features=1024, out_features=10)         # only saves the output from the last timestep to train
        # torch.nn.init.xavier_normal_(self.decoder.weight)
        # torch.nn.init.zeros_(self.decoder.bias)

    def initialize_tempDynamics(self):
        ''' Initiates the modules that apply temporal adaptation based on previous model inputs. '''

        if (self.tempDynamics == 'add_supp') | (self.tempDynamics == 'int_supp_mult'):
            self.sconv1 = module_add_supp()
            self.sconv2 = module_add_supp()
            self.sconv3 = module_add_supp()
        elif (self.tempDynamics == 'div_norm') | (self.tempDynamics == 'div_norm_add'):
            self.sconv1 = module_div_norm()
            self.sconv2 = module_div_norm()
            self.sconv3 = module_div_norm()
        elif (self.tempDynamics == 'lat_recurrence') | (self.tempDynamics == 'lat_recurrence_mult'):
            self.sconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            self.sconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            self.sconv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            
    def apply_adaptation(self, t, x, layer_idx):
        ''' Applies temporal adaptation '''

        # temporal adaptation
        if (self.tempDynamics == 'add_supp'):
            self.g[layer_idx][t+1], feedback = self.sconv1(self.actvs[layer_idx][t], self.g[layer_idx][t])
            x = torch.subtract(x, feedback)
        elif (self.tempDynamics == 'div_norm'):
            feedback = self.sconv1.forward(self.g[layer_idx][t])
            x = torch.multiply(x, feedback)
            self.g[layer_idx][t+1] = self.sconv1.update_g(self.g[layer_idx][t], self.actvs[layer_idx][t])
        elif (self.tempDynamics == 'lat_recurrence'):
            x_l = self.sconv1(self.actvs[layer_idx][t])
            x = torch.add(x, x_l)
        elif (self.tempDynamics == 'lat_recurrence_mult'):
            x_l = self.sconv1(self.actvs[layer_idx][t])
            x = torch.multiply(x, x_l)
        
        # activation function
        x = self.relu(x)  

        return x

    def forward(self, input):

        """ Feedforward sweep. 
        
        Activations are saved in nestled dictionairies: {0: {}, 1: {}, 2: {}, 3: {}, 4: {}},
        where the number indicates the layer
        
        """

        # initiate activations
        actvsc1 = {}
        actvsc2 = {}
        actvsc3 = {}

        self.actvs = {}
        self.actvs[0] = actvsc1
        self.actvs[1] = actvsc2
        self.actvs[2] = actvsc3

        # initiate feedback states (idle with no temporal dynamics and lateral recurrence)
        g1 = {}
        g2 = {}
        g3 = {}
        
        self.g = {}
        self.g[0] = g1
        self.g[1] = g2
        self.g[2] = g3

        # conv1
        x = self.conv1(input[:, 0, :, :, :])
        x = self.relu(x)
        self.g[0][0] = torch.zeros(x.shape).to(device)
        self.actvs[0][0] = x
        x = self.pool(x)
        
        # conv2
        x = self.conv2(x)
        x = self.relu(x)
        self.g[1][0] = torch.zeros(x.shape).to(device)
        self.actvs[1][0] = x
        x = self.pool(x)

        # conv3
        x = self.conv3(x)
        x = self.relu(x)
        self.g[2][0] = torch.zeros(x.shape).to(device)
        self.actvs[2][0] = x
        
        # dropout
        x = self.dropout(x)

        # flatten output
        x = x.view(x.size(0), -1)

        # fc1
        x = self.fc1(x)

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # conv1
                x = self.conv1(input[:, t+1, :, :, :])
                x = self.apply_adaptation(t, x, 0)
            
                # activation function
                x = self.relu(x)  

                self.actvs[0][t+1] = x 
                x = self.pool(x)
                
                # conv2
                x = self.conv2(x)
                x = self.apply_adaptation(t, x, 1)
            
                # activation function
                x = self.relu(x)  

                self.actvs[1][t+1] = x
                x = self.pool(x)

                # conv3
                x = self.conv3(x)
                x = self.apply_adaptation(t, x, 2)
            
                # activation function
                x = self.relu(x)  

                self.actvs[2][t+1] = x
                x = self.dropout(x) 

                # fc1
                x = x.view(x.size(0), -1)
                x = self.fc1(x)

        # compute output
        outp = self.decoder(x)

        return outp
