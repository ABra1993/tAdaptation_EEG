import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module_div_norm import module_div_norm
from models.module_add_supp import module_add_supp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class cnn_feedforward(nn.Module):
    def __init__(self):
        super(cnn_feedforward, self).__init__()

        # activation functions, pooling and dropout layers
        self.relu                   = nn.ReLU()
        self.dropout                = nn.Dropout()
        self.pool                   = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten                = nn.Flatten()

        # conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)

        # conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        
        # fully connected
        self.fc1 = nn.Linear(in_features=128, out_features=1024)

    def init_t_steps(self, t_steps=None):

        # set number of timesteps
        self.t_steps = t_steps

        # initiate readout          
        self.decoder = nn.Linear(in_features=1024, out_features=10)
        
        # gpu
        self.decoder.to(device)

    def initialize_tempDynamics(self, tempDynamics):
        ''' Initiates the modules that apply temporal adaptation based on previous model inputs. '''

        # set temporal dynamics
        self.tempDynamics   = tempDynamics

        if (self.tempDynamics == 'add_supp'):
            self.sconv1 = module_add_supp()
            self.sconv2 = module_add_supp()
            self.sconv3 = module_add_supp()
        elif ('div_norm' in self.tempDynamics):
            self.sconv1 = module_div_norm(tempDynamics)
            self.sconv2 = module_div_norm(tempDynamics)
            self.sconv3 = module_div_norm(tempDynamics)
        elif (self.tempDynamics == 'l_recurrence_A') | (self.tempDynamics == 'l_recurrence_M'):
            self.sconv1 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            self.sconv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            self.sconv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=1)
            
    def apply_adaptation(self, t, x, layer_idx):
        ''' Applies temporal adaptation '''

        if layer_idx == 0:

                # temporal adaptation
                if (self.tempDynamics == 'add_supp'):
                    self.g[layer_idx][t+1], feedback = self.sconv1(self.actvs[layer_idx][t], self.g[layer_idx][t])
                    x = torch.subtract(x, feedback)

                elif ('div_norm' in self.tempDynamics):
                    self.g[layer_idx][t+1], feedback = self.sconv1(self.actvs[layer_idx][t], self.g[layer_idx][t])
                    x = torch.multiply(x, feedback)

                elif (self.tempDynamics == 'l_recurrence_A'):
                    x_l = self.sconv1(self.actvs[layer_idx][t])
                    x = torch.add(x, x_l)

        if layer_idx == 1:

                # temporal adaptation
                if (self.tempDynamics == 'add_supp'):
                    self.g[layer_idx][t+1], feedback = self.sconv2(self.actvs[layer_idx][t], self.g[layer_idx][t])
                    x = torch.subtract(x, feedback)

                elif ('div_norm' in self.tempDynamics):
                    self.g[layer_idx][t+1], feedback = self.sconv2(self.actvs[layer_idx][t], self.g[layer_idx][t])
                    x = torch.multiply(x, feedback)

                elif (self.tempDynamics == 'l_recurrence_A'):
                    x_l = self.sconv2(self.actvs[layer_idx][t])
                    x = torch.add(x, x_l)

        if layer_idx == 2:

                # temporal adaptation
                if (self.tempDynamics == 'add_supp'):
                    self.g[layer_idx][t+1], feedback = self.sconv3(self.actvs[layer_idx][t], self.g[layer_idx][t])
                    x = torch.subtract(x, feedback)

                elif ('div_norm' in self.tempDynamics):
                    self.g[layer_idx][t+1], feedback = self.sconv3(self.actvs[layer_idx][t], self.g[layer_idx][t])
                    x = torch.multiply(x, feedback)

                elif (self.tempDynamics == 'l_recurrence_A'):
                    
                    x_l = self.sconv3(self.actvs[layer_idx][t])
                    x = torch.add(x, x_l)
                
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
        fc1     = {}

        self.actvs = {}
        self.actvs[0] = actvsc1
        self.actvs[1] = actvsc2
        self.actvs[2] = actvsc3
        self.actvs[3] = fc1

        # RECURREN-dependent ------ initiate feedback states (idle with NONE and L-, T-, LT-recurrence)
        g1 = {}
        g2 = {}
        g3 = {}
        
        self.g = {}
        self.g[0] = g1
        self.g[1] = g2
        self.g[2] = g3
        
        # TASK-dependent ------ idle with CORE, REPEATED_NOISE
        outp = []

        # ------- CONV1
        x = self.conv1(input[0].to(device))
        x = self.relu(x)

        self.g[0][0] = torch.zeros(x.shape).to(device)
        self.actvs[0][0] = x

        # pooling
        x = self.pool(x)
        
        # ------- CONV2
        x = self.conv2(x)
        x = self.relu(x)

        self.g[1][0] = torch.zeros(x.shape).to(device)
        self.actvs[1][0] = x

        # pooling
        x = self.pool(x)

        # ------- CONV3
        x = self.conv3(x)
        x = self.relu(x)

        self.g[2][0] = torch.zeros(x.shape).to(device)
        self.actvs[2][0] = x
        
        # dropout
        x = self.dropout(x)

        # flatten output
        x = self.flatten(x)

        # ------- FC
        x = self.fc1(x)
        self.actvs[3][0] = x

        if self.t_steps > 0:
            for t in range(self.t_steps-1):

                # ------- CONV1
                x = self.conv1(input[t+1].to(device))
                x = self.apply_adaptation(t=t, x=x, layer_idx=0)
                x = self.relu(x)

                self.actvs[0][t+1] = x 

                # pooling
                x = self.pool(x)
                
                # ------- CONV2
                x = self.conv2(x)
                x = self.apply_adaptation(t=t, x=x, layer_idx=1)
                x = self.relu(x)

                self.actvs[1][t+1] = x

                # pooling
                x = self.pool(x)

                # ------- CONV3
                x = self.conv3(x)
                x = self.apply_adaptation(t=t, x=x, layer_idx=2)
                x = self.relu(x)

                self.actvs[2][t+1] = x

                # dropout
                x = self.dropout(x) 

                # ------- FC
                x = self.flatten(x)
                x = self.fc1(x)
                self.actvs[3][t+1] = x

        # compute output
        outp = self.decoder(x)

        return outp
