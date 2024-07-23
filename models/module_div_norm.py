import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns

class module_div_norm(nn.Module):
    def __init__(self, K=None, alpha=None, sigma=None):
        super().__init__()

        # prevent zero division
        self.epsilon = 1.e-8

        # parameters
        self.K          = nn.Parameter(torch.Tensor([0.1]), requires_grad=True)
        self.alpha      = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
        self.sigma      = nn.Parameter(torch.Tensor([1]), requires_grad=True)
    
    def update_g(self, g_previous, x_previous):
        """ Update feedback signal """

        g = torch.add(torch.mul(1 - self.alpha, g_previous), torch.mul(self.alpha, x_previous))      # update feedback signal

        return g
    
    def forward(self, g_previous):
        """ x is the current input computed for the linear response, F is the multiplicative feedback
        and G is the feedback signal. """

        # compute feedback for current timestep (using g_previous)
        if torch.max(g_previous) > self.K:                                                  # rescale if feedback signal exceeds maximal attainable response
            g_previous = g_previous/torch.max(g_previous)
            g_previous = g_previous*self.K
        feedback = torch.sqrt(torch.subtract(self.K, g_previous)+self.epsilon)/self.sigma          # multiplicative feedback                                                    # response

        return feedback

# ####################  plot timecourse for example unit