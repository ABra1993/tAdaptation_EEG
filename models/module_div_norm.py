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

        # # parameters
        # self.K          = nn.Parameter(torch.Tensor([K]), requires_grad=True)
        # self.alpha      = nn.Parameter(torch.Tensor([alpha]), requires_grad=True)
        # self.sigma      = nn.Parameter(torch.Tensor([sigma]), requires_grad=True)
    
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
    
# def toy_Heeger1992_variable_K():


#     # set time
#     t = torch.arange(0, 100)
#     t_steps = len(t)

#     # set stimulus
#     x = torch.zeros(t_steps)

#     x[20:80] = 1
#     # x[600:900] = 1

#     # set gain
#     K_values = torch.arange(0.15, 1, 0.1)

#     # visualize
#     fig = plt.figure(figsize=(6, 3))
#     ax = plt.gca()

#     # sns.despine(offset=10)

#     fontsize_label      = 15
#     fontsize_legend     = 10
#     fontsize_title      = 15
#     fontsize_ticks      = 15

#     # set colors
#     cmap = plt.get_cmap('plasma')
#     colors = cmap(torch.linspace(0.1, 1, len(K_values)+1))

#     for iK in range(len(K_values)):

#         # initiat model
#         module = module_div_norm(K=K_values[iK], alpha=0.01, sigma=0.6)

#         # reset dataframes
#         r = torch.zeros(t_steps)
#         G = torch.zeros(t_steps)

#         # compute response
#         for tt in range(1, t_steps):
#             feedback = module.forward(G[tt-1])
#             print(feedback)
#             r[tt] = x[tt]*feedback
#             G[tt] = module.update_g(G[tt-1], r[tt-1])

#         # plot
#         # ax.plot(x[:, 0, 0, 0, 0].detach().numpy(), color=colors[iK+1], alpha=0.5, linestyle='--', lw=0.75)
#         ax.plot(r.detach().numpy(), color=colors[iK+1], label=K_values[iK], lw=3)

#     # adjust axes
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#     ax.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

#     ax.set_yticks([])
#     # ax.set_title('$\it{+}$  $\it{div. norm}$ \n $\it{with}$ $\it{variable}$ $\it{gain}$', fontsize=fontsize_label)
#     ax.set_xticks([])
#     # ax.set_xlabel('Time', fontsize=fontsize_label)
#     # ax.set_ylabel('Activation (a.u.)', fontsize=fontsize_label)

#     # save fig
#     plt.tight_layout()
#     plt.savefig('visualizations/chapter3/contrast_gain/toy_Heeger1992_variable_K.svg')
#     plt.savefig('visualizations/chapter3/contrast_gain/toy_Heeger1992_variable_K')
#     # plt.show()

# toy_Heeger1992_variable_K()