import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F

class module_add_supp(nn.Module):
    def __init__(self):
        super().__init__()

        # parameters
        self.alpha  = nn.Parameter(torch.rand(1), requires_grad=True)
        self.beta   = nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x_previous, g_previous):

        # update feedback signal
        g =  torch.add(torch.mul(g_previous, self.alpha), torch.mul(x_previous, 1 - self.alpha))
        
        # compute feedback for current timestep (using g_previous)
        feedback = torch.mul(self.beta, g)

        return g, feedback


####################  plot timecourse for example unit

# # create input
# batchsiz = 3
# height = 2
# width = 2
# channels = 3
# t = torch.arange(0.001, 10, 0.001)
# t_steps = len(t)

# r = torch.zeros(t_steps, batchsiz, channels, height, width)
# s = torch.zeros(t_steps, batchsiz, channels, height, width)

# x = torch.zeros(t_steps, batchsiz, channels, height, width)
# x[201:1100, :, :, :] = 1
# x[1801:1900, :, :, :] = 1

# x[3501:4000, :, :, :] = 1
# x[4301:4800, :, :, :] = 1

# x[6501:7000, :, :, :] = 0.05
# x[7501:8000, :, :, :] = 0.2
# x[8501:9000, :, :, :] = 0.5

# # initiate figure
# fig = plt.figure(figsize=(10, 2))
# plt.title('Vinken et al. (2019)')
# plt.ylabel('Activations (a.u.)')
# plt.xlabel('Timesteps')

# # initiate network
# module = module_div_norm_full = module_exp_decay(height=height, 
#                                                         width=width, 
#                                                         channels=channels, 
#                                                         alpha_init=torch.Tensor([0.99]), 
#                                                         train_alpha=True, 
#                                                         beta_init=torch.Tensor([0.7]), 
#                                                         train_beta=True)

# # compute
# for tt in range(1, t_steps):

#     # compute suppression state
#     s[tt, :, :, :, :], s_beta_updt = module.forward(r[tt-1, :, :, :, :], s[tt-1, :, :, :, :])

#     # compute response
#     r[tt, :, :, :, :] = torch.relu(torch.subtract(x[tt, :, :, :, :], s_beta_updt))

# # plot
# # plt.plot(t, x[:, 0, 0, 0, 0].detach().numpy()/max(x[:, 0, 0, 0, 0].detach().numpy()), color='grey', label='stimulus', lw=0.5)
# # plt.plot(t, s[:, 0, 0, 0, 0].detach().numpy()/max(s[:, 0, 0, 0, 0].detach().numpy()), color='crimson', alpha=0.5, linestyle='--', label='suppression')
# # plt.plot(t, r[:, 0, 0, 0, 0].detach().numpy()/max(r[:, 0, 0, 0, 0].detach().numpy()), color='dodgerblue', label='response')
# plt.plot(t, x[:, 0, 0, 0, 0].detach().numpy(), color='grey', label='stimulus', lw=0.5)
# plt.plot(t, s[:, 0, 0, 0, 0].detach().numpy(), color='crimson', alpha=0.5, linestyle='--', label='suppression')
# plt.plot(t, r[:, 0, 0, 0, 0].detach().numpy(), color='dodgerblue', label='response')
# plt.legend(fontsize=8, bbox_to_anchor=(1.16, 1))
# plt.tight_layout()
# plt.savefig('visualizations/toy_Vinken2019')
# plt.show()