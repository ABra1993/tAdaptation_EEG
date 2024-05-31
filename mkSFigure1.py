import numpy as np
import matplotlib.pyplot as plt

# define root
root                = '/home/amber/OneDrive/code/nAdaptation_EEG_git/'

global C
C = np.linspace(1, 100, 50)

def Naka_Rushton(C, R_max, c50, n, b):

    R = R_max * (C**n/(C**n + c50**n)) + b
    
    return R

def simulate_Naka_Rushton():

    # parameters
    A       = np.linspace(5, 26, 8)
    c50     = np.linspace(20, 52, 8)
    n       = np.linspace(1, 10, 8)
    b       = np.linspace(20, 50, 8)

    param   = [A, c50, n, b]
    fixed   = [10, 40, 5, 0]

    # initiate figure
    row             = 2
    column          = 2
    _, axs        = plt.subplots(row, column, figsize=(20,20))
    titles              = ['A', r'c$_{50}$', 'q', 'B']

    cmap = plt.get_cmap('plasma')
    lw = 3

    fontsize_label      = 25
    fontsize_legend     = 20
    fontsize_tick       = 25
    fontsize_title      = 30

    # compute and plot
    count = 0
    for i in range(row):
        for j in range(column):

            # color map
            colors = cmap(cmap(np.linspace(0.1, 1, 1+len(param[count]))))

            for v in range(len(param[count])):

                if count == 0:

                    R = Naka_Rushton(C, param[count][v], fixed[1], fixed[2], fixed[3])

                elif count == 1:

                    R = Naka_Rushton(C, fixed[0], param[count][v], fixed[2], fixed[3])

                elif count == 2:

                    R = Naka_Rushton(C, fixed[0], fixed[1], param[count][v], fixed[3])

                elif count == 3:

                    R = Naka_Rushton(C, fixed[0], fixed[1], fixed[2], param[count][v])

                # plot
                axs[i, j].plot(C, R, color=colors[v][0, :], label=int(np.round(param[count][v], 2)), lw=lw)

            # add legend
            axs[i, j].legend(fontsize=fontsize_legend)
            axs[i, j].spines['right'].set_visible(False)
            axs[i, j].spines['top'].set_visible(False)
            axs[i, j].tick_params(axis='both', which='major', labelsize=fontsize_tick)
            if i == row - 1:
                axs[i, j].set_xlabel('Contrast', fontsize=fontsize_label)
            if j == 0:
                axs[i, j].set_ylabel('Response', fontsize=fontsize_label)
            axs[i, j].set_title('Response modulation by ' + titles[count], fontsize=fontsize_title)

            # increment count
            count+= 1

    # save figure
    plt.savefig(root + 'visualization/SFig1', dpi=300)
    plt.savefig(root + 'visualization/SFig1.svg')


simulate_Naka_Rushton()


