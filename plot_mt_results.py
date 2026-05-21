# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data_path = "simPEG_data/"
analytic_path = "analytic_data/"

# load and parse analytic data
Adata = np.zeros([71, 45, 4]) # not holding Z data right now
for f_ind in np.arange(71):
    try:
        appresA = np.load(analytic_path+'appres'+str(f_ind)+'.npy')
        impA = np.load(analytic_path+'imp'+str(f_ind)+'.npy')
        phaseA = np.load(analytic_path+'phase'+str(f_ind)+'.npy')

        Adata[f_ind, :, 2] = appresA[0, :, 0, 1, 0] # rho_yx in simpeg convention
        Adata[f_ind, :, 3] = phaseA[0, :, 0, 1, 0] + 180 # phase_yx in simpeg convention
        Adata[f_ind, :, 0] = appresA[0, :, 0, 0, 1] # rho_xy in simpeg convention
        Adata[f_ind, :, 1] = phaseA[0, :, 0, 0, 1] # phase_xy in simpeg convention
    except FileNotFoundError:
        continue

# load and parse simPEG data
dpred = np.load(data_path+'dpred.npy')
freqs = np.load(data_path+'freqs.npy')
rx_locs = np.load(data_path+'rx_locs.npy')

data = dpred.reshape(len(freqs), 4, rx_locs.shape[0])
data[:, 1, :] += 180 # app resistivity quadrant correction
rho_xy = data[:, 0, :]
phase_xy = data[:, 1, :]
rho_yx = data[:, 2, :]
phase_yx = data[:, 3, :]



plot_freqs_ind = [0, 10, 20, 30, 40, 50]

x_cut = rx_locs[0:45, 0]
labels = ['App Res xy', 'Phase xy', 'App Res yx', 'Phase yx']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    print("=====================")
    print(f"Frequency {freqs[j]} Hz:")

    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(x_cut, data[j, i, :], '.-', label="Simulated")
        ax.plot(x_cut, Adata[j, :, i], '.-', label="Analytic")
        ax.set_title(label)
        ax.set_xlabel('Easting (m)')
        if i % 2 == 0:
            ax.set_ylabel('App Res (Ωm)')
            ax.plot(x_cut, data[j, i, :]-Adata[j, :, i], '.-', label="Residual: Sim - Analytic")
            print(f"{label}:     Max residual: {max(np.abs(data[j, i, :]-Adata[j, :, i])) :.3}       Squared Mean: {np.mean(np.square(np.abs(data[j, i, :]-Adata[j, :, i]))) :.3}")
        else:
            ax.set_ylabel('Phase (Degrees)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')

    plt.suptitle(f'Apparent Resistivity and Phase along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.show()