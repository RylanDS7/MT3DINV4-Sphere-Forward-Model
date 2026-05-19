# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data_path = "simPEG_data/"
analytic_path = "analytic_data/"

# load and parse analytic data
appresA = np.load(analytic_path+'appres1.npy')
impA = np.load(analytic_path+'imp1.npy')
phaseA = np.load(analytic_path+'phase1.npy')

Adata = np.ones([appresA.shape[1], 4])
Adata[:, 0] = appresA[0, :, 0, 1, 0]
Adata[:, 1] = phaseA[0, :, 0, 1, 0] + 180
Adata[:, 2] = appresA[0, :, 0, 0, 1]
Adata[:, 3] = phaseA[0, :, 0, 0, 1]


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

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
axes = axes.flatten()

x_cut = rx_locs[0:35, 1]
freq_idx = 0
labels = ['App Res xy', 'Phase xy', 'App Res yx', 'Phase yx']

for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(x_cut, data[freq_idx, i, 0:35], '.-', label="Simulated")
    ax.plot(x_cut, Adata[5:40, i], '.-', label="Analytic")
    ax.set_title(label)
    ax.set_xlabel('Easting (m)')
    if i % 2 == 0:
        ax.set_ylabel('App Res (Ωm)')
    else:
        ax.set_ylabel('Phase (Degrees)')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='lower left')

plt.suptitle(f'Apparent Resistivity and Phase along Cut at x=0 for {freqs[freq_idx]}Hz')
plt.tight_layout()
plt.show()