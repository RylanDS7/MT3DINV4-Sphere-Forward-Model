# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

dpred = np.load('dpred.npy')
freqs = np.load('freqs.npy')
rx_locs = np.load('rx_locs.npy')

print(dpred.shape)
print(freqs.shape)
print(rx_locs.shape)

data = dpred.reshape(len(freqs), 4, rx_locs.shape[0])
data[:, 1, :] += 180 # app resistivity quadrant correction
rho_xy = data[:, 0, :]
phase_xy = data[:, 1, :]
rho_yx = data[:, 2, :]
phase_yx = data[:, 3, :]

fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
axes = axes.flatten()

x_cut = rx_locs[:, 1]
freq_idx = 30
labels = ['AppRes xy', 'Phase xy', 'App Res yx', 'Phase yx']

for i, (ax, label) in enumerate(zip(axes, labels)):
    ax.plot(x_cut, data[freq_idx, i, :], 'o-')
    ax.set_title(label)
    ax.set_xlabel('Easting (m)')
    if i % 2 == 0:
        ax.set_ylabel('App Res (Ωm)')
    else:
        ax.set_ylabel('Phase (Degrees)')
    ax.grid(True, which='both', alpha=0.3)

plt.suptitle(f'Apparent Resistivity and Phase along Cut at x=0 for {freqs[freq_idx]}Hz')
plt.tight_layout()
plt.show()