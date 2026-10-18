# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import scipy.constants
from simpeg import utils

mu0 = scipy.constants.mu_0

run_num = "013"
data_path = f"simPEG_data/{run_num}/"
analytic_path = "analytic_data/pub_analytic_data/"

# load and parse analytic data
Adata = np.zeros([24, 15, 15, 20])

appresA = np.load(analytic_path+'appres.npy')
impA = np.load(analytic_path+'imp.npy')
phaseA = np.load(analytic_path+'phase.npy')

Adata[:, :, :, 0] = impA[:, :, :, 0, 0].real # impedence_xx
Adata[:, :, :, 1] = impA[:, :, :, 0, 0].imag # impedence_xx
Adata[:, :, :, 2] = impA[:, :, :, 0, 1].real # impedance_xy
Adata[:, :, :, 3] = impA[:, :, :, 0, 1].imag # impedance_xy
Adata[:, :, :, 4] = impA[:, :, :, 1, 0].real # impedence_yx       
Adata[:, :, :, 5] = impA[:, :, :, 1, 0].imag # impedence_yx         
Adata[:, :, :, 6] = impA[:, :, :, 1, 1].real # impedance_yy
Adata[:, :, :, 7] = impA[:, :, :, 1, 1].imag # impedance_yy

Adata[:, :, :, 8] = impA[:, :, :, 2, 0].real # tipper_zx       
Adata[:, :, :, 9] = impA[:, :, :, 2, 0].imag # tipper_zx          
Adata[:, :, :, 10] = impA[:, :, :, 2, 1].real # tipper_zy  
Adata[:, :, :, 11] = impA[:, :, :, 2, 1].imag # tipper_zy 

Adata[:, :, :, 12] = appresA[:, :, :, 0, 0] # rho_xx
Adata[:, :, :, 13] = appresA[:, :, :, 0, 1] # rho_xy
Adata[:, :, :, 14] = appresA[:, :, :, 1, 0] # rho_yx
Adata[:, :, :, 15] = appresA[:, :, :, 1, 1] # rho_yy
Adata[:, :, :, 16] = phaseA[:, :, :, 0, 0] # phase_xx
Adata[:, :, :, 17] = phaseA[:, :, :, 0, 1] # phase_xy
Adata[:, :, :, 18] = phaseA[:, :, :, 1, 0] + 180 # phase_yx
Adata[:, :, :, 19] = phaseA[:, :, :, 1, 1] # phase_yy


# load and parse simPEG data
dpred = np.load(data_path+'dpredVA.npy')
freqs = np.load(data_path+'freqsVA.npy')
rx_locs = np.load(data_path+'rx_locsVA.npy')

dpred = dpred.reshape(len(freqs), 20, len(rx_locs))
data = np.zeros_like(dpred)

data[:, 0, :] = dpred[:, 6, :] # Zxx -> Zyy
data[:, 1, :] = dpred[:, 7, :]

data[:, 2, :] = dpred[:, 4, :] # Zxy -> Zyx
data[:, 3, :] = dpred[:, 5, :]

data[:, 4, :] = dpred[:, 2, :] # Zyx -> Zxy
data[:, 5, :] = dpred[:, 3, :]

data[:, 6, :] = dpred[:, 0, :] # Zyy -> Zxx
data[:, 7, :] = dpred[:, 1, :]

data[:, 8, :] = -dpred[:, 10, :] # Tzx -> -Tzy
data[:, 9, :] = -dpred[:, 11, :]

data[:, 10, :] = -dpred[:, 8, :] # Tzy -> -Tzx
data[:, 11, :] = -dpred[:, 9, :]

data[:, 12, :] = dpred[:, 15, :] # rhoxx -> rhoyy
data[:, 13, :] = dpred[:, 14, :] # rhoxy -> rhoyx
data[:, 14, :] = dpred[:, 13, :] # rhoyx -> rhoxy
data[:, 15, :] = dpred[:, 12, :] # rhoyy -> rhoxx

data[:, 16, :] = dpred[:, 19, :] # phasexx -> phaseyy
data[:, 17, :] = dpred[:, 18, :] # phasexy -> phaseyx
data[:, 18, :] = dpred[:, 17, :] # phaseyx -> phasexy
data[:, 19, :] = dpred[:, 16, :] # phaseyy -> phasexx

data = data.reshape(len(freqs), 20, 15, 15) # reshape for xy rx locs
data = data.transpose(0, 1, 3, 2)   # rotate coordinates into z down convention

n_profiles = 15        # number of profiles (lines)
n_per_profile = 15     # receivers per profile
spacing = 250.0        # m, both along-profile and between profiles

x = (np.arange(n_per_profile) - (n_per_profile - 1) / 2) * spacing

plot_freqs_ind = np.arange(len(freqs))


# ======================
# Impedances for individual freqs
# ======================

labels = ['Real Impedance xx', 'Imag Impedance xx', 'Real Impedance xy', 'Imag Impedance xy', 'Real Impedance yx', 'Imag Impedance yx', 'Real Impedance yy', 'Imag Impedance yy']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 4, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        if i % 2 == 0: # Real Impedance plotting
            ax.plot(x, data[j, i, :, 7], '.-', label="Simulated")
            ax.plot(x, Adata[j, :, 7, i], '.-', label="Analytic")
            ax.set_ylabel('Real Impedance (Ω)')
        else: # Imag Impedance plotting
            ax.plot(x, data[j, i, :, 7], '.-', label="Simulated")
            ax.plot(x, Adata[j, :, 7, i], '.-', label="Analytic")
            ax.set_ylabel('Imag Impedance (Ω)')

        ax.set_title(label)
        ax.set_xlabel('Northing (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')

    plt.suptitle(f'Impedance along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/compare{freqs[j]}Hz.png")
    plt.close()



# ======================
# App rho and phase for individual freqs
# ======================

labels = ['App Resistivity xy', 'Phase xy', 'App Resistivity yx', 'Phase yx']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()

    axes[0].plot(x, np.abs(data[j, 2, :, 7] + data[j, 3, :, 7]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Calculated")
    axes[0].plot(x, data[j, 13, :, 7], 'o-', label="Simulated")
    axes[0].plot(x, np.abs(Adata[j, :, 7, 2] + Adata[j, :, 7, 3]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Analytic")
    axes[0].set_title(labels[0])

    axes[1].plot(x, np.angle(data[j, 2, :, 7] + data[j, 3, :, 7]*1j, deg=True), '.-', label="Calculated")
    axes[1].plot(x, data[j, 17, :, 7], 'o-', label="Simulated")
    axes[1].plot(x, np.angle(Adata[j, :, 7, 2] + Adata[j, :, 7, 3]*1j, deg=True), '.-', label="Analytic")
    axes[1].set_title(labels[1])

    axes[2].plot(x, np.abs(data[j, 4, :, 7] + data[j, 5, :, 7]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Calculated")
    axes[2].plot(x, data[j, 14, :, 7], 'o-', label="Simulated")
    axes[2].plot(x, np.abs(Adata[j, :, 7, 4] + Adata[j, :, 7, 5]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Analytic")
    axes[2].set_title(labels[2])

    axes[3].plot(x, np.angle(data[j, 4, :, 7] + data[j, 5, :, 7]*1j, deg=True), '.-', label="Calculated")
    axes[3].plot(x, data[j, 18, :, 7], 'o-', label="Simulated")
    axes[3].plot(x, np.angle(Adata[j, :, 7, 4] + Adata[j, :, 7, 5]*1j, deg=True), '.-', label="Analytic")
    axes[3].set_title(labels[3])

    for ax in axes:
        ax.set_xlabel('Northing (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')
        

    plt.suptitle(f'Apparent Resisitivity and Phase along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/compareRho{freqs[j]}Hz.png")
    plt.close()


# ======================
# Impedance percent residuals for individual freqs
# ======================

labels = ['Real Impedance xy', 'Imag Impedance xy', 'Real Impedance yx', 'Imag Impedance yx']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        if i % 2 == 0: # Real Impedance plotting
            ax.plot(x, 100 * (data[j, i+2, :, 7] - Adata[j, :, 7, i+2]) / Adata[j, :, 7, i+2], '.-')
            ax.set_ylabel('Percent Difference')
        else: # Imag Impedance plotting
            ax.plot(x, 100 * (data[j, i+2, :, 7] - Adata[j, :, 7, i+2]) / Adata[j, :, 7, i+2], '.-')
            ax.set_ylabel('Percent Difference')

        ax.set_title(label)
        ax.set_xlabel('Northing (m)')
        ax.grid(True, which='both', alpha=0.3)

    plt.suptitle(f'Impedance Percent Residuals along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/residuals{freqs[j]}Hz.png")
    plt.close()