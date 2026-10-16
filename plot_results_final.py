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
analytic_path = "analytic_data/"

# load and parse analytic data
Adata = np.zeros([71, 45, 45, 20])

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

data[:, 8, :] = dpred[:, 10, :] # Tzx -> -Tzy
data[:, 9, :] = dpred[:, 11, :]

data[:, 10, :] = dpred[:, 8, :] # Tzy -> -Tzx
data[:, 11, :] = dpred[:, 9, :]

data[:, 12, :] = dpred[:, 15, :] # rhoxx -> rhoyy
data[:, 13, :] = dpred[:, 14, :] # rhoxy -> rhoyx
data[:, 14, :] = dpred[:, 13, :] # rhoyx -> rhoxy
data[:, 15, :] = dpred[:, 12, :] # rhoyy -> rhoxx

data[:, 16, :] = dpred[:, 19, :] # phasexx -> phaseyy
data[:, 17, :] = dpred[:, 18, :] # phasexy -> phaseyx
data[:, 18, :] = dpred[:, 17, :] # phaseyx -> phasexy
data[:, 19, :] = dpred[:, 16, :] # phaseyy -> phasexx

n_profiles = 15        # number of profiles (lines)
n_per_profile = 15     # receivers per profile
spacing = 250.0        # m, both along-profile and between profiles

x_simpeg = (np.arange(n_per_profile) - (n_per_profile - 1) / 2) * spacing

x_analytic = [
        -5000, -2000, -1500, -1000, -750, -500, -400, -300, -250, -200,
        -150, -125, -100, -90, -80, -70, -60, -50, -40, -30,
        -20, -10, 0, 10, 20, 30, 40, 50, 60, 70,
        80, 90, 100, 125, 150, 200, 250, 300, 400, 500,
        750, 1000, 1500, 2000, 5000
    ] # cut along y=0

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
            ax.plot(x_simpeg, data[j, i, 105:120], '.-', label="Simulated")
            ax.plot(x_analytic, Adata[j, :, 7, i], '.-', label="Analytic")
            ax.set_ylabel('Real Impedance (Ω)')
        else: # Imag Impedance plotting
            ax.plot(x_simpeg, data[j, i, 105:120], '.-', label="Simulated")
            ax.plot(x_analytic, Adata[j, :, 7, i], '.-', label="Analytic")
            ax.set_ylabel('Imag Impedance (Ω)')

        ax.set_title(label)
        ax.set_xlabel('Easting (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')

    plt.suptitle(f'Impedance along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/compare{freqs[j]}Hz.png")



# ======================
# App rho and phase for individual freqs
# ======================

labels = ['App Resistivity xy', 'Phase xy', 'App Resistivity yx', 'Phase yx']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()

    axes[0].plot(x_simpeg, np.abs(data[j, 2, 105:120] + data[j, 3, 105:120]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Calculated")
    axes[0].plot(x_simpeg, data[j, 13, 105:120], 'o-', label="Simulated")
    axes[0].plot(x_analytic, np.abs(Adata[j, :, 7, 2] + Adata[j, :, 7, 3]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Analytic")
    axes[0].set_title(labels[0])

    axes[1].plot(x_simpeg, np.angle(data[j, 2, 105:120] + data[j, 3, 105:120]*1j, deg=True), '.-', label="Calculated")
    axes[1].plot(x_simpeg, data[j, 17, 105:120], 'o-', label="Simulated")
    axes[1].plot(x_analytic, np.angle(Adata[j, :, 7, 2] + Adata[j, :, 7, 3]*1j, deg=True), '.-', label="Analytic")
    axes[1].set_title(labels[1])

    axes[2].plot(x_simpeg, np.abs(data[j, 4, 105:120] + data[j, 5, 105:120]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Calculated")
    axes[2].plot(x_simpeg, data[j, 14, 105:120], 'o-', label="Simulated")
    axes[2].plot(x_analytic, np.abs(Adata[j, :, 7, 4] + Adata[j, :, 7, 5]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Analytic")
    axes[2].set_title(labels[2])

    axes[3].plot(x_simpeg, np.angle(data[j, 4, 105:120] + data[j, 5, 105:120]*1j, deg=True), '.-', label="Calculated")
    axes[3].plot(x_simpeg, data[j, 18, 105:120], 'o-', label="Simulated")
    axes[3].plot(x_analytic, np.angle(Adata[j, :, 7, 4] + Adata[j, :, 7, 5]*1j, deg=True), '.-', label="Analytic")
    axes[3].set_title(labels[3])

    for ax in axes:
        ax.set_xlabel('Easting (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')
        

    plt.suptitle(f'Apparent Resisitivity and Phase along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/compareRho{freqs[j]}Hz.png")

