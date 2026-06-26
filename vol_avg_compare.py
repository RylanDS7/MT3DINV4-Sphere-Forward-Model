# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

data_path = f"simPEG_data/vol_avg/"
analytic_path = "analytic_data/"

# load and parse analytic data
Adata = np.zeros([71, 45, 45, 8])

appresA = np.load(analytic_path+'appres.npy')
impA = np.load(analytic_path+'imp.npy')
phaseA = np.load(analytic_path+'phase.npy')

Adata[:, :, :, 4] = -impA[:, :, :, 1, 0].real # impedence_yx in simpeg convention
Adata[:, :, :, 5] = -impA[:, :, :, 1, 0].imag # impedence_yx in simpeg convention
Adata[:, :, :, 6] = appresA[:, :, :, 1, 0] # rho_yx in simpeg convention
Adata[:, :, :, 7] = phaseA[:, :, :, 1, 0] + 180 # phase_yx in simpeg convention
Adata[:, :, :, 0] = impA[:, :, :, 0, 1].real # impedence_xy in simpeg convention       
Adata[:, :, :, 1] = impA[:, :, :, 0, 1].imag # impedence_xy in simpeg convention         
Adata[:, :, :, 2] = appresA[:, :, :, 0, 1] # rho_xy in simpeg convention
Adata[:, :, :, 3] = phaseA[:, :, :, 0, 1] # phase_xy in simpeg convention

# load and parse simPEG data
dpred = np.load(data_path+'dpred.npy')
freqs = np.load(data_path+'freqs.npy')
rx_locs = np.load(data_path+'rx_locs.npy')

data = dpred
data[:, 3, :] += 180 # app resistivity phase quadrant correction
data[:, 0, :] = -data[:, 0, :]
data[:, 1, :] = -data[:, 1, :]

dataVA = np.load(data_path+'dpredVA.npy')
dataVA[:, 3, :] += 180 # app resistivity phase quadrant correction
dataVA[:, 0, :] = -dataVA[:, 0, :]
dataVA[:, 1, :] = -dataVA[:, 1, :]

plot_freqs_ind = [0, 10, 20, 30, 40, 50] # plot 1 freq per decade

x_cut = rx_locs[22::45, 0] # cut along y=0


# ======================
# Impedance residuals
# ======================

for i in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    axes[0].plot(x_cut, 100*(data[i, 0, 22::45]-Adata[i, :, 22, 0])/Adata[i, :, 22, 0], '.-', label=f"Non Vol Avg")
    axes[1].plot(x_cut, 100*(data[i, 1, 22::45]-Adata[i, :, 22, 1])/Adata[i, :, 22, 1], '.-', label=f"Non Vol Avg")
    axes[2].plot(x_cut, 100*(data[i, 4, 22::45]-Adata[i, :, 22, 4])/Adata[i, :, 22, 4], '.-', label=f"Non Vol Avg")
    axes[3].plot(x_cut, 100*(data[i, 5, 22::45]-Adata[i, :, 22, 5])/Adata[i, :, 22, 5], '.-', label=f"Non Vol Avg")

    axes[0].plot(x_cut, 100*(dataVA[i, 0, 22::45]-Adata[i, :, 22, 0])/Adata[i, :, 22, 0], '.-', label=f"Vol Avg")
    axes[1].plot(x_cut, 100*(dataVA[i, 1, 22::45]-Adata[i, :, 22, 1])/Adata[i, :, 22, 1], '.-', label=f"Vol Avg")
    axes[2].plot(x_cut, 100*(dataVA[i, 4, 22::45]-Adata[i, :, 22, 4])/Adata[i, :, 22, 4], '.-', label=f"Vol Avg")
    axes[3].plot(x_cut, 100*(dataVA[i, 5, 22::45]-Adata[i, :, 22, 5])/Adata[i, :, 22, 5], '.-', label=f"Vol Avg")

    axes[0].set_title("Real Impedance xy")
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Percent')
    axes[0].legend()
    axes[1].set_title("Imag Impedance xy")
    axes[1].set_xlabel('Easting (m)')
    axes[1].set_ylabel('Percent')
    axes[1].legend()
    axes[2].set_title("Real Impedance yx")
    axes[2].set_xlabel('Easting (m)')
    axes[2].set_ylabel('Percent')
    axes[2].legend()
    axes[3].set_title("Imag Impedance yx")
    axes[3].set_xlabel('Easting (m)')
    axes[3].set_ylabel('Percent')
    axes[3].legend()
    plt.suptitle(f"Impedance Residuals Volume Averaged Comparison, {freqs[i]}Hz")
    plt.savefig(f"figure_out/vol_avg/zResiduals{freqs[i]}Hz.png")


# ======================
# Impedances
# ======================

for i in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    axes[0].plot(x_cut, data[i, 0, 22::45], '.-', label=f"Non Vol Avg")
    axes[1].plot(x_cut, data[i, 1, 22::45], '.-', label=f"Non Vol Avg")
    axes[2].plot(x_cut, data[i, 4, 22::45], '.-', label=f"Non Vol Avg")
    axes[3].plot(x_cut, data[i, 5, 22::45], '.-', label=f"Non Vol Avg")

    axes[0].plot(x_cut, dataVA[i, 0, 22::45], '.-', label=f"Vol Avg")
    axes[1].plot(x_cut, dataVA[i, 1, 22::45], '.-', label=f"Vol Avg")
    axes[2].plot(x_cut, dataVA[i, 4, 22::45], '.-', label=f"Vol Avg")
    axes[3].plot(x_cut, dataVA[i, 5, 22::45], '.-', label=f"Vol Avg")

    axes[0].plot(x_cut, Adata[i, :, 22, 0], '.-', label=f"Analytic")
    axes[1].plot(x_cut, Adata[i, :, 22, 1], '.-', label=f"Analytic")
    axes[2].plot(x_cut, Adata[i, :, 22, 4], '.-', label=f"Analytic")
    axes[3].plot(x_cut, Adata[i, :, 22, 5], '.-', label=f"Analytic")

    axes[0].set_title("Real Impedance xy")
    axes[0].set_xlabel('Easting (m)')
    axes[0].set_ylabel('Impedance (Ω)')
    axes[0].legend()
    axes[1].set_title("Imag Impedance xy")
    axes[1].set_xlabel('Easting (m)')
    axes[1].set_ylabel('Impedance (Ω)')
    axes[1].legend()
    axes[2].set_title("Real Impedance yx")
    axes[2].set_xlabel('Easting (m)')
    axes[2].set_ylabel('Impedance (Ω)')
    axes[2].legend()
    axes[3].set_title("Imag Impedance yx")
    axes[3].set_xlabel('Easting (m)')
    axes[3].set_ylabel('Impedance (Ω)')
    axes[3].legend()
    plt.suptitle(f"Impedance Volume Averaged Comparison, {freqs[i]}Hz")
    plt.savefig(f"figure_out/vol_avg/z{freqs[i]}Hz.png")
