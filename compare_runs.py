# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

run_nums = ["001", "002", "003", "004", "005"]
data_path = f"simPEG_data/"
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
run_data = {}

for run in run_nums:
    dpred = np.load(data_path+run+'/'+'dpred.npy')
    freqs = np.load(data_path+run+'/'+'freqs.npy')
    rx_locs = np.load(data_path+run+'/'+'rx_locs.npy')

    data = dpred
    data[:, 3, :] += 180 # app resistivity phase quadrant correction
    data[:, 0, :] = -data[:, 0, :]
    data[:, 1, :] = -data[:, 1, :]

    run_data[run] = data

plot_freqs_ind = [0, 10, 20, 30, 40, 50] # plot 1 freq per decade

x_cut = rx_locs[22::45, 0] # cut along y=0

mesh_ind = [int(x) for x in run_nums]


# ======================
# Impedence residuals at center sphere
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for freq in plot_freqs_ind:
    real_xy = []
    imag_xy = []
    real_yx = []
    imag_yx = []

    for run in run_data.values():
        real_xy.append(100*(run[freq, 0, 1012]-Adata[freq, 22, 22, 0])/Adata[freq, 22, 22, 0])
        imag_xy.append(100*(run[freq, 1, 1012]-Adata[freq, 22, 22, 1])/Adata[freq, 22, 22, 1])
        real_yx.append(100*(run[freq, 4, 1012]-Adata[freq, 22, 22, 4])/Adata[freq, 22, 22, 4])
        imag_yx.append(100*(run[freq, 5, 1012]-Adata[freq, 22, 22, 5])/Adata[freq, 22, 22, 5])

    axes[0].plot(mesh_ind, real_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[1].plot(mesh_ind, imag_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[2].plot(mesh_ind, real_yx, '.-', label=f"{freqs[freq]}Hz")
    axes[3].plot(mesh_ind, imag_yx, '.-', label=f"{freqs[freq]}Hz")

    axes[0].set_title("Real Impedance xy")
    axes[0].set_xlabel('Mesh Index')
    axes[0].set_ylabel('Percent')
    axes[0].legend()
    axes[1].set_title("Imag Impedance xy")
    axes[1].set_xlabel('Mesh Index')
    axes[1].set_ylabel('Percent')
    axes[1].legend()
    axes[2].set_title("Real Impedance yx")
    axes[2].set_xlabel('Mesh Index')
    axes[2].set_ylabel('Percent')
    axes[2].legend()
    axes[3].set_title("Imag Impedance yx")
    axes[3].set_xlabel('Mesh Index')
    axes[3].set_ylabel('Percent')
    axes[3].legend()

plt.xticks(mesh_ind)
plt.suptitle("Impedance Residuals at x=0, y=0")
plt.show()


# ======================
# Impedence residuals at outer receiver
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for freq in plot_freqs_ind:
    real_xy = []
    imag_xy = []
    real_yx = []
    imag_yx = []

    for run in run_data.values():
        real_xy.append(100*(run[freq, 0, 22]-Adata[freq, 0, 22, 0])/Adata[freq, 0, 22, 0])
        imag_xy.append(100*(run[freq, 1, 22]-Adata[freq, 0, 22, 1])/Adata[freq, 0, 22, 1])
        real_yx.append(100*(run[freq, 4, 22]-Adata[freq, 0, 22, 4])/Adata[freq, 0, 22, 4])
        imag_yx.append(100*(run[freq, 5, 22]-Adata[freq, 0, 22, 5])/Adata[freq, 0, 22, 5])

    axes[0].plot(mesh_ind, real_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[1].plot(mesh_ind, imag_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[2].plot(mesh_ind, real_yx, '.-', label=f"{freqs[freq]}Hz")
    axes[3].plot(mesh_ind, imag_yx, '.-', label=f"{freqs[freq]}Hz")

    axes[0].set_title("Real Impedance xy")
    axes[0].set_xlabel('Mesh Index')
    axes[0].set_ylabel('Percent')
    axes[0].legend()
    axes[1].set_title("Imag Impedance xy")
    axes[1].set_xlabel('Mesh Index')
    axes[1].set_ylabel('Percent')
    axes[1].legend()
    axes[2].set_title("Real Impedance yx")
    axes[2].set_xlabel('Mesh Index')
    axes[2].set_ylabel('Percent')
    axes[2].legend()
    axes[3].set_title("Imag Impedance yx")
    axes[3].set_xlabel('Mesh Index')
    axes[3].set_ylabel('Percent')
    axes[3].legend()

plt.xticks(mesh_ind)
plt.suptitle("Impedance Residuals at x=-5000, y=0")
plt.show()