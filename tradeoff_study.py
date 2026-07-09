# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

run_nums = ["41", "42", "43", "44", "45", "46"]
times = np.array([6175, 5765, 5542, 5349, 5180, 4645])
data_path = "simPEG_data/tradeoff2/"
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
    dpred = np.load(data_path+f'dpred{run}.npy')
    freqs = np.load(data_path+f'freqs.npy')
    rx_locs = np.load(data_path+f'rx_locs.npy')

    data = dpred
    data[:, 3, :] += 180 # app resistivity phase quadrant correction
    data[:, 0, :] = -data[:, 0, :]
    data[:, 1, :] = -data[:, 1, :]

    run_data[run] = data

plot_freqs_ind = [0, 10, 20, 30, 40, 50] # plot 1 freq per decade

x_cut = rx_locs[22::45, 0] # cut along y=0

mesh_ind = np.arange(len(run_nums))

pdf = PdfPages("figure_out/tradeoff2.pdf")


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
        real_xy.append(100*(np.abs(run[freq, 0, 1012]-Adata[freq, 22, 22, 0]))/Adata[freq, 22, 22, 0])
        imag_xy.append(100*(np.abs(run[freq, 1, 1012]-Adata[freq, 22, 22, 1]))/Adata[freq, 22, 22, 1])
        real_yx.append(100*(np.abs(run[freq, 4, 1012]-Adata[freq, 22, 22, 4]))/Adata[freq, 22, 22, 4])
        imag_yx.append(100*(np.abs(run[freq, 5, 1012]-Adata[freq, 22, 22, 5]))/Adata[freq, 22, 22, 5])

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
pdf.savefig(fig)
plt.close(fig)


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
        real_xy.append(100*(np.abs(run[freq, 0, 22]-Adata[freq, 0, 22, 0]))/Adata[freq, 0, 22, 0])
        imag_xy.append(100*(np.abs(run[freq, 1, 22]-Adata[freq, 0, 22, 1]))/Adata[freq, 0, 22, 1])
        real_yx.append(100*(np.abs(run[freq, 4, 22]-Adata[freq, 0, 22, 4]))/Adata[freq, 0, 22, 4])
        imag_yx.append(100*(np.abs(run[freq, 5, 22]-Adata[freq, 0, 22, 5]))/Adata[freq, 0, 22, 5])

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
pdf.savefig(fig)
plt.close(fig)


# ======================
# App res residuals at center sphere
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for freq in plot_freqs_ind:
    rho_xy = []
    phase_xy = []
    rho_yx = []
    phase_yx = []

    for run in run_data.values():
        rho_xy.append(100*(np.abs(run[freq, 2, 1012]-Adata[freq, 22, 22, 2]))/Adata[freq, 22, 22, 2])
        phase_xy.append(100*(np.abs(run[freq, 3, 1012]-Adata[freq, 22, 22, 3]))/Adata[freq, 22, 22, 3])
        rho_yx.append(100*(np.abs(run[freq, 6, 1012]-Adata[freq, 22, 22, 6]))/Adata[freq, 22, 22, 6])
        phase_yx.append(100*(np.abs(run[freq, 7, 1012]-Adata[freq, 22, 22, 7]))/Adata[freq, 22, 22, 7])

    axes[0].plot(mesh_ind, rho_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[1].plot(mesh_ind, phase_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[2].plot(mesh_ind, rho_yx, '.-', label=f"{freqs[freq]}Hz")
    axes[3].plot(mesh_ind, phase_yx, '.-', label=f"{freqs[freq]}Hz")

    axes[0].set_title("Apparent Resistivity xy")
    axes[0].set_xlabel('Mesh Index')
    axes[0].set_ylabel('Percent')
    axes[0].legend()
    axes[1].set_title("Phase xy")
    axes[1].set_xlabel('Mesh Index')
    axes[1].set_ylabel('Percent')
    axes[1].legend()
    axes[2].set_title("Apparent Resistivity yx")
    axes[2].set_xlabel('Mesh Index')
    axes[2].set_ylabel('Percent')
    axes[2].legend()
    axes[3].set_title("Phase yx")
    axes[3].set_xlabel('Mesh Index')
    axes[3].set_ylabel('Percent')
    axes[3].legend()

plt.xticks(mesh_ind)
plt.suptitle("Apparent Resistivity and Phase Residuals at x=0, y=0")
pdf.savefig(fig)
plt.close(fig)


# ======================
# App res residuals at outer receiver
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for freq in plot_freqs_ind:
    rho_xy = []
    phase_xy = []
    rho_yx = []
    phase_yx = []

    for run in run_data.values():
        rho_xy.append(100*(np.abs(run[freq, 2, 22]-Adata[freq, 0, 22, 2]))/Adata[freq, 0, 22, 2])
        phase_xy.append(100*(np.abs(run[freq, 3, 22]-Adata[freq, 0, 22, 3]))/Adata[freq, 0, 22, 3])
        rho_yx.append(100*(np.abs(run[freq, 6, 22]-Adata[freq, 0, 22, 6]))/Adata[freq, 00, 22, 6])
        phase_yx.append(100*(np.abs(run[freq, 7, 22]-Adata[freq, 0, 22, 7]))/Adata[freq, 0, 22, 7])

    axes[0].plot(mesh_ind, rho_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[1].plot(mesh_ind, phase_xy, '.-', label=f"{freqs[freq]}Hz")
    axes[2].plot(mesh_ind, rho_yx, '.-', label=f"{freqs[freq]}Hz")
    axes[3].plot(mesh_ind, phase_yx, '.-', label=f"{freqs[freq]}Hz")

    axes[0].set_title("Apparent Resistivity xy")
    axes[0].set_xlabel('Mesh Index')
    axes[0].set_ylabel('Percent')
    axes[0].legend()
    axes[1].set_title("Phase xy")
    axes[1].set_xlabel('Mesh Index')
    axes[1].set_ylabel('Percent')
    axes[1].legend()
    axes[2].set_title("Apparent Resistivity yx")
    axes[2].set_xlabel('Mesh Index')
    axes[2].set_ylabel('Percent')
    axes[2].legend()
    axes[3].set_title("Phase yx")
    axes[3].set_xlabel('Mesh Index')
    axes[3].set_ylabel('Percent')
    axes[3].legend()

plt.xticks(mesh_ind)
plt.suptitle("Apparent Resistivity and Phase Residuals at x=-5000, y=0")
pdf.savefig(fig)
plt.close(fig)



# ======================
# Condensed residuals plots
# ======================

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True, dpi=300)
axes = axes.flatten()

refine_levels = [2.5, 5, 10, 20, 40, 80]

plot_freqs_ind = [30, 40, 50] # change to smaller range


for freq in plot_freqs_ind:
    real_xy_center = []
    imag_xy_center = []
    real_xy_edge = []
    imag_xy_edge = []

    for run in run_data.values():
        real_xy_center.append(100*(np.abs(run[freq, 0, 1012]-Adata[freq, 22, 22, 0]))/Adata[freq, 22, 22, 0])
        imag_xy_center.append(100*(np.abs(run[freq, 1, 1012]-Adata[freq, 22, 22, 1]))/Adata[freq, 22, 22, 1])
        real_xy_edge.append(100*(np.abs(run[freq, 0, 22]-Adata[freq, 0, 22, 0]))/Adata[freq, 0, 22, 0])
        imag_xy_edge.append(100*(np.abs(run[freq, 1, 22]-Adata[freq, 0, 22, 1]))/Adata[freq, 0, 22, 1])

    axes[0].plot(refine_levels, real_xy_center, 'o-', label=f"Real {freqs[freq]}Hz")
    axes[0].plot(refine_levels, imag_xy_center, 'o-', label=f"Imag {freqs[freq]}Hz")
    axes[1].plot(refine_levels, real_xy_edge, 'o-', label=f"Real {freqs[freq]}Hz")
    axes[1].plot(refine_levels, imag_xy_edge, 'o-', label=f"Imag {freqs[freq]}Hz")

    axes[0].set_title("Impedance xy at x=0, y=0", fontsize=24)
    axes[0].set_ylabel('Percent Residuals', fontsize=20)
    axes[0].tick_params(axis='both', labelsize=16) 
    axes[0].grid()
    axes[1].set_title("Impedance xy at x=-5000, y=0", fontsize=24)
    axes[1].set_xlabel('Reciever Refinement (m)', fontsize=20)
    axes[1].set_ylabel('Percent Residuals', fontsize=20)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].legend(fontsize=15)
    axes[1].grid()

plt.yticks([1, 2, 3, 4, 5])
plt.xticks([2.5, 10, 20, 40, 80], fontsize=16)
plt.tight_layout()
plt.savefig("figure_out/impRXtradeoff.png")
pdf.savefig(fig)
plt.close(fig)


# ======================
# Time plot
# ======================

fig = plt.figure(figsize=(10, 8))
plt.plot(refine_levels, times / 3600, '.-')
plt.xticks([2.5, 10, 20, 40, 80], fontsize=12)
plt.xlabel('Reciever Refinement (m)', fontsize=15)
plt.ylabel('Computation Time (Hr)', fontsize=15)
plt.title("Computation Time as a Function of Reciever Refinement", fontsize=18)
plt.savefig("figure_out/timeRXtradeoff.png")
pdf.savefig(fig)
plt.close(fig)

# close the pdf
pdf.close()