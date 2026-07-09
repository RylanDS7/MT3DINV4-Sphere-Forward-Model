# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

run_num = "009"
data_path = f"simPEG_data/tradeoff_vol_avg/"
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
dpred = np.load(data_path+'dpred53.npy')
freqs = np.load(data_path+'freqs.npy')
rx_locs = np.load(data_path+'rx_locs.npy')

data = dpred
data[:, 3, :] += 180 # app resistivity phase quadrant correction
data[:, 0, :] = -data[:, 0, :]
data[:, 1, :] = -data[:, 1, :]

plot_freqs_ind = [0, 10, 20, 30, 40, 50, 60] # plot 1 freq per decade

x_cut = rx_locs[22::45, 0] # cut along y=0

# ======================
# Apparent resisitivty and phase plots for individual freqs
# ======================

labels = ['Real Impedance xy', 'Imag Impedance xy', 'App Res xy', 'Phase xy', 'Real Impedance yx', 'Imag Impedance yx', 'App Res yx', 'Phase yx']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 4, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        if i % 4 == 2: # Apparent resistivity plotting
            ax.plot(x_cut, data[j, i, 22::45], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, :, 22, i], '.-', label="Analytic")
            ax.set_ylabel('App Res (Ωm)')
            ax.plot(x_cut, data[j, i, 22::45]-Adata[j, :, 22, i], '.-', label="Residual: Sim - Analytic")
        elif i % 4 == 3: # Phase plotting
            ax.plot(x_cut, data[j, i, 22::45], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, :, 22, i], '.-', label="Analytic")
            ax.set_ylabel('Phase (Degrees)')
        elif i % 4 == 0: # Real Impedance plotting
            ax.plot(x_cut, data[j, i, 22::45], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, :, 22, i], '.-', label="Analytic")
            ax.set_ylabel('Real Impedance (Ω)')
        else: # Imag Impedance plotting
            ax.plot(x_cut, data[j, i, 22::45], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, :, 22, i], '.-', label="Analytic")
            ax.set_ylabel('Imag Impedance (Ω)')

        ax.set_title(label)
        ax.set_xlabel('Easting (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')

    plt.suptitle(f'Apparent Resistivity, Phase, and Impedance along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/compare{freqs[j]}Hz.png")



# ======================
# Impedance residuals
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i in plot_freqs_ind:
    axes[0].plot(x_cut, 100*(data[i, 0, 22::45]-Adata[i, :, 22, 0])/Adata[i, :, 22, 0], '.-', label=f"Percent Residuals {freqs[i]}Hz")
    axes[1].plot(x_cut, 100*(data[i, 1, 22::45]-Adata[i, :, 22, 1])/Adata[i, :, 22, 1], '.-', label=f"Percent Residuals {freqs[i]}Hz")
    axes[2].plot(x_cut, 100*(data[i, 4, 22::45]-Adata[i, :, 22, 4])/Adata[i, :, 22, 4], '.-', label=f"Percent Residuals {freqs[i]}Hz")
    axes[3].plot(x_cut, 100*(data[i, 5, 22::45]-Adata[i, :, 22, 5])/Adata[i, :, 22, 5], '.-', label=f"Percent Residuals {freqs[i]}Hz")

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
plt.suptitle("Impedance Residuals along Cut at y=0")
plt.savefig(f"figure_out/{run_num}/impResiduals.png")


# ======================
# Apparent resisitivty and phase xy
# ======================

fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
axes = axes.flatten()

for i in plot_freqs_ind:
    axes[0].plot(x_cut, data[i, 2, 22::45], '.-', label=f"Simulated {freqs[i]}Hz")
    axes[0].plot(x_cut, Adata[i, :, 22, 2], '.-', label=f"Analytic {freqs[i]}Hz")
    axes[1].plot(x_cut, data[i, 3, 22::45], '.-', label=f"Simulated {freqs[i]}Hz")
    axes[1].plot(x_cut, Adata[i, :, 22, 3], '.-', label=f"Analytic {freqs[i]}Hz")

axes[0].set_title("Apparent Resisitivy xy")
axes[0].set_xlabel('Easting (m)')
axes[0].set_ylabel('App Res (Ωm)')
axes[0].legend()
axes[1].set_title("Phase xy")
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Phase (Degrees)')
axes[1].legend()
plt.suptitle("xy Apparent Resistivity and Phase along Cut at y=0")
plt.savefig(f"figure_out/{run_num}/xyCompare.png")



# ======================
# Apparent resisitivty and phase residuals xy
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i in plot_freqs_ind:
    axes[0].plot(x_cut, data[i, 2, 22::45]-Adata[i, :, 22, 2], '.-', label=f"Residuals {freqs[i]}Hz")
    axes[1].plot(x_cut, data[i, 3, 22::45]-Adata[i, :, 22, 3], '.-', label=f"Residuals {freqs[i]}Hz")
    axes[2].plot(x_cut, 100*(data[i, 2, 22::45]-Adata[i, :, 22, 2])/Adata[i, :, 22, 2], '.-', label=f"Percent Residuals {freqs[i]}Hz")
    axes[3].plot(x_cut, 100*(data[i, 3, 22::45]-Adata[i, :, 22, 3])/Adata[i, :, 22, 3], '.-', label=f"Percent Residuals {freqs[i]}Hz")

axes[0].set_title("Apparent Resisitivy xy")
axes[0].set_xlabel('Easting (m)')
axes[0].set_ylabel('App Res (Ωm)')
axes[0].legend()
axes[1].set_title("Phase xy")
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Phase (Degrees)')
axes[1].legend()
axes[2].set_xlabel('Easting (m)')
axes[2].set_ylabel('Percent')
axes[2].legend()
axes[3].set_xlabel('Easting (m)')
axes[3].set_ylabel('Percent')
axes[3].legend()
plt.suptitle("xy Apparent Resistivity and Phase Residuals along Cut at y=0")
plt.savefig(f"figure_out/{run_num}/xyResiduals.png")



# ======================
# Apparent resisitivty and phase yx
# ======================

fig, axes = plt.subplots(1, 2, figsize=(16, 8), sharex=True)
axes = axes.flatten()

for i in plot_freqs_ind:
    axes[0].plot(x_cut, data[i, 6, 22::45], '.-', label=f"Simulated {freqs[i]}Hz")
    axes[0].plot(x_cut, Adata[i, :, 22, 6], '.-', label=f"Analytic {freqs[i]}Hz")
    axes[1].plot(x_cut, data[i, 7, 22::45], '.-', label=f"Simulated {freqs[i]}Hz")
    axes[1].plot(x_cut, Adata[i, :, 22, 7], '.-', label=f"Analytic {freqs[i]}Hz")

axes[0].set_title("Apparent Resisitivy yx")
axes[0].set_xlabel('Easting (m)')
axes[0].set_ylabel('App Res (Ωm)')
axes[0].legend()
axes[1].set_title("Phase yx")
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Phase (Degrees)')
axes[1].legend()
plt.suptitle("yx Apparent Resistivity and Phase along Cut at y=0")
plt.savefig(f"figure_out/{run_num}/yxCompare.png")



# ======================
# Apparent resisitivty and phase residuals yx
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i in plot_freqs_ind:
    axes[0].plot(x_cut, data[i, 6, 22::45]-Adata[i, :, 22, 6], '.-', label=f"Residuals {freqs[i]}Hz")
    axes[1].plot(x_cut, data[i, 7, 22::45]-Adata[i, :, 22, 7], '.-', label=f"Residuals {freqs[i]}Hz")
    axes[2].plot(x_cut, 100*(data[i, 6, 22::45]-Adata[i, :, 22, 6])/Adata[i, :, 22, 6], '.-', label=f"Percent Residuals {freqs[i]}Hz")
    axes[3].plot(x_cut, 100*(data[i, 7, 22::45]-Adata[i, :, 22, 7])/Adata[i, :, 22, 7], '.-', label=f"Percent Residuals {freqs[i]}Hz")

axes[0].set_title("Apparent Resisitivy yx")
axes[0].set_xlabel('Easting (m)')
axes[0].set_ylabel('App Res (Ωm)')
axes[0].legend()
axes[1].set_title("Phase yx")
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Phase (Degrees)')
axes[1].legend()
axes[2].set_xlabel('Easting (m)')
axes[2].set_ylabel('Percent')
axes[2].legend()
axes[3].set_xlabel('Easting (m)')
axes[3].set_ylabel('Percent')
axes[3].legend()
plt.suptitle("yx Apparent Resistivity and Phase Residuals along Cut at y=0")
plt.savefig(f"figure_out/{run_num}/yxResiduals.png")


# ======================
# Plot center residuals at all freqs
# ======================

plt.figure()

plt.plot(freqs[0:70:10], 100*(data[0:70:10, 0, 1012]-Adata[0:70:10, 22, 22, 0])/Adata[0:70:10, 22, 22, 0], '.-', label="Real Impedance xy")
plt.plot(freqs[0:70:10], 100*(data[0:70:10, 1, 1012]-Adata[0:70:10, 22, 22, 1])/Adata[0:70:10, 22, 22, 1], '.-', label="Imag Impedance xy")
plt.plot(freqs[0:70:10], 100*(data[0:70:10, 4, 1012]-Adata[0:70:10, 22, 22, 4])/Adata[0:70:10, 22, 22, 4], '.-', label="Real Impedance yx")
plt.plot(freqs[0:70:10], 100*(data[0:70:10, 5, 1012]-Adata[0:70:10, 22, 22, 5])/Adata[0:70:10, 22, 22, 5], '.-', label="Imag Impedance yx")

plt.xscale('log')
plt.legend()
plt.savefig(f"figure_out/{run_num}/centerFreqResiduals.png")


# ======================
# Compile into pdf
# ======================

import os
import re
import img2pdf

# Path to the folder containing your PNG images
folder_path = f'./figure_out/{run_num}/' 

# Find all PNG files in the directory
png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]

def extract_hz(filename):
    # Force mesh.png to the very beginning
    if filename.lower() == 'mesh.png':
        return float('-inf')  # Negative infinity is lower than any number
        
    # Searches for a decimal number between "compare" and "Hz"
    match = re.search(r"compare([\d\.]+)Hz", filename)
    if match:
        return float(match.group(1)) # Converts "0.001" into 0.001
        
    return float('inf') # Move any other mismatched files to the very end

# 3. Sort using the custom numeric key
png_files.sort(key=extract_hz)

# Add full paths
image_paths = [os.path.join(folder_path, f) for f in png_files]

# Convert to PDF and save
output_pdf = f'figure_out/{run_num}/rstutters_fwd_results.pdf'
with open(output_pdf, "wb") as f:
    f.write(img2pdf.convert(image_paths))
