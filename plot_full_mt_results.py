# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

run_num = "012"
data_path = f"simPEG_data/{run_num}/"
analytic_path = "analytic_data/"

# load and parse analytic data
Adata = np.zeros([71, 45, 45, 12])

impA = np.load(analytic_path+'imp.npy')

Adata[:, :, :, 0] = impA[:, :, :, 0, 0].real # impedence_xx
Adata[:, :, :, 1] = impA[:, :, :, 0, 0].imag # impedence_xx
Adata[:, :, :, 2] = impA[:, :, :, 0, 1].real # impedance_xy
Adata[:, :, :, 3] = impA[:, :, :, 0, 1].imag # impedance_xy
Adata[:, :, :, 4] = impA[:, :, :, 1, 0].real # impedence_yx       
Adata[:, :, :, 5] = impA[:, :, :, 1, 0].imag # impedence_yx         
Adata[:, :, :, 6] = impA[:, :, :, 0, 0].real # impedance_yy
Adata[:, :, :, 7] = impA[:, :, :, 0, 0].imag # impedance_yy

Adata[:, :, :, 8] = impA[:, :, :, 2, 0].real # tipper_zx       
Adata[:, :, :, 9] = impA[:, :, :, 2, 0].imag # tipper_zx          
Adata[:, :, :, 10] = impA[:, :, :, 2, 1].real # tipper_zy  
Adata[:, :, :, 11] = impA[:, :, :, 2, 1].imag # tipper_zy 

# load and parse simPEG data
dpred = np.load(data_path+'dpredVA.npy')
freqs = np.load(data_path+'freqsVA.npy')
rx_locs = np.load(data_path+'rx_locsVA.npy')

data = -dpred.reshape(len(freqs), 12, len(rx_locs)) # sign flip for convention change

plot_freqs_ind = [0, 10, 20, 30, 40, 50, 60, 70] # plot 1 freq per decade

x_cut = rx_locs[22::45, 0] # cut along y=0

# ======================
# Impedances for individual freqs
# ======================

labels = ['Real Impedance xx', 'Imag Impedance xx', 'Real Impedance xy', 'Imag Impedance xy', 'Real Impedance yx', 'Imag Impedance yx', 'Real Impedance yy', 'Imag Impedance yy']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 4, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        if i % 2 == 0: # Real Impedance plotting
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

    plt.suptitle(f'Impedance along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/compare{freqs[j]}Hz.png")


# ======================
# Tipper for individual freqs
# ======================

labels = ['Real Tipper zx', 'Imag Tipper zx', 'Real Tipper zy', 'Imag Tipper zy']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        if i % 2 == 0: # Real Impedance plotting
            ax.plot(x_cut, data[j, i+8, 22::45], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, :, 22, i+8], '.-', label="Analytic")
            ax.set_ylabel('Real Impedance (Ω)')
        else: # Imag Impedance plotting
            ax.plot(x_cut, data[j, i+8, 22::45], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, :, 22, i+8], '.-', label="Analytic")
            ax.set_ylabel('Imag Impedance (Ω)')

        ax.set_title(label)
        ax.set_xlabel('Easting (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')

    plt.suptitle(f'Tipper along Cut at y=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/{run_num}/compareTipper{freqs[j]}Hz.png")


# ======================
# Impedance residuals
# ======================

fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i in plot_freqs_ind:
    axes[0].plot(x_cut, 100*(data[i, 2, 22::45]-Adata[i, :, 22, 2])/Adata[i, :, 22, 2], '.-', label=f"Percent Residuals {freqs[i]}Hz")
    axes[1].plot(x_cut, 100*(data[i, 3, 22::45]-Adata[i, :, 22, 3])/Adata[i, :, 22, 3], '.-', label=f"Percent Residuals {freqs[i]}Hz")
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
# Tipper residuals
# ======================

plot_freqs_ind = [0, 10, 20, 30, 40, 50, 60]

fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
axes = axes.flatten()

for i in plot_freqs_ind:
    axes[0].plot(x_cut, 100*(data[i, 8, 22::45]-Adata[i, :, 22, 8])/Adata[i, :, 22, 8], '.-', label=f"Percent Residuals {freqs[i]}Hz")
    axes[1].plot(x_cut, 100*(data[i, 9, 22::45]-Adata[i, :, 22, 9])/Adata[i, :, 22, 9], '.-', label=f"Percent Residuals {freqs[i]}Hz")

axes[0].set_title("Real Tipper zx")
axes[0].set_xlabel('Easting (m)')
axes[0].set_ylabel('Percent')
axes[0].legend()
axes[1].set_title("Imag Tipper zx")
axes[1].set_xlabel('Easting (m)')
axes[1].set_ylabel('Percent')
axes[1].legend()

plt.suptitle("Tipper Residuals along Cut at y=0")
plt.savefig(f"figure_out/{run_num}/tipperResiduals.png")


# ======================
# Plot center residuals at all freqs
# ======================

plt.figure()

plt.plot(freqs, 100*(data[:, 2, 1012]-Adata[:, 22, 22, 2])/Adata[:, 22, 22, 2], '.', label="Real Impedance xy")
plt.plot(freqs, 100*(data[:, 3, 1012]-Adata[:, 22, 22, 3])/Adata[:, 22, 22, 3], '.', label="Imag Impedance xy")
plt.plot(freqs, 100*(data[:, 4, 1012]-Adata[:, 22, 22, 4])/Adata[:, 22, 22, 4], '.', label="Real Impedance yx")
plt.plot(freqs, 100*(data[:, 5, 1012]-Adata[:, 22, 22, 5])/Adata[:, 22, 22, 5], '.', label="Imag Impedance yx")

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance Percent Residuals')
plt.title('Impedance Residuals at (x,y) = (0,0)')
plt.legend()
plt.savefig(f"figure_out/{run_num}/centerFreqResiduals.png")


# ======================
# Plot edge residuals at all freqs
# ======================

plt.figure()

plt.plot(freqs, 100*(data[:, 2, 22]-Adata[:, 0, 22, 2])/Adata[:, 0, 22, 2], '.', label="Real Impedance xy")
plt.plot(freqs, 100*(data[:, 3, 22]-Adata[:, 0, 22, 3])/Adata[:, 0, 22, 3], '.', label="Imag Impedance xy")
plt.plot(freqs, 100*(data[:, 4, 22]-Adata[:, 0, 22, 4])/Adata[:, 0, 22, 4], '.', label="Real Impedance yx")
plt.plot(freqs, 100*(data[:, 5, 22]-Adata[:, 0, 22, 5])/Adata[:, 0, 22, 5], '.', label="Imag Impedance yx")

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance Percent Residuals')
plt.title('Impedance Residuals at (x,y) = (-5000,0)')
plt.legend()
plt.savefig(f"figure_out/{run_num}/edgeFreqResiduals.png")


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
