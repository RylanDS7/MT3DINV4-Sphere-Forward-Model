# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import matplotlib.patches as patches
import scipy.constants
from simpeg import utils

mu0 = scipy.constants.mu_0

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

x_cut = rx_locs[990:1035, 1] # cut along x=0

plot_freqs_ind = [35, 44, 45, 55] # plot 1 freq per decade


# ======================
# Impedances for individual freqs
# ======================

labels = ['Real Impedance xx', 'Imag Impedance xx', 'Real Impedance xy', 'Imag Impedance xy', 'Real Impedance yx', 'Imag Impedance yx', 'Real Impedance yy', 'Imag Impedance yy']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 4, figsize=(20, 14), sharex=True)
    axes = axes.flatten()

    for i, (ax, label) in enumerate(zip(axes, labels)):
        if i % 2 == 0: # Real Impedance plotting
            ax.plot(x_cut, data[j, i, 990:1035], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, 22, :, i], '.-', label="Analytic")
            ax.set_ylabel('Real Impedance (Ω)')
        else: # Imag Impedance plotting
            ax.plot(x_cut, data[j, i, 990:1035], '.-', label="Simulated")
            ax.plot(x_cut, Adata[j, 22, :, i], '.-', label="Analytic")
            ax.set_ylabel('Imag Impedance (Ω)')

        ax.set_title(label)
        ax.set_xlabel('Easting (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')

    plt.suptitle(f'Impedance along Cut at x=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/final_plots/compare{freqs[j]}Hz.png")



# ======================
# App rho and phase for individual freqs
# ======================

labels = ['App Resistivity xy', 'Phase xy', 'App Resistivity yx', 'Phase yx']

for j in plot_freqs_ind:
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), sharex=True)
    axes = axes.flatten()

    axes[0].plot(x_cut, np.abs(data[j, 2, 990:1035] + data[j, 3, 990:1035]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Simulated")
    axes[0].plot(x_cut, np.abs(Adata[j, 22, :, 2] + Adata[j, 22, :, 3]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Analytic")
    axes[0].set_title(labels[0])

    axes[1].plot(x_cut, np.angle(data[j, 2, 990:1035] + data[j, 3, 990:1035]*1j, deg=True), '.-', label="Simulated")
    axes[1].plot(x_cut, np.angle(Adata[j, 22, :, 2] + Adata[j, 22, :, 3]*1j, deg=True), '.-', label="Analytic")
    axes[1].set_title(labels[1])

    axes[2].plot(x_cut, np.abs(data[j, 4, 990:1035] + data[j, 5, 990:1035]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Simulated")
    axes[2].plot(x_cut, np.abs(Adata[j, 22, :, 4] + Adata[j, 22, :, 5]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]), '.-', label="Analytic")
    axes[2].set_title(labels[2])

    axes[3].plot(x_cut, np.angle(data[j, 4, 990:1035] + data[j, 5, 990:1035]*1j, deg=True), '.-', label="Simulated")
    axes[3].plot(x_cut, np.angle(Adata[j, 22, :, 4] + Adata[j, 22, :, 5]*1j, deg=True), '.-', label="Analytic")
    axes[3].set_title(labels[3])

    for ax in axes:
        ax.set_xlabel('Easting (m)')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(loc='lower left')
        

    plt.suptitle(f'Apparent Resisitivity and Phase along Cut at x=0 for {freqs[j]}Hz')
    plt.tight_layout()
    plt.savefig(f"figure_out/final_plots/compareRho{freqs[j]}Hz.png")


# ======================
# Scatter plot of app rho and phase for individual freqs
# ======================

for j in plot_freqs_ind:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    x, y, z = zip(*rx_locs)

    valuesxy = np.log10(np.abs(data[j, 2, :] + data[j, 3, :]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]))
    valuesyx = np.log10(np.abs(data[j, 4, :] + data[j, 5, :]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]))

    vmin = np.log10(500)
    vmax = np.log10(1000)

    x_min, x_max = -1050, 1050 
    y_min, y_max = -1050, 1050

    # --- Top Plot (XY) ---
    scatter0 = axes[0].scatter(x, y, c=valuesxy, cmap='gist_rainbow', s=100, vmin=vmin, vmax=vmax)
    axes[0].set_xlabel('Northing (m)')
    axes[0].set_ylabel('Easting (m)')
    axes[0].set_title('Receivers Coloured by Log10 Apparent Resistivity XY')

    # Fix aspect ratio and window limits for Top Plot
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # --- Bottom Plot (YX) ---
    scatter1 = axes[1].scatter(x, y, c=valuesyx, cmap='gist_rainbow', s=100, vmin=vmin, vmax=vmax)
    axes[1].set_xlabel('Northing (m)')
    axes[1].set_ylabel('Easting (m)')
    axes[1].set_title('Receivers Coloured by Log10 Apparent Resistivity YX')

    # Fix aspect ratio and window limits for Bottom Plot
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    fig.colorbar(scatter1, ax=axes.ravel(), orientation='vertical', label='Log10 Apparent Resistivity')

    # Save the final figure
    plt.savefig(f"figure_out/final_plots/rxImp{freqs[j]}Hz.png")


# ======================
# Map plot of app rho for individual freqs
# ======================

for j in plot_freqs_ind:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    vmin = np.log10(500)
    vmax = np.log10(1000)

    x_min, x_max = -1050, 1050 
    y_min, y_max = -1050, 1050

    valuesxy = np.log10(np.abs(data[j, 2, :] + data[j, 3, :]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]))
    valuesyx = np.log10(np.abs(data[j, 4, :] + data[j, 5, :]*1j)**2 / (mu0 * 2 * np.pi * freqs[j]))

    # Fix aspect ratio and window limits for Top Plot
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # Fix aspect ratio and window limits for Bottom Plot
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    base_transform = axes[0].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesxy, ax=axes[0], nx=1000, ny=1000, clim=[vmin, vmax], contourOpts={"cmap": "gist_rainbow", "transform": rotation_transform + base_transform})
    base_transform = axes[1].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesyx, ax=axes[1], nx=1000, ny=1000, clim=[vmin, vmax], contourOpts={"cmap": "gist_rainbow", "transform": rotation_transform + base_transform})

    axes[0].set_xlabel('Northing (m)')
    axes[0].set_ylabel('Easting (m)')
    axes[0].set_title('Receivers Coloured by Log10 Apparent Resistivity XY')

    axes[1].set_xlabel('Northing (m)')
    axes[1].set_ylabel('Easting (m)')
    axes[1].set_title('Receivers Coloured by Log10 Apparent Resistivity YX')

    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[0].add_patch(circle)
    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[1].add_patch(circle)

    fig.colorbar(plot, ax=axes.ravel(), orientation='vertical', label='Log10 Apparent Resistivity')

    plt.savefig(f"figure_out/final_plots/rxImpMap{freqs[j]}Hz.png")


# ======================
# Map plot of app rho difference for individual freqs
# ======================

AdataS = Adata.reshape(71, 2025, 12)

for j in plot_freqs_ind:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    vmin = -3
    vmax = 3

    x_min, x_max = -1050, 1050 
    y_min, y_max = -1050, 1050

    dataxy = np.abs(data[j, 2, :] + data[j, 3, :]*1j)**2 / (mu0 * 2 * np.pi * freqs[j])
    datayx = np.abs(data[j, 4, :] + data[j, 5, :]*1j)**2 / (mu0 * 2 * np.pi * freqs[j])
    Adataxy = np.abs(AdataS[j, :, 2] + AdataS[j, :, 3]*1j)**2 / (mu0 * 2 * np.pi * freqs[j])
    Adatayx = np.abs(AdataS[j, :, 4] + AdataS[j, :, 5]*1j)**2 / (mu0 * 2 * np.pi * freqs[j])

    valuesxy = 100*(dataxy - Adataxy) / Adataxy
    valuesyx = 100*(datayx - Adatayx) / Adatayx

    # Fix aspect ratio and window limits for Top Plot
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # Fix aspect ratio and window limits for Bottom Plot
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    base_transform = axes[0].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesxy, ax=axes[0], nx=1000, ny=1000, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})
    base_transform = axes[1].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesyx, ax=axes[1], nx=1000, ny=1000, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})

    axes[0].set_xlabel('Northing (m)')
    axes[0].set_ylabel('Easting (m)')
    axes[0].set_title('Receivers Coloured by Apparent Resistivity XY Percent Difference')

    axes[1].set_xlabel('Northing (m)')
    axes[1].set_ylabel('Easting (m)')
    axes[1].set_title('Receivers Coloured by Apparent Resistivity YX Percent Difference')

    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[0].add_patch(circle)
    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[1].add_patch(circle)

    fig.colorbar(plot, ax=axes.ravel(), orientation='vertical', label='Percent')

    plt.savefig(f"figure_out/final_plots/rxImpDiffMap{freqs[j]}Hz.png")


# ======================
# Map plot of phase for individual freqs
# ======================

for j in plot_freqs_ind:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    vmin = 45
    vmax = 55

    x_min, x_max = -1050, 1050 
    y_min, y_max = -1050, 1050

    valuesxy = np.angle(data[j, 2, :] + data[j, 3, :]*1j, deg=True)
    valuesyx = np.angle(data[j, 4, :] + data[j, 5, :]*1j, deg=True) + 180

    # Fix aspect ratio and window limits for Top Plot
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # Fix aspect ratio and window limits for Bottom Plot
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    base_transform = axes[0].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesxy, nx=1000, ny=1000, ax=axes[0], ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "gist_rainbow_r", "transform": rotation_transform + base_transform})
    base_transform = axes[1].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesyx, ax=axes[1], nx=1000, ny=1000, ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "gist_rainbow_r", "transform": rotation_transform + base_transform})

    axes[0].set_xlabel('Northing (m)')
    axes[0].set_ylabel('Easting (m)')
    axes[0].set_title('Receivers Coloured by Phase XY')

    axes[1].set_xlabel('Northing (m)')
    axes[1].set_ylabel('Easting (m)')
    axes[1].set_title('Receivers Coloured by Phase YX')

    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[0].add_patch(circle)
    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[1].add_patch(circle)

    fig.colorbar(plot, ax=axes.ravel(), orientation='vertical', label='Phase (Degrees)')

    plt.savefig(f"figure_out/final_plots/rxPhaseMap{freqs[j]}Hz.png")


# ======================
# Map plot of phase differences for individual freqs
# ======================

for j in plot_freqs_ind:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    vmin = -1
    vmax = 1

    x_min, x_max = -1050, 1050 
    y_min, y_max = -1050, 1050

    dataxy = np.angle(data[j, 2, :] + data[j, 3, :]*1j, deg=True)
    datayx = np.angle(data[j, 4, :] + data[j, 5, :]*1j, deg=True)
    Adataxy = np.angle(AdataS[j, :, 2] + AdataS[j, :, 3]*1j, deg=True)
    Adatayx = np.angle(AdataS[j, :, 4] + AdataS[j, :, 5]*1j, deg=True)

    valuesxy = (dataxy - Adataxy)
    valuesyx = (datayx - Adatayx)

    # Fix aspect ratio and window limits for Top Plot
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # Fix aspect ratio and window limits for Bottom Plot
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    base_transform = axes[0].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesxy, nx=1000, ny=1000, ax=axes[0], ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})
    base_transform = axes[1].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesyx, ax=axes[1], nx=1000, ny=1000, ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})

    axes[0].set_xlabel('Northing (m)')
    axes[0].set_ylabel('Easting (m)')
    axes[0].set_title('Receivers Coloured by Phase XY Difference')

    axes[1].set_xlabel('Northing (m)')
    axes[1].set_ylabel('Easting (m)')
    axes[1].set_title('Receivers Coloured by Phase YX Difference')

    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[0].add_patch(circle)
    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[1].add_patch(circle)

    fig.colorbar(plot, ax=axes.ravel(), orientation='vertical', label='Phase (Degrees)')

    plt.savefig(f"figure_out/final_plots/rxPhaseDiffMap{freqs[j]}Hz.png")


# ======================
# Map plot of tipper for individual freqs
# ======================

for j in plot_freqs_ind:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    vmin = -0.05
    vmax = 0.05

    x_min, x_max = -1050, 1050 
    y_min, y_max = -1050, 1050

    valuesxy = data[j, 8, :]
    valuesyx = data[j, 10, :]

    # Fix aspect ratio and window limits for Top Plot
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # Fix aspect ratio and window limits for Bottom Plot
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    base_transform = axes[0].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesxy, nx=1000, ny=1000, ax=axes[0], ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})
    base_transform = axes[1].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesyx, ax=axes[1], nx=1000, ny=1000, ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})

    axes[0].set_xlabel('Northing (m)')
    axes[0].set_ylabel('Easting (m)')
    axes[0].set_title('Receivers Coloured by Real Tipper ZX')

    axes[1].set_xlabel('Northing (m)')
    axes[1].set_ylabel('Easting (m)')
    axes[1].set_title('Receivers Coloured by Real Tipper ZY')

    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[0].add_patch(circle)
    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[1].add_patch(circle)

    fig.colorbar(plot, ax=axes.ravel(), orientation='vertical', label='Tipper')

    plt.savefig(f"figure_out/final_plots/rxTipperMap{freqs[j]}Hz.png")


# ======================
# Map plot of tipper differences for individual freqs
# ======================

for j in plot_freqs_ind:
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    vmin = -0.005
    vmax = 0.005

    x_min, x_max = -1050, 1050 
    y_min, y_max = -1050, 1050

    valuesxy = data[j, 8, :] - AdataS[j, :, 8]
    valuesyx = data[j, 10, :] - AdataS[j, :, 10]

    # Fix aspect ratio and window limits for Top Plot
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(x_min, x_max)
    axes[0].set_ylim(y_min, y_max)

    # Fix aspect ratio and window limits for Bottom Plot
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_ylim(y_min, y_max)

    base_transform = axes[0].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesxy, nx=1000, ny=1000, ax=axes[0], ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})
    base_transform = axes[1].transData
    rotation_transform = transforms.Affine2D().rotate_deg(90)
    plot, _ = utils.plot2Ddata(rx_locs, valuesyx, ax=axes[1], nx=1000, ny=1000, ncontour=20, clim=[vmin, vmax], contourOpts={"cmap": "bwr", "transform": rotation_transform + base_transform})

    axes[0].set_xlabel('Northing (m)')
    axes[0].set_ylabel('Easting (m)')
    axes[0].set_title('Receivers Coloured by Real Tipper ZX Difference')

    axes[1].set_xlabel('Northing (m)')
    axes[1].set_ylabel('Easting (m)')
    axes[1].set_title('Receivers Coloured by Real Tipper ZY Difference')

    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[0].add_patch(circle)
    circle = patches.Circle((0, 0), radius=500, edgecolor='white', fill=False, linewidth=2)
    axes[1].add_patch(circle)

    fig.colorbar(plot, ax=axes.ravel(), orientation='vertical', label='Tipper')

    plt.savefig(f"figure_out/final_plots/rxTipperDiffMap{freqs[j]}Hz.png")