# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

run_num = "011"
data_path = Path(f"simPEG_data/{run_num}/")
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

freqs = np.load(data_path / 'freqs.npy')
rx_locs = np.load(data_path / 'rx_locs.npy')

data = np.zeros((len(freqs), 8, len(rx_locs)))
times = np.zeros((len(freqs)))
for file in data_path.iterdir():
    if file.stem[:5] == "dpred":
        dpred = np.load(file)
        index = int(file.stem[5:])
        data[index] = dpred
    if file.stem[:4] == "time":
        time = np.load(file)
        index = int(file.stem[4:])
        times[index] = time[0]
    
data[:, 3, :] += 180 # app resistivity phase quadrant correction
data[:, 0, :] = -data[:, 0, :]
data[:, 1, :] = -data[:, 1, :]


# ======================
# Plot center residuals at all freqs
# ======================

plt.figure()

plt.plot(freqs, 100*(data[:, 0, 1012]-Adata[:, 22, 22, 0])/Adata[:, 22, 22, 0], '.', label="Real Impedance xy")
plt.plot(freqs, 100*(data[:, 1, 1012]-Adata[:, 22, 22, 1])/Adata[:, 22, 22, 1], '.', label="Imag Impedance xy")
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

plt.plot(freqs, 100*(data[:, 0, 22]-Adata[:, 0, 22, 0])/Adata[:, 0, 22, 0], '.', label="Real Impedance xy")
plt.plot(freqs, 100*(data[:, 1, 22]-Adata[:, 0, 22, 1])/Adata[:, 0, 22, 1], '.', label="Imag Impedance xy")
plt.plot(freqs, 100*(data[:, 4, 22]-Adata[:, 0, 22, 4])/Adata[:, 0, 22, 4], '.', label="Real Impedance yx")
plt.plot(freqs, 100*(data[:, 5, 22]-Adata[:, 0, 22, 5])/Adata[:, 0, 22, 5], '.', label="Imag Impedance yx")

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Impedance Percent Residuals')
plt.title('Impedance Residuals at (x,y) = (-5000,0)')
plt.legend()
plt.savefig(f"figure_out/{run_num}/edgeFreqResiduals.png")


# ======================
# Plot times at all freqs
# ======================

plt.figure()

plt.plot(freqs, times / 60, '.')
plt.axhline(y=np.mean(times/60), color='orange', linestyle='--')

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Computation Time (min)')
plt.title('Computation Times per Frequency')
plt.savefig(f"figure_out/{run_num}/freqTimes.png")


# ======================
# Calculate total residuals
# ======================

data_grid = data.reshape(len(freqs), 8, 45, 45)

residuals_real_xy = 100*(data_grid[:, 0, :, :]-Adata[:, :, :, 0])/Adata[:, :, :, 0]
residuals_imag_xy = 100*(data_grid[:, 1, :, :]-Adata[:, :, :, 1])/Adata[:, :, :, 1]
residuals_real_yx = 100*(data_grid[:, 4, :, :]-Adata[:, :, :, 4])/Adata[:, :, :, 4]
residuals_imag_yx = 100*(data_grid[:, 5, :, :]-Adata[:, :, :, 5])/Adata[:, :, :, 5]

mean_real_xy = []
mean_imag_xy = []
mean_real_yx = []
mean_imag_yx = []
for i in np.arange(71):
    mean_real_xy.append(np.mean(np.vstack(residuals_real_xy[i])))
    mean_imag_xy.append(np.mean(np.vstack(residuals_imag_xy[i])))
    mean_real_yx.append(np.mean(np.vstack(residuals_real_yx[i])))
    mean_imag_yx.append(np.mean(np.vstack(residuals_imag_yx[i])))

plt.figure()

plt.plot(freqs, mean_real_xy, '.', label="Real Impedance xy")
plt.plot(freqs, mean_imag_xy, '.', label="Imag Impedance xy")
plt.plot(freqs, mean_real_yx, '.', label="Real Impedance yx")
plt.plot(freqs, mean_imag_yx, '.', label="Imag Impedance yx")

all_mean = np.concatenate((mean_real_xy, mean_imag_xy, mean_real_yx, mean_imag_yx))
print(f"Mean of Absolute Value Mean Errors: {np.mean(np.abs(all_mean))}")

plt.xscale('log')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Mean Impedance Percent Residuals')
plt.title('Mean Impedance Residuals for all Receivers')
plt.legend()
plt.savefig(f"figure_out/{run_num}/meanFreqResiduals.png")