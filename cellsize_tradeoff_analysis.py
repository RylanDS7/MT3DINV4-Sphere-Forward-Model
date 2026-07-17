# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

data_path = Path(f"simPEG_data/cellsize_tradeoff/")
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

VAdata = np.zeros((11, len(freqs), 8, len(rx_locs)))
NAdata = np.zeros((11, len(freqs), 8, len(rx_locs)))
for file in data_path.iterdir():
    if file.stem[:7] == "dpredNA":
        dpred = np.load(file).reshape(len(freqs), 8, len(rx_locs))
        index = int(file.stem[7:]) - 5

        dpred[:, 3, :] += 180 # app resistivity phase quadrant correction
        dpred[:, 0, :] = -dpred[:, 0, :]
        dpred[:, 1, :] = -dpred[:, 1, :]

        NAdata[index] = dpred
    elif file.stem[:5] == "dpred":
        dpred = np.load(file).reshape(len(freqs), 8, len(rx_locs))
        index = int(file.stem[5:]) - 5

        dpred[:, 3, :] += 180 # app resistivity phase quadrant correction
        dpred[:, 0, :] = -dpred[:, 0, :]
        dpred[:, 1, :] = -dpred[:, 1, :]

        VAdata[index] = dpred

VAtimes = np.array([14938, 13103, 12364, 12180, 10918, 8681, 7382, 7434, 8826, 8107, 6577])
NAtimes = np.array([14977, 13093, 12358, 12289, 11128, 8767, 7511, 7600, 8888, 8198, 6759])

VAcells = np.array([880944, 745368, 649132, 577788, 529152, 488328, 445236, 410712, 392596, 365884, 352696])
NAcells = np.array([880944, 745368, 649132, 577788, 529152, 488328, 445236, 410712, 392596, 365884, 352696])

VAram = np.array([30.99, 26.50, 23.68, 21.72, 19.84, 17.46, 15.89, 14.91, 15.02, 14.02, 13.16])
NAram = np.array([31.07, 26.49, 23.73, 21.76, 19.78, 17.49, 15.91, 14.93, 15.05, 13.92, 13.16])


# ======================
# Center residuals vs time, 100Hz
# ======================

plt.figure()

plt.scatter(NAtimes / 3600, 100*(NAdata[:, 50, 0, 1012]-Adata[50, 22, 22, 0])/Adata[50, 22, 22, 0], label='Real Normal Procedure')
plt.scatter(NAtimes / 3600, 100*(NAdata[:, 50, 1, 1012]-Adata[50, 22, 22, 1])/Adata[50, 22, 22, 1], label='Imag Normal Procedure')
plt.scatter(VAtimes / 3600, 100*(VAdata[:, 50, 0, 1012]-Adata[50, 22, 22, 0])/Adata[50, 22, 22, 0], label='Real Harmonic Averaged')
plt.scatter(VAtimes / 3600, 100*(VAdata[:, 50, 1, 1012]-Adata[50, 22, 22, 1])/Adata[50, 22, 22, 1], label='Imag Harmonic Averaged')

plt.xlabel('Computation Time (hr)')
plt.ylabel('Impedance Percent Residuals')
plt.title('Impedance Residuals at (x,y) = (0,0) vs Time, 100Hz')
plt.legend()
plt.grid()
plt.savefig(f"figure_out/cellsize_tradeoff/centerResidualsTime100.png")
    


# ======================
# Edge residuals vs time, 100Hz
# ======================

plt.figure()

plt.scatter(NAtimes / 3600, 100*(NAdata[:, 50, 0, 22]-Adata[50, 0, 22, 0])/Adata[50, 0, 22, 0], label='Real Normal Procedure')
plt.scatter(NAtimes / 3600, 100*(NAdata[:, 50, 1, 22]-Adata[50, 0, 22, 1])/Adata[50, 0, 22, 1], label='Imag Normal Procedure')
plt.scatter(VAtimes / 3600, 100*(VAdata[:, 50, 0, 22]-Adata[50, 0, 22, 0])/Adata[50, 0, 22, 0], label='Real Harmonic Averaged')
plt.scatter(VAtimes / 3600, 100*(VAdata[:, 50, 1, 22]-Adata[50, 0, 22, 1])/Adata[50, 0, 22, 1], label='Imag Harmonic Averaged')

plt.xlabel('Computation Time (hr)')
plt.ylabel('Impedance Percent Residuals')
plt.title('Impedance Residuals at (x,y) = (-5000,0) vs Time, 100Hz')
plt.legend()
plt.grid()
plt.savefig(f"figure_out/cellsize_tradeoff/edgeResidualsTime100.png")


# ======================
# Center residuals vs time, all freqs
# ======================

plt.figure()

NAresiduals_real = np.array([100*(NAdata[:, f, 0, 1012]-Adata[f, 22, 22, 0])/Adata[f, 22, 22, 0] for f in np.arange(71)])
NAresiduals_imag = np.array([100*(NAdata[:, f, 1, 1012]-Adata[f, 22, 22, 1])/Adata[f, 22, 22, 1] for f in np.arange(71)])
VAresiduals_real = np.array([100*(VAdata[:, f, 0, 1012]-Adata[f, 22, 22, 0])/Adata[f, 22, 22, 0] for f in np.arange(71)])
VAresiduals_imag = np.array([100*(VAdata[:, f, 1, 1012]-Adata[f, 22, 22, 1])/Adata[f, 22, 22, 1] for f in np.arange(71)])

plt.scatter(NAtimes / 3600, np.mean(np.abs(NAresiduals_real), axis=0), label='Real Normal Procedure', marker='x')
plt.scatter(NAtimes / 3600, np.mean(np.abs(NAresiduals_imag), axis=0), label='Imag Normal Procedure', marker='x')
plt.scatter(VAtimes / 3600, np.mean(np.abs(VAresiduals_real), axis=0), label='Real Harmonic Averaged', marker='o')
plt.scatter(VAtimes / 3600, np.mean(np.abs(VAresiduals_imag), axis=0), label='Imag Harmonic Averaged', marker='o')

plt.xlabel('Computation Time (hr)')
plt.ylabel('Abs Impedance Percent Residuals')
plt.title('Impedance Residuals at (x,y) = (0,0) vs Time, Averaged Across Freq')
plt.legend()
plt.grid()
plt.savefig(f"figure_out/cellsize_tradeoff/centerResidualsTime.png")



# ======================
# Center residuals vs cells, all freqs
# ======================

plt.figure(figsize=(7, 7))

NAresiduals_real = np.array([100*(NAdata[:, f, 0, 1012]-Adata[f, 22, 22, 0])/Adata[f, 22, 22, 0] for f in np.arange(71)])
NAresiduals_imag = np.array([100*(NAdata[:, f, 1, 1012]-Adata[f, 22, 22, 1])/Adata[f, 22, 22, 1] for f in np.arange(71)])
VAresiduals_real = np.array([100*(VAdata[:, f, 0, 1012]-Adata[f, 22, 22, 0])/Adata[f, 22, 22, 0] for f in np.arange(71)])
VAresiduals_imag = np.array([100*(VAdata[:, f, 1, 1012]-Adata[f, 22, 22, 1])/Adata[f, 22, 22, 1] for f in np.arange(71)])

plt.scatter(NAcells, np.mean(np.abs(NAresiduals_real), axis=0), label='Real Normal Procedure', marker='x', s=100)
plt.scatter(NAcells, np.mean(np.abs(NAresiduals_imag), axis=0), label='Imag Normal Procedure', marker='x', s=100)
plt.scatter(VAcells, np.mean(np.abs(VAresiduals_real), axis=0), label='Real Harmonic Averaged', marker='o', s=100)
plt.scatter(VAcells, np.mean(np.abs(VAresiduals_imag), axis=0), label='Imag Harmonic Averaged', marker='o', s=100)

plt.xlabel('Cell Count', fontsize=20)
plt.ylabel('Abs Impedance Percent Residuals', fontsize=20)
plt.tick_params(axis='both', labelsize=16) 
plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0), useMathText=True)
plt.gca().xaxis.get_offset_text().set_fontsize(16) 
plt.title('Impedance Residuals at (x,y) = (0,0)\nvs Cell Count, Averaged Across Freq', fontsize=22)
plt.legend(fontsize=15)
plt.grid()
plt.savefig(f"figure_out/cellsize_tradeoff/centerResidualsCells.png", bbox_inches='tight')


# ======================
# Center residuals vs RAM, all freqs
# ======================

plt.figure()

NAresiduals_real = np.array([100*(NAdata[:, f, 0, 1012]-Adata[f, 22, 22, 0])/Adata[f, 22, 22, 0] for f in np.arange(71)])
NAresiduals_imag = np.array([100*(NAdata[:, f, 1, 1012]-Adata[f, 22, 22, 1])/Adata[f, 22, 22, 1] for f in np.arange(71)])
VAresiduals_real = np.array([100*(VAdata[:, f, 0, 1012]-Adata[f, 22, 22, 0])/Adata[f, 22, 22, 0] for f in np.arange(71)])
VAresiduals_imag = np.array([100*(VAdata[:, f, 1, 1012]-Adata[f, 22, 22, 1])/Adata[f, 22, 22, 1] for f in np.arange(71)])

plt.scatter(NAram, np.mean(np.abs(NAresiduals_real), axis=0), label='Real Normal Procedure', marker='x')
plt.scatter(NAram, np.mean(np.abs(NAresiduals_imag), axis=0), label='Imag Normal Procedure', marker='x')
plt.scatter(VAram, np.mean(np.abs(VAresiduals_real), axis=0), label='Real Harmonic Averaged', marker='o')
plt.scatter(VAram, np.mean(np.abs(VAresiduals_imag), axis=0), label='Imag Harmonic Averaged', marker='o')

plt.xlabel('Maximum Memory Used (Gb)')
plt.ylabel('Abs Impedance Percent Residuals')
plt.title('Impedance Residuals at (x,y) = (0,0) vs Memory Usage, Averaged Across Freq')
plt.legend()
plt.grid()
plt.savefig(f"figure_out/cellsize_tradeoff/centerResidualsMemory.png")