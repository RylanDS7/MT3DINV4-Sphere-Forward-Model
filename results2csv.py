# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import csv

run_num = "010"
data_path = f"simPEG_data/{run_num}/"
csv_file = f"simPEG_data/{run_num}/forward_response.csv"

# load and parse simPEG data
dpred = np.load(data_path+'dpredVA.npy')
freqs = np.load(data_path+'freqsVA.npy')
rx_locs = np.load(data_path+'rx_locsVA.npy')

SI_TO_FIELD = 795.775  # Ohm -> mV/km/nT

data = SI_TO_FIELD * dpred.reshape(len(freqs), 8, len(rx_locs))
data[:, 3, :] += 180 # app resistivity phase quadrant correction
data[:, 0, :] = -data[:, 0, :]
data[:, 1, :] = -data[:, 1, :]

SI_TO_FIELD = 795.775  # Ohm -> mV/km/nT

with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([len(rx_locs), 2, len(freqs)])
    writer.writerow(["ZXY", "ZYX"])
    writer.writerow(np.flip(freqs))

    for i, rx in enumerate(rx_locs):
        writer.writerow([rx[0], rx[1], "ZXYR"] + np.flip(data[:, 0, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZXYI"] + np.flip(data[:, 1, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZYXR"] + np.flip(data[:, 4, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZYXI"] + np.flip(data[:, 5, i]).tolist())

