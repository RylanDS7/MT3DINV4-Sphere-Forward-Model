# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import csv

run_num = "012"
data_path = f"simPEG_data/{run_num}/"
csv_file = f"simPEG_data/{run_num}/forward_response.csv"

# load and parse simPEG data
dpred = np.load(data_path+'dpredVA.npy')
freqs = np.load(data_path+'freqsVA.npy')
rx_locs = np.load(data_path+'rx_locsVA.npy')

SI_TO_FIELD = 795.775  # Ohm -> mV/km/nT

data = -SI_TO_FIELD * dpred.reshape(len(freqs), 12, len(rx_locs)) # negative sign for convention shift

with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([len(rx_locs), 6, len(freqs)])
    writer.writerow(["ZXX", "ZXY", "ZYX", "ZYY", "TZX", "TZY"])
    writer.writerow(np.flip(freqs)) # descending freqs

    for i, rx in enumerate(rx_locs):
        writer.writerow([rx[0], rx[1], "ZXXR"] + np.flip(data[:, 0, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZXXI"] + np.flip(data[:, 1, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZXYR"] + np.flip(data[:, 2, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZXYI"] + np.flip(data[:, 3, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZYXR"] + np.flip(data[:, 4, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZYXI"] + np.flip(data[:, 5, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZYYR"] + np.flip(data[:, 6, i]).tolist())
        writer.writerow([rx[0], rx[1], "ZYYI"] + np.flip(data[:, 7, i]).tolist())
        writer.writerow([rx[0], rx[1], "TZXR"] + np.flip(data[:, 8, i]).tolist())
        writer.writerow([rx[0], rx[1], "TZXI"] + np.flip(data[:, 9, i]).tolist())
        writer.writerow([rx[0], rx[1], "TZYR"] + np.flip(data[:, 10, i]).tolist())
        writer.writerow([rx[0], rx[1], "TZYI"] + np.flip(data[:, 11, i]).tolist())

