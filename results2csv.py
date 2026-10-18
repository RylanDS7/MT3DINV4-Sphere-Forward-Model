# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import csv

run_num = "013"
data_path = f"simPEG_data/{run_num}/"
csv_file = f"simPEG_data/{run_num}/137_FWrsp.dat"

# load and parse simPEG data
dpred = np.load(data_path+'dpredVA.npy')
freqs = np.load(data_path+'freqsVA.npy')
rx_locs = np.load(data_path+'rx_locsVA.npy')


dpred = dpred.reshape(len(freqs), 20, len(rx_locs))
data = np.zeros_like(dpred)

data[:, 0, :] = dpred[:, 6, :] # Zxx -> Zyy
data[:, 1, :] = dpred[:, 7, :]

data[:, 2, :] = dpred[:, 4, :] # Zxy -> Zyx
data[:, 3, :] = dpred[:, 5, :]

data[:, 4, :] = dpred[:, 2, :] # Zyx -> Zxy
data[:, 5, :] = dpred[:, 3, :]

data[:, 6, :] = dpred[:, 0, :] # Zyy -> Zxx
data[:, 7, :] = dpred[:, 1, :]

data[:, 8, :] = -dpred[:, 10, :] # Tzx -> -Tzy
data[:, 9, :] = -dpred[:, 11, :]

data[:, 10, :] = -dpred[:, 8, :] # Tzy -> -Tzx
data[:, 11, :] = -dpred[:, 9, :]

data[:, 12, :] = dpred[:, 15, :] # rhoxx -> rhoyy
data[:, 13, :] = dpred[:, 14, :] # rhoxy -> rhoyx
data[:, 14, :] = dpred[:, 13, :] # rhoyx -> rhoxy
data[:, 15, :] = dpred[:, 12, :] # rhoyy -> rhoxx

data[:, 16, :] = dpred[:, 19, :] # phasexx -> phaseyy
data[:, 17, :] = dpred[:, 18, :] # phasexy -> phaseyx
data[:, 18, :] = dpred[:, 17, :] # phaseyx -> phasexy
data[:, 19, :] = dpred[:, 16, :] # phaseyy -> phasexx

data = data.reshape(len(freqs), 20, 15, 15) # reshape for xy rx locs
data = data.transpose(0, 1, 3, 2)   # rotate coordinates into z down convention

# Old rx_locs: columns [east, north, up], index = ix_east*15 + iy_north
rx_grid = rx_locs.reshape(15, 15, 3)        # [east_idx, north_idx, (e, n, up)]
rx_grid = rx_grid.transpose(1, 0, 2)          # [north_idx, east_idx, (e, n, up)]  convert to new frame

# Reorder rx indexing into new frame
# New frame: x' = north, y' = east, z' = down
rx_ned = np.stack([rx_grid[..., 1],           # x' = north
                   rx_grid[..., 0],           # y' = east
                   -rx_grid[..., 2]], axis=-1).reshape(-1, 3)   # (225, 3), i = ix'*15 + iy'

# Flatten data
data_flat = data.reshape(len(freqs), 20, -1)

rows = [
    ("ZXXR", 0),   ("ZXXI", 1),
    ("ZXYR", 2),   ("ZXYI", 3),
    ("ZYXR", 4),   ("ZYXI", 5),
    ("ZYYR", 6),   ("ZYYI", 7),
    ("TZXR", 8),   ("TZXI", 9),
    ("TZYR", 10),  ("TZYI", 11),
    ("rhoXX", 12), ("rhoXY", 13), ("rhoYX", 14), ("rhoYY", 15),
    ("phiXX", 16), ("phiXY", 17), ("phiYX", 18), ("phiYY", 19),
]

with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)

    writer.writerow([len(rx_ned), 14, len(freqs)])
    writer.writerow(["ZXX", "ZXY", "ZYX", "ZYY", "TZX", "TZY",
                     "rhoXX", "rhoXY", "rhoYX", "rhoYY",
                     "phiXX", "phiXY", "phiYX", "phiYY"])
    writer.writerow(freqs.tolist())   # descending freqs

    for i, rx in enumerate(rx_ned):
        for label, comp in rows:
            writer.writerow([rx[0], rx[1], label] + data_flat[:, comp, i].tolist())
