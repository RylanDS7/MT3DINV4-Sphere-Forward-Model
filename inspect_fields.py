# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import scipy.constants
import matplotlib as mpl
import matplotlib.pyplot as plt
import discretize

data_path = 'simPEG_data/fields/'

with np.load(f'{data_path}model_fields.npz', allow_pickle=True) as data:
    dpred = data['dpred']
    freqs = data['freqs']
    rx_locs = data['rx_locs']
    fields = data['fields'].item()
    mesh = data['mesh'].item()

mesh = discretize.TreeMesh.deserialize(mesh)

data = dpred
data[:, 3, :] += 180 # app resistivity phase quadrant correction
data[:, 0, :] = -data[:, 0, :]
data[:, 1, :] = -data[:, 1, :]

e = fields['e']
e_xy = e[:, 0]
e_yx = e[:, 1]

# --- build a horizontal grid at z=0 ---
x = np.linspace(-5000, 5000, 20)  
y = np.linspace(-5000, 5000, 20)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
locs = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

# --- interpolation matrices (built once, reused for all 4 panels) ---
Pex = mesh.get_interpolation_matrix(locs, location_type='Ex')
Pey = mesh.get_interpolation_matrix(locs, location_type='Ey')

# --- interpolate each polarization onto the grid ---
Ex_xy = (Pex @ e_xy).reshape(X.shape)
Ey_yx = (Pey @ e_yx).reshape(X.shape)

# --- 2x2 panel setup: rows = polarization (xy, yx), cols = component (real, imag) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

panels = [
    (axes[0, 0], Ex_xy.real, 'E_x — real'),
    (axes[0, 1], Ex_xy.imag, 'E_x — imag'),
    (axes[1, 0], Ey_yx.real, 'E_y — real'),
    (axes[1, 1], Ey_yx.imag, 'E_y — imag'),
]

for ax, Ec, title in panels:
    sc = ax.scatter(X.ravel(), Y.ravel(), c=Ec.ravel(), cmap='viridis', s=20)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax, label='E (component)')
    cbar.formatter.set_useOffset(False)
    cbar.update_normal()

fig.suptitle('E-field scalar maps at z=0, f=100 Hz', fontsize=14)
plt.tight_layout()
plt.show()

E_avgs = {}
for ax, Ec, title in panels:
    E_avgs[title] = Ec.mean()

for title in E_avgs.keys():
    print(f"Average {title}: {E_avgs[title]}")

print(f"E_x Phase: {180 + 180 / np.pi * np.arctan(E_avgs['E_x — imag'] / E_avgs['E_x — real'])}")
print(f"E_y Phase: {180 + 180 / np.pi * np.arctan(E_avgs['E_y — imag'] / E_avgs['E_y — real'])}")


h = fields['h']
h_xy = h[:, 0]
h_yx = h[:, 1]

# --- build a horizontal grid at z=0 ---
x = np.linspace(-5000, 5000, 20)   
y = np.linspace(-5000, 5000, 20)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
locs = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

# --- interpolation matrices (built once, reused for all 4 panels) ---
Phx = mesh.get_interpolation_matrix(locs, location_type='Fx')
Phy = mesh.get_interpolation_matrix(locs, location_type='Fy')

# --- interpolate each polarization onto the grid ---
Hy_xy = (Phy @ h_xy).reshape(X.shape)
Hx_yx = (Phx @ h_yx).reshape(X.shape)

# --- 2x2 panel setup: rows = polarization (xy, yx), cols = component (real, imag) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 14))

panels = [
    (axes[0, 0], Hy_xy.real, 'H_x — real'),
    (axes[0, 1], Hy_xy.imag, 'H_x — imag'),
    (axes[1, 0], Hx_yx.real, 'H_y — real'),
    (axes[1, 1], Hx_yx.imag, 'H_y — imag'),
]

for ax, Hc, title in panels:
    sc = ax.scatter(X.ravel(), Y.ravel(), c=Hc.ravel(), cmap='viridis', s=20)
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_aspect('equal')
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label='H (component)')

fig.suptitle('H-field scalar maps at z=0, f=100 Hz', fontsize=14)
plt.tight_layout()
plt.show()


H_avgs = {}
for ax, Hc, title in panels:
    H_avgs[title] = Hc.mean()

for title in H_avgs.keys():
    print(f"Average {title}: {H_avgs[title]}")

print(f"H_x Phase: {180 / np.pi * np.arctan(H_avgs['H_x — imag'] / H_avgs['H_x — real'])}")
print(f"H_y Phase: {180 / np.pi * np.arctan(H_avgs['H_y — imag'] / H_avgs['H_y — real'])}")

print(f"\nPhase Difference: {180 + 180 / np.pi * np.arctan(E_avgs['E_y — imag'] / E_avgs['E_y — real']) - 180 / np.pi * np.arctan(H_avgs['H_y — imag'] / H_avgs['H_y — real'])}")

print(f"\nRe(E_x)*Re(H_y) + Im(E_x)*Im(H_y): {E_avgs['E_x — real'] * H_avgs['H_y — real'] + E_avgs['E_x — imag'] * H_avgs['H_y — imag']}")
print(f"Re(H_y)*Im(E_x) - Re(E_x)*Im(H_y): {H_avgs['H_y — real'] * E_avgs['E_x — imag'] - E_avgs['E_x — real'] * H_avgs['H_y — imag']}")

mag_Ex = np.sqrt(E_avgs['E_x — real']**2 + E_avgs['E_x — imag']**2)
mag_Hy = np.sqrt(H_avgs['H_y — real']**2 + H_avgs['H_y — imag']**2)
print(f"\nSimulated E_x Magnitude: {mag_Ex}")
print(f"Simulated H_y Magnitude: {mag_Hy}")
print(f"Simulated Z_xy Magnitude: {mag_Ex / mag_Hy}")
print(f"Predicted Z_xy Magnitude: {np.sqrt((100 * 2 * np.pi * scipy.constants.mu_0) / (0.001))}")
