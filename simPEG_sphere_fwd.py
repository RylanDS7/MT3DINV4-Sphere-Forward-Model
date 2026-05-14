# Code by Rylan Stutters - github.com/RylanDS7

# SimPEG functionality
from simpeg import maps, utils, data, optimization, maps, regularization, directives
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import model_builder

# discretize functionality
from discretize import TreeMesh
from discretize.utils import mkvc, active_from_xyz

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


# ======================================
# Define receiver locations
# ======================================

# receiver locations
x_positions = [
    -5000, -2000, -1500, -1000, -750, -500, -400, -300, -250, -200,
    -150, -125, -100, -90, -80, -70, -60, -50, -40, -30,
    -20, -10, 0, 10, 20, 30, 40, 50, 60, 70,
    80, 90, 100, 125, 150, 200, 250, 300, 400, 500,
    750, 1000, 1500, 2000, 5000
]
y_positions = [
    -5000, -2000, -1500, -1000, -750, -500, -400, -300, -250, -200,
    -150, -125, -100, -90, -80, -70, -60, -50, -40, -30,
    -20, -10, 0, 10, 20, 30, 40, 50, 60, 70,
    80, 90, 100, 125, 150, 200, 250, 300, 400, 500,
    750, 1000, 1500, 2000, 5000
]

rx_locs = []

for x in x_positions:
    for y in y_positions:
        rx_locs.append([x, y, 0])
rx_locs = np.array(rx_locs)

# CHECKPOINT
plt.scatter(rx_locs[:,0], rx_locs[:,1])
plt.show()


# ======================================
# SETUP MESH
# ======================================

dh = 50  # fine cell size

# Skin depth at 0.001 Hz ~ 500 km, use 5x = 2500 km
dom_width_x = 500000.0  # 500 km
dom_width_y = 500000.0  # 500 km
dom_width_z = 500000.0  # 500 km

nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
nbcy = 2 ** int(np.round(np.log(dom_width_y / dh) / np.log(2.0)))
nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))

hx = [(dh, nbcx)]
hy = [(dh, nbcy)]
hz = [(dh, nbcz)]
mesh = TreeMesh([hx, hy, hz], x0="CCN", diagonal_balance=True)

# Coarse refinement over the whole domain first
mesh.refine_box(
    [-250000, -250000, -250000],
    [250000, 250000, 0],
    levels=2,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-10000, -10000, -5000],
    [10000, 10000, 0],
    levels=4,
    finalize=False
)

# Fine refinement near receivers
refine_pts = np.zeros((len(rx_locs), 3))
for i, pt in enumerate(rx_locs):
    refine_pts[i] = [pt[0], pt[1], 0]
mesh.refine_points(refine_pts, padding_cells_by_level=[2, 1], finalize=False)

# Fine refinement around sphere
mesh.refine_points([[0, 0, -1000]], padding_cells_by_level=[4, 3, 2, 1], finalize=False)

mesh.finalize()

# CHECKPOINT
print(f"Mesh cells: {mesh.nC:,}")
print(f"Mesh x extent: {mesh.nodes_x[[0,-1]]/1000} km")
print(f"Mesh z extent: {mesh.nodes_z[[0,-1]]/1000} km")


# ======================================
# SETUP MODEL
# ======================================

background_conductivity = 0.001
sphere_conductivity = 10

background_model = background_conductivity * np.ones(mesh.nC)

sphere_indices = model_builder.get_indices_sphere(center=[0,0,-1000], radius=500, cell_centers=mesh.cell_centers)
conductivity_model = background_conductivity * np.ones(mesh.nC)
conductivity_model[sphere_indices] = sphere_conductivity

# CHECKPOINT
fig = plt.figure(figsize=(10, 4.5))
ax1 = fig.add_axes([0.15, 0.15, 0.68, 0.75])
out = mesh.plot_slice(
    conductivity_model,
    ax=ax1,
    normal="Y",
    ind=int(len(mesh.h[1]) / 2),
    grid=True,
    pcolor_opts={"cmap": "viridis"}
)

cb = plt.colorbar(out[0], ax=ax1, orientation='vertical')
cb.set_label('Conductivity (S/m)')

# plot a zoomed in cross section
ax1.set_xlim(mesh.nodes_x[[0,-1]]/20)
ax1.set_ylim(mesh.nodes_z[[0,-1]]/40)
plt.show()

# ======================================
# SETUP FREQUENCIES AND SURVEY
# ======================================

low_freq_order = -3 # 1mHz
high_freq_order = 4 # 10kHz
samples_per_dec = 10

freqs = np.logspace(low_freq_order, 
                    high_freq_order, 
                    samples_per_dec*(high_freq_order-low_freq_order)+1)

# Data structued as freq x dataType x rx
source_list = []

rx_list = []
rx_list.append(nsem.receivers.Impedance(rx_locs, orientation='xy', component='apparent_resistivity'))
rx_list.append(nsem.receivers.Impedance(rx_locs, orientation='xy', component='phase'))
rx_list.append(nsem.receivers.Impedance(rx_locs, orientation='yx', component='apparent_resistivity'))
rx_list.append(nsem.receivers.Impedance(rx_locs, orientation='yx', component='phase'))

for f in freqs:
    source_list.append(nsem.sources.PlanewaveXYPrimary(rx_list, frequency=f, sigma_primary=background_model))

survey = nsem.survey.Survey(source_list)


# ======================================
# SETUP SIMULATION
# ======================================

sim = nsem.Simulation3DPrimarySecondary(
    mesh,
    survey=survey,
    sigmaMap=maps.IdentityMap(mesh),
    sigmaPrimary=background_model,
    forward_only=True
)


# ======================================
# RUN SIMULATION AND SAVE RESULTS
# ======================================

print("Running Forward Simulation")
dpred = sim.dpred(conductivity_model)
print("Expected data shape:", len(freqs) * len(rx_locs) * 4)
print("Survey data shape:", dpred.shape)

np.save('dpred.npy', dpred)
np.save('freqs.npy', freqs)
np.save('rx_locs.npy', rx_locs)