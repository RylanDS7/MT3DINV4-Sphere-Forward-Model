# Code by Rylan Stutters - github.com/RylanDS7

import numpy as np
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from discretize import TreeMesh
from simpeg.utils import model_builder

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


meshes = []
times = []

# ======================================
# Mesh 1
# ======================================

dh = 20 # fine cell size

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
mesh = TreeMesh([hx, hy, hz], x0="CCC", diagonal_balance=True)

# Coarse refinement over the whole domain first
mesh.refine_box(
    [-250000, -250000, -250000],
    [250000, 250000, 0],
    levels=4,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-10000, -10000, -5000],
    [10000, 10000, 0],
    levels=11,
    finalize=False
)

# Finer refinement within the rxs area
mesh.refine_box(
    [-5000, -5000, -2000],
    [5000, 5000, 0],
    levels=10,
    finalize=False
)

# Fine refinement around the sphere
mesh.refine_ball(
    [0,0,-1000],
    1000,
    levels=-1,
    finalize=False
)

# Fine refinement near receivers
refine_pts = np.zeros((len(rx_locs), 3))
for i, pt in enumerate(rx_locs):
    refine_pts[i] = [pt[0], pt[1], 0]
mesh.refine_points(refine_pts, padding_cells_by_level=[2, 1], finalize=False)

mesh.finalize()

meshes.append(mesh)
times.append(31929)

# ======================================
# Mesh 2
# ======================================

dh = 10 # fine cell size

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
mesh = TreeMesh([hx, hy, hz], x0="CCC", diagonal_balance=True)

# Coarse refinement over the whole domain first
mesh.refine_box(
    [-250000, -250000, -250000],
    [250000, 250000, 0],
    levels=4,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-10000, -10000, -5000],
    [10000, 10000, 0],
    levels=7,
    finalize=False
)

# Finer refinement within the rxs area
mesh.refine_box(
    [-5000, -5000, -2000],
    [5000, 5000, 0],
    levels=10,
    finalize=False
)

# Finer refinement around the rxs surface
mesh.refine_box(
    [-5000, -5000, -50],
    [5000, 5000, 0],
    levels=-3,
    finalize=False
)

# Fine refinement around the sphere
mesh.refine_ball(
    [0,0,-1000],
    500,
    levels=-2,
    finalize=False
)

# Fine refinement near receivers
refine_pts = np.zeros((len(rx_locs), 3))
for i, pt in enumerate(rx_locs):
    refine_pts[i] = [pt[0], pt[1], 0]
mesh.refine_points(refine_pts, padding_cells_by_level=[3, 2, 1], finalize=False)

mesh.finalize()

meshes.append(mesh)
times.append(2051)

# ======================================
# Mesh 3
# ======================================

dh = 25 # fine cell size

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
mesh = TreeMesh([hx, hy, hz], x0="CCC", diagonal_balance=True)

# Coarse refinement over the whole domain first
mesh.refine_box(
    [-250000, -250000, -250000],
    [250000, 250000, 0],
    levels=4,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-10000, -10000, -5000],
    [10000, 10000, 0],
    levels=7,
    finalize=False
)

# Finer refinement within the rxs area
mesh.refine_box(
    [-5000, -5000, -2000],
    [5000, 5000, 0],
    levels=10,
    finalize=False
)

# Finer refinement around the rxs surface
mesh.refine_box(
    [-5000, -5000, -25],
    [5000, 5000, 0],
    levels=-1,
    finalize=False
)

# Fine refinement around the sphere
mesh.refine_ball(
    [0,0,-1000],
    500,
    levels=-1,
    finalize=False
)

mesh.finalize()

meshes.append(mesh)
times.append(1836)

# ======================================
# Mesh 4
# ======================================

dh = 25 # fine cell size

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
mesh = TreeMesh([hx, hy, hz], x0="CCC", diagonal_balance=True)

# Coarse refinement over the whole domain first
mesh.refine_box(
    [-250000, -250000, -250000],
    [250000, 250000, 0],
    levels=4,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-10000, -10000, -5000],
    [10000, 10000, 0],
    levels=7,
    finalize=False
)

# Finer refinement within the rxs area
mesh.refine_box(
    [-5000, -5000, -2000],
    [5000, 5000, 0],
    levels=10,
    finalize=False
)

# Finer refinement around the rxs surface
mesh.refine_box(
    [-5000, -5000, -50],
    [5000, 5000, 0],
    levels=-1,
    finalize=False
)

# Fine refinement around the sphere
mesh.refine_ball(
    [0,0,-1000],
    500,
    levels=-1,
    finalize=False
)

mesh.finalize()

meshes.append(mesh)
times.append(6807)

# ======================================
# Mesh 5
# ======================================

dh = 25 # fine cell size

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
mesh = TreeMesh([hx, hy, hz], x0="CCC", diagonal_balance=True)

# Coarse refinement over the whole domain first
mesh.refine_box(
    [-250000, -250000, -250000],
    [250000, 250000, 0],
    levels=4,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-10000, -10000, -5000],
    [10000, 10000, 0],
    levels=7,
    finalize=False
)

# Finer refinement within the rxs area
mesh.refine_box(
    [-5000, -5000, -2000],
    [5000, 5000, 0],
    levels=10,
    finalize=False
)

# Finer refinement around the rxs surface
mesh.refine_box(
    [-5000, -5000, -100],
    [5000, 5000, 0],
    levels=-1,
    finalize=False
)

# Fine refinement around the sphere
mesh.refine_ball(
    [0,0,-1000],
    750,
    levels=-1,
    finalize=False
)

mesh.finalize()

meshes.append(mesh)
times.append(10420)

# ======================================
# Mesh 6
# ======================================

dh = 5 # fine cell size

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
mesh = TreeMesh([hx, hy, hz], x0="CCC", diagonal_balance=True)

# Coarse refinement over the whole domain first
mesh.refine_box(
    [-250000, -250000, -250000],
    [250000, 250000, 0],
    levels=4,
    finalize=False
)

# Medium refinement in the core region
mesh.refine_box(
    [-10000, -10000, -5000],
    [10000, 10000, 0],
    levels=7,
    finalize=False
)

# Finer refinement within the rxs area
mesh.refine_box(
    [-5000, -5000, -2000],
    [5000, 5000, 0],
    levels=10,
    finalize=False
)

# Finer refinement around the rxs surface
mesh.refine_box(
    [-5000, -5000, -50],
    [5000, 5000, 0],
    levels=-4,
    finalize=False
)

# Fine refinement around the sphere
mesh.refine_ball(
    [0,0,-1000],
    750,
    levels=-3,
    finalize=False
)

# Fine refinement near receivers
refine_pts = np.zeros((len(rx_locs), 3))
for i, pt in enumerate(rx_locs):
    refine_pts[i] = [pt[0], pt[1], 0]
mesh.refine_points(refine_pts, padding_cells_by_level=[6, 2, 2], finalize=False)

mesh.finalize()

meshes.append(mesh)
times.append(10790)


# ======================================
# Plot Meshes
# ======================================

with PdfPages("figure_out/meshes.pdf") as pdf:
    for i, mesh in enumerate(meshes):
        background_conductivity = 0.001
        sphere_conductivity = 10
        sigma_air = 1e-8

        conductivity_model = sigma_air * np.ones(mesh.nC)

        earth_inds = mesh.cell_centers[:,2] < 0
        conductivity_model[earth_inds] = background_conductivity

        sphere_indices = model_builder.get_indices_sphere(
            center=[0,0,-1000],
            radius=500,
            cell_centers=mesh.cell_centers
        )

        conductivity_model[sphere_indices] = sphere_conductivity

        background_model = sigma_air * np.ones(mesh.nC)
        background_model[earth_inds] = background_conductivity

        fig = plt.figure(figsize=(20, 12))
        ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        out = mesh.plot_slice(
            conductivity_model,
            ax=ax1,
            normal="Y",
            ind=int(len(mesh.h[1]) / 2),
            grid=True,
            grid_opts={
                "color": "black", 
                "linewidth": 0.5,
                "alpha": 0.3
            },
            pcolor_opts={
                "cmap": "viridis",
                "norm": LogNorm(vmin=1e-8, vmax=10)
            }
        )

        cb = plt.colorbar(out[0], ax=ax1, orientation='vertical')
        cb.set_label('Conductivity (S/m)')

        # plot a zoomed in cross section
        ax1.set_xlim([rx_locs[:, 0].min()/3, rx_locs[:, 0].max()/3])
        ax1.set_ylim([-2000, 200]) # zoom in around the sphere
        plt.title(f"Mesh {i+1}, Simulated in {times[i]/3600:.2f} hours \n Conductivity Model Cross Section at y=0, {mesh.nC:,.0f} cells")
        
        pdf.savefig(fig)
        plt.savefig(f"figure_out/{(i+1):03}/mesh.png")
        plt.close(fig)

