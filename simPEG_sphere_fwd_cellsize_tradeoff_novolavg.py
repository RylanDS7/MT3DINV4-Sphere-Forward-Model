# Code by Rylan Stutters - github.com/RylanDS7

# SimPEG functionality
from simpeg import maps
from simpeg.electromagnetics import natural_source as nsem
from simpeg.utils import model_builder
from pymatsolver import Pardiso

# discretize functionality
from discretize import TreeMesh, TensorMesh
from discretize.utils import mkvc, active_from_xyz

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import time
import argparse


def volume_average_sphere(center, radius, mesh, sphere_sigma, background_sigma, model):
    nodes = mesh.total_nodes
    cells = mesh.cell_nodes
    cell_vols = mesh.cell_volumes
    cell_centers = mesh.cell_centers

    def in_sphere(cell):
        truth = []
        for node in cell:
            node_loc = nodes[node]
            if np.linalg.norm(node_loc - center) > radius:
                truth.append(False)
            else:
                truth.append(True)
        return np.array(truth)
    

    def cell_fill_fraction(cell_index):
        subsamples=10
        cell_center = cell_centers[cell_index]
        width = mesh[cell_index].h[0]

        xs = np.linspace(cell_center[0] - width/2, cell_center[0] + width/2, subsamples)
        ys = np.linspace(cell_center[1] - width/2, cell_center[1] + width/2, subsamples)
        zs = np.linspace(cell_center[2] - width/2, cell_center[2] + width/2, subsamples)

        xx, yy, zz = np.meshgrid(xs, ys, zs)

        sample_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        sphere_count = 0
        for pt in sample_points:
            if np.linalg.norm(pt - center) <= radius:
                sphere_count += 1

        return sphere_count / (subsamples)**3
    

    for i, cell in enumerate(cells):
        if not any(in_sphere(cell)):
            continue
        elif all(in_sphere(cell)):
            model[i] = sphere_sigma
        else:
            f = cell_fill_fraction(i)
            sigma = 1 / ((1-f) * (1 / background_sigma) + f * (1 / sphere_sigma))
            model[i] = sigma



def main(cell_size):

    run_num = f"{cell_size}"

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

    # ======================================
    # SETUP MESH
    # ======================================

    dh = cell_size # fine cell size

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
        levels=-6,
        finalize=False
    )

    # Fine refinement around the sphere
    mesh.refine_ball(
        [0,0,-1000],
        750,
        levels=-4,
        finalize=False
    )

    # Fine refinement near receivers
    refine_pts = np.zeros((len(rx_locs), 3))
    for i, pt in enumerate(rx_locs):
        refine_pts[i] = [pt[0], pt[1], 0]
    mesh.refine_points(refine_pts, padding_cells_by_level=[3, 2, 2], finalize=False)

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

    # CHECKPOINT
    # fig = plt.figure(figsize=(20, 12))
    # ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # out = mesh.plot_slice(
    #     conductivity_model,
    #     ax=ax1,
    #     normal="Y",
    #     ind=int(len(mesh.h[1]) / 2),
    #     grid=True,
    #     grid_opts={
    #         "color": "black", 
    #         "linewidth": 0.5,
    #         "alpha": 0.3
    #     },
    #     pcolor_opts={
    #         "cmap": "viridis",
    #         "norm": LogNorm(vmin=1e-8, vmax=10)
    #     }
    # )

    # cb = plt.colorbar(out[0], ax=ax1, orientation='vertical')
    # cb.set_label('Conductivity (S/m)')

    # plot a zoomed in cross section
    # ax1.set_xlim([rx_locs[:, 0].min()/3, rx_locs[:, 0].max()/3])
    # ax1.set_ylim([-2000, 200]) # zoom in around the sphere
    # plt.title(f"Conductivity Model Cross Section at y=0, {mesh.nC} cells")
    # plt.show()

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

    for f in freqs: # running on reduced freqs
        rx_list = []
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='xy', component='real'))
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='xy', component='imag'))
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='xy', component='apparent_resistivity'))
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='xy', component='phase'))
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='yx', component='real'))
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='yx', component='imag'))
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='yx', component='apparent_resistivity'))
        rx_list.append(nsem.receivers.Impedance(locations_e=rx_locs, locations_h=rx_locs, orientation='yx', component='phase'))
        source_list.append(nsem.sources.FictitiousSource(rx_list, frequency=f))

    survey = nsem.survey.Survey(source_list)


    # ======================================
    # SETUP SIMULATION
    # ======================================

    mesh_1d = TensorMesh([hz], origin=np.array([mesh.origin[-1]]))
    sigma_1d = sigma_air * np.ones(mesh_1d.n_cells)
    sigma_1d[mesh_1d.cell_centers < 0.] = background_conductivity

    sim = nsem.simulation.Simulation3DElectricFieldFictitious(
        mesh,
        survey=survey,
        sigmaMap=maps.IdentityMap(mesh),
        sigma_background=sigma_1d,
        forward_only=True,
        solver=Pardiso
    )


    # ======================================
    # RUN SIMULATION AND SAVE RESULTS
    # ======================================

    print(f"{run_num} Running Forward Simulation")
    start_time = time.time()
    dpred = sim.dpred(conductivity_model)
    end_time = time.time()
    sim_time = end_time - start_time
    print(f"{run_num} Finished Forward Simulation in {sim_time:.4f} seconds")
    print(f"{run_num} Expected data shape: {len(freqs)} x {len(rx_locs)} x 8 = {len(freqs) * len(rx_locs) * 8}") # for reduced freqs
    print(f"{run_num} Survey data shape:", dpred.shape)

    
    outdir = "/scratch/rstutter"

    np.save(f'{outdir}/dpred{run_num}.npy', dpred)
    np.save(f'{outdir}/freqs{run_num}.npy', freqs)
    np.save(f'{outdir}/rx_locs{run_num}.npy', rx_locs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_size", type=int, required=True)
    args = parser.parse_args()

    main(args.cell_size)