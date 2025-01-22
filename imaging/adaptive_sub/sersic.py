"""
Simulator: Sersic
=================

This script simulates `Imaging` of a galaxy using light profiles where:

 - The galaxy's bulge is an `Sersic`.
 - The galaxy's disk is an `Exponential`.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import numpy as np
from os import path
import time
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__
"""
dataset_path = path.join("imaging", "adaptive_sub", "plots")

"""
__Galaxies__
"""
galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.1,
        sersic_index=3.0,
    ),
)

"""
__Images__
"""
grid = ag.Grid2D.uniform(shape_native=(100, 100), pixel_scales=0.01, sub_size=4)

time_normal = time.time()

image = galaxy.image_2d_from(grid=grid).binned

time_normal = time.time() - time_normal

time_adapt = time.time()

grid = ag.Grid2DIterate.lp(
    shape_native=grid.shape_native,
    pixel_scales=grid.pixel_scales,
    fractional_accuracy=0.9999,
    sub_steps=[2, 4, 8, 16],
)

image_adapt = galaxy.image_2d_from(grid=grid)

time_adapt = time.time() - time_normal

residuals = image_adapt - image

residuals_relative = abs(residuals) / image

print(f"Residual Max = {np.max(residuals)}")
print(f"Residual Relative Max = {np.max(residuals_relative)}")

plotter = aplt.Array2DPlotter(
    array=residuals,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(filename="residuals", path=dataset_path, format="png")
    ),
)
plotter.figure_2d()

plotter = aplt.Array2DPlotter(
    array=residuals_relative,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            filename="residuals_relative", path=dataset_path, format="png"
        )
    ),
)
plotter.figure_2d()

"""
The dataset can be viewed in the folder `autogalaxy_workspace/imaging/light_sersic_exp`.
"""
