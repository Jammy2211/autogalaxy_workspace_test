"""
Modeling: Light Parametric
==========================

This script fits an `Imaging` dataset of a galaxy with a model where:

 - The galaxy's light is a parametric `Sersic` bulge and `Exponential` disk.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

cmap = aplt.Cmap(cmap="jet", norm="linear", vmin=0.0, vmax=0.1)

mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

"""
__Dataset__

Load and plot the galaxy dataset `light_sersic_exp` via .fits files, which we will fit with 
the model.
"""
dataset_path = path.join("paper", "image")

image_2d = ag.Array2D.from_fits(
    file_path=path.join(dataset_path, "image_hdf_example.fits"), pixel_scales=0.03
)

model_image_2d = ag.Array2D.from_fits(
    file_path=path.join(dataset_path, "model_image.fits"), pixel_scales=0.03
)

residual_map_2d = ag.Array2D.from_fits(
    file_path=path.join(dataset_path, "residual_map.fits"), pixel_scales=0.03
)

# cmap = aplt.Cmap(cmap="jet", norm="log", vmin=1e-3, vmax=0.1)

# cmap = aplt.Cmap(cmap="jet", norm="linear", vmin=1e-3, vmax=0.1)

import cmastro

cmap = aplt.Cmap(
    cmap=cmastro.cmaps["cma:hesperia"],
    norm="symmetric_log",
    vmin=1e-3,
    vmax=0.1,
    linthresh=0.001,
)

xticks = aplt.XTicks(manual_units='%d"', fontsize=20)
yticks = aplt.YTicks(manual_units='%d"', fontsize=20)
xlabel = aplt.XLabel(label="")
ylabel = aplt.YLabel(label="")
title = aplt.Title(label="Hubble Space Telescope Galaxy Imaging", fontsize=18)
output = aplt.Output(path=dataset_path, filename="observed", format=["png", "pdf"])

mat_plot_2d = aplt.MatPlot2D(
    cmap=cmap,
    title=title,
    xlabel=xlabel,
    ylabel=ylabel,
    xticks=xticks,
    yticks=yticks,
    output=output,
)
mat_plot_2d.colorbar = False

array_2d_plotter = aplt.Array2DPlotter(array=image_2d, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()

tickparams = aplt.TickParams(labelleft=False, labelbottom=False)
title = aplt.Title(label="Parametric Fit (Bulge + Disk)", fontsize=18)
output = aplt.Output(path=dataset_path, filename="parametric", format=["png", "pdf"])

mat_plot_2d = aplt.MatPlot2D(
    cmap=cmap,
    title=title,
    xlabel=xlabel,
    ylabel=ylabel,
    xticks=xticks,
    yticks=yticks,
    tickparams=tickparams,
    output=output,
)
mat_plot_2d.colorbar = False

array_2d_plotter = aplt.Array2DPlotter(array=model_image_2d, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()

title = aplt.Title(label="Non Parametric Fit (Rectangular Pixelization)", fontsize=18)
output = aplt.Output(
    path=dataset_path, filename="non_parametric_cb", format=["png", "pdf"]
)

mat_plot_2d = aplt.MatPlot2D(
    cmap=cmap,
    title=title,
    xlabel=xlabel,
    ylabel=ylabel,
    xticks=xticks,
    yticks=yticks,
    tickparams=tickparams,
    output=output,
)
# mat_plot_2d.colorbar = False

array_2d_plotter = aplt.Array2DPlotter(array=residual_map_2d, mat_plot_2d=mat_plot_2d)
array_2d_plotter.figure_2d()


# import matplotlib.pyplot as plt
#
# # draw a new figure and replot the colorbar there
# fig,ax = plt.subplots(figsize=FIGSIZE)
# plt.colorbar()
# ax.remove()
# plt.savefig('plot_onlycbar.png')
