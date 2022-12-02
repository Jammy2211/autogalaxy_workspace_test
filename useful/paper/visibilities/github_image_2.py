"""
Modeling: Mass Total + Source Parametric
========================================

This script fits `Interferometer` dataset of a galaxy with a model where:

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
import numpy as np

"""
__Masking__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask_2d = ag.Mask2D.circular(
    shape_native=(128, 128), pixel_scales=0.05, radius=3.0, sub_size=1
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `light_sersic` from .fits files, which we will fit 
with the model.
"""
dataset_name = "visibilities"
dataset_path = path.join("paper", dataset_name)

interferometer = ag.Interferometer.from_fits(
    visibilities_path=path.join(dataset_path, "visibilities.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask_2d,
)

title = aplt.Title(label="Dirty Image via Fourier Transform", fontsize=18)
xticks = aplt.XTicks(
    manual_units='%.1f"', manual_values=[-2.8, -1.4, 0.0, 1.4, 2.8], fontsize=20
)
yticks = aplt.YTicks(manual_units='%.1f"', fontsize=20)
xlabel = aplt.XLabel(label="")
ylabel = aplt.YLabel(label="")
cmap = aplt.Cmap(cmap="inferno")
tickparams = aplt.TickParams(labelleft=False)

output = aplt.Output(
    path=dataset_path, filename="dirty_image_cb", format=["png", "pdf"]
)

mat_plot_2d = aplt.MatPlot2D(
    title=title,
    xticks=xticks,
    yticks=yticks,
    xlabel=xlabel,
    ylabel=ylabel,
    output=output,
    cmap=cmap,
    tickparams=tickparams,
)
# mat_plot_2d.colorbar = False

include_2d = aplt.Include2D(mask=False, border=False)

interferometer_plotter = aplt.InterferometerPlotter(
    interferometer=interferometer, mat_plot_2d=mat_plot_2d, include_2d=include_2d
)
interferometer_plotter.figures_2d(dirty_image=True)


galaxy = ag.Galaxy(
    redshift=0.5,
    light=ag.lp.Sersic(
        centre=(0.105, 0.052),
        ell_comps=(-0.039, 0.174),
        effective_radius=0.424,
        sersic_index=0.517,
        intensity=0.000395514379,
    ),
)

plane = ag.Plane(galaxies=[galaxy])

fit = ag.FitInterferometer(dataset=interferometer, plane=plane)


title = aplt.Title(label="Model Dirty Image via Fourier Transform", fontsize=18)

output = aplt.Output(
    path=dataset_path, filename="model_dirty_image", format=["png", "pdf"]
)

tickparams = aplt.TickParams(labelleft=False, labelbottom=False)

mat_plot_2d = aplt.MatPlot2D(
    title=title,
    xticks=xticks,
    yticks=yticks,
    xlabel=xlabel,
    ylabel=ylabel,
    output=output,
    tickparams=tickparams,
    cmap=cmap,
)
mat_plot_2d.colorbar = False

fit_plotter = aplt.FitInterferometerPlotter(
    fit=fit, mat_plot_2d=mat_plot_2d, include_2d=include_2d
)
fit_plotter.figures_2d(dirty_model_image=True)


no_visibilities = interferometer.visibilities.shape[0]

interferometer.data = interferometer.visibilities[0 : int(no_visibilities / 50)]

size = 0.4
axis = aplt.Axis(extent=[-size, size, -size, size])
grid_scatter = aplt.GridScatter(marker=".", s=1)

xticks = aplt.XTicks(
    manual_units="%.1fJy", fontsize=20, manual_values=[-0.3, -0.1, 0.1, 0.3]
)
yticks = aplt.YTicks(
    manual_units="%.1fJy", fontsize=20, manual_values=[-0.3, -0.1, 0.1, 0.3]
)

title = aplt.Title(label="ALMA Visibilities", fontsize=18)

output = aplt.Output(path=dataset_path, filename="visibilities", format=["png", "pdf"])

mat_plot_2d = aplt.MatPlot2D(
    axis=axis,
    grid_scatter=grid_scatter,
    title=title,
    xlabel=xlabel,
    ylabel=ylabel,
    xticks=xticks,
    yticks=yticks,
    output=output,
)

interferometer_plotter = aplt.InterferometerPlotter(
    interferometer=interferometer, mat_plot_2d=mat_plot_2d
)
interferometer_plotter.figures_2d(visibilities=True)


# visibilities = interferometer.visibilities
#
# size = 0.7
# axis = aplt.Axis(extent=[-size, size, -size, size])
#
# visibilities_plotter = aplt.Grid2DPlotter(grid=visibilities)
# visibilities_plotter.figure_2d()
