"""
Preprocess 1: Image
===================

The image is the image of your galaxy, which likely comes from atelescope like the Hubble Space telescope (HST).

Throughout all these tutorials, we'll refer to a "pixel_scale" when loading data. The pixel-scale describes the
pixel-units to arcsecond-units conversion factor of your telescope, which you should look up now if you are unsure
of the value. HST `Imaging` typically has a pixel_scale of 0.05", however this varies depending on the detector and
data reduction procedure so DOUBLE CHECK THIS!

This tutorial describes preprocessing your dataset`s image to adhere to the units and formats required by PyAutoGalaxy.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

# %matplotlib inline
from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
Setup the path the datasets we'll use to illustrate preprocessing, which is the folder `dataset/imaging/preprocess`.
"""
dataset_path = path.join("paper_image")

"""
__Loading Data From Individual Fits Files__

First, lets load an image as an `Array2D`.

This image represents a good data-reduction that conforms **PyAutoLens** formatting standards!
"""
image_2d = ag.Array2D.from_fits(
    file_path=path.join(
        dataset_path, "hlsp_hudf12_hst_acs_udfpar2_f814w_v1.0_drz.fits"
    ),
    pixel_scales=0.03,
)

cmap = aplt.Cmap(cmap="jet", norm="linear", vmin=0.0, vmax=0.1)

mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

# array_plotter = aplt.Array2DPlotter(array=image_2d, mat_plot_2d=mat_plot_2d)
# array_plotter.figure_2d()

image_2d = image_2d.native[5540:6140, 3740:4340]
image_2d = ag.Array2D.no_mask(values=image_2d, pixel_scales=0.03)

"""
There are numerous reasons why the image below is a good data-set for modeling. I strongly recommend 
you adapt your data reduction pipelines to conform to the formats discussed below - it`ll make your time using 
PyAutoGalaxy a lot simpler.

However, you may not have access to the data-reduction tools that made the data, so we've included in-built functions 
in PyAutoGalaxy to convert the data to a suitable format.
"""
array_plotter = aplt.Array2DPlotter(array=image_2d, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

image_2d.output_to_fits(
    file_path=path.join(dataset_path, "image_hdf_example.fits"), overwrite=True
)


noise_map_2d = ag.Array2D.from_fits(
    file_path=path.join(
        dataset_path, "hlsp_hudf12_hst_acs_udfpar2_f814w_v1.0_wht.fits"
    ),
    pixel_scales=0.03,
)


cmap = aplt.Cmap(cmap="jet", norm="linear", vmin=0.0)

mat_plot_2d = aplt.MatPlot2D(cmap=cmap)

# array_plotter = aplt.Array2DPlotter(array=noise_map_2d, mat_plot_2d=mat_plot_2d)
# array_plotter.figure_2d()

noise_map_2d = noise_map_2d.native[5540:6140, 3740:4340]

exposure_time_map = ag.Array2D.full(
    fill_value=322944, shape_native=image_2d.shape_native, pixel_scales=0.03
)

background_noise_map_2d = ag.preprocess.background_noise_map_via_edges_of_image_from(
    image=image_2d.native, no_edges=2
)

noise_map_2d = ag.preprocess.noise_map_via_data_eps_exposure_time_map_and_background_noise_map_from(
    data_eps=image_2d.native,
    exposure_time_map=exposure_time_map.native,
    background_noise_map=background_noise_map_2d.native,
)

noise_map_2d = ag.Array2D.no_mask(values=noise_map_2d, pixel_scales=0.03)


array_plotter = aplt.Array2DPlotter(array=noise_map_2d, mat_plot_2d=mat_plot_2d)
array_plotter.figure_2d()

noise_map_2d.output_to_fits(
    file_path=path.join(dataset_path, "noise_map_hdf_example.fits"), overwrite=True
)


signal_to_noise_map_2d = ag.Array2D.no_mask(
    values=image_2d.native / noise_map_2d.native, pixel_scales=0.03
)

array_plotter = aplt.Array2DPlotter(
    array=signal_to_noise_map_2d, mat_plot_2d=mat_plot_2d
)
array_plotter.figure_2d()

signal_to_noise_map_2d.output_to_fits(
    file_path=path.join(dataset_path, "signal_to_noise_map_hdf_example.fits"),
    overwrite=True,
)


psf = ag.Kernel2D.from_gaussian(shape_native=(11, 11), sigma=0.05, pixel_scales=0.03)
psf.output_to_fits(
    file_path=path.join(dataset_path, "psf_hdf_example.fits"), overwrite=True
)
