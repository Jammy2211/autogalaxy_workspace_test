"""
Overview: Simulate
------------------

**PyAutoGalaxy** provides tool for simulating galaxy data-sets, which can be used to test modeling pipelines
and train neural networks to recognise and analyse images of galaxies.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autogalaxy as ag
import autogalaxy.plot as aplt

import os

workspace_path = os.getcwd()

"""
__Grid + Lens__

In this overview we used a plane and grid to create an image of a galaxy.
"""
grid_2d = ag.Grid2D.uniform(
    shape_native=(80, 80),
    pixel_scales=0.1,  # <- The pixel-scale describes the conversion from pixel units to arc-seconds.
)

galaxy = ag.Galaxy(
    redshift=0.5,
    bulge=ag.lp.Sersic(
        centre=(0.0, 0.0),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0),
        intensity=1.0,
        effective_radius=0.8,
        sersic_index=4.0,
    ),
)


galaxies = ag.Galaxies(galaxies=[galaxy])

plane_plotter = aplt.GalaxiesPlotter(plane=plane, grid=grid_2d)
plane_plotter.figures_2d(image=True)

"""
__Simulator__

Simulating galaxy images uses a `SimulatorImaging` object, which models the process that an instrument like the
Hubble Space Telescope goes through to observe a galaxy. This includes accounting for the exposure time to 
determine the signal-to-noise of the data, blurring the observed light of the galaxy with the telescope optics 
and accounting for the background sky in the exposure which adds Poisson noise.
"""
psf = ag.Kernel2D.from_gaussian(shape_native=(11, 11), sigma=0.1, pixel_scales=0.05)

simulator = ag.SimulatorImaging(
    exposure_time=300.0, background_sky_level=1.0, psf=psf, add_poisson_noise=True
)

"""
Once we have a simulator, we can use it to create an imaging dataset which consists of an image, noise-map and 
Point Spread Function (PSF) by passing it a plane and grid.

This uses the plane above to create the image of the galaxy and then add the effects that occur during data
acquisition.
"""
dataset = simulator.via_galaxies_from(plane=plane, grid=grid_2d)

"""
By plotting a subplot of the `Imaging` dataset, we can see this object includes the observed image of the galaxy
(which has had noise and other instrumental effects added to it) as well as a noise-map and PSF:
"""
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="image", format="png")
    ),
)
dataset_plotter.figures_2d(data=True)
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="noise_map", format="png")
    ),
)
dataset_plotter.figures_2d(noise_map=True)
dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="psf", format="png")
    ),
)
dataset_plotter.figures_2d(psf=True)

"""
__Wrap Up__

The `autogalaxy_workspace` includes many example simulators for simulating galaxies with a range of different 
physical properties, to make imaging datasets for a variety of telescopes (e.g. Hubble, Euclid) as well as 
interferometer datasets.
"""
