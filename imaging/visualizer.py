"""
Viusalize: Imaging
==================

This script performs an imaging model fit, where all images are output during visualization as .png and .fits
files.

This tests all visualization outputs in **PyAutoLens** for imaging data.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "visualizer"))

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt

"""
__Dataset__

Load and plot the galaxy dataset `light_sersic_exp` via .fits files, which we will fit with 
the model.
"""
dataset_name = "light_sersic_exp"
dataset_path = path.join("dataset", "imaging", dataset_name)

imaging = ag.Imaging.from_fits(
    image_path=path.join(dataset_path, "image.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Masking__
"""
mask_2d = ag.Mask2D.circular(
    shape_native=imaging.shape_native, pixel_scales=imaging.pixel_scales, radius=3.0
)

imaging = imaging.apply_mask(mask=mask_2d)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging)
imaging_plotter.subplot_imaging()

"""
__Model__
"""
bulge = af.Model(ag.lp.Sersic)

bulge.centre = (0.0, 0.0)
bulge.ell_comps = ag.convert.ell_comps_from(axis_ratio=0.9, angle=45.0)
bulge.intensity = 1.0
bulge.effective_radius = 0.6
bulge.sersic_index = 3.0

disk = af.Model(ag.lp_linear.Exponential)
disk.centre = (0.05, 0.05)
disk.ell_comps = ag.convert.ell_comps_from(axis_ratio=0.7, angle=30.0)
disk.effective_radius = af.UniformPrior(lower_limit=1.59, upper_limit=1.61)

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

pixelization = af.Model(
    ag.Pixelization,
    mesh=ag.mesh.DelaunayMagnification,
    regularization=ag.reg.ConstantSplit,
)

galaxy_pix = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy, galaxy_pix=galaxy_pix))

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path.join("visualizer"),
    name="imaging",
    unique_tag=dataset_name,
    nlive=100,
    walks=10,
    maxcall=100,
    maxiter=100,
    number_of_cores=1,
)

"""
__Analysis__
"""
analysis = ag.AnalysisImaging(dataset=imaging)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
