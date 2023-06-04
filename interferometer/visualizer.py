"""
Viusalize: Interferometer
==================

This script performs an interferometer model fit, where all images are output during visualization as .png and .fits
files.

This tests all visualization outputs in **PyAutoLens** for interferometer data.
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
import numpy as np

"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask_2d = ag.Mask2D.circular(
    shape_native=(400, 400), pixel_scales=0.2, radius=4.0, sub_size=1
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `light_sersic` from .fits files, which we will fit 
with the model.
"""
dataset_name = "light_sersic_exp"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask_2d,
)

settings_dataset = ag.SettingsInterferometer(transformer_class=ag.TransformerNUFFT)

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
    mesh=ag.mesh.DelaunayMagnification(shape=(30, 30)),
    regularization=ag.reg.ConstantSplit,
)

galaxy_pix = af.Model(ag.Galaxy, redshift=0.5, pixelization=pixelization)


model = af.Collection(galaxies=af.Collection(galaxy=galaxy, galaxy_pix=galaxy_pix))

"""
__Search__
"""
search = af.DynestyStatic(
    path_prefix=path.join("visualizer"),
    name="interferometer",
    unique_tag=dataset_name,
    nlive=50,
    maxcall=100,
    maxiter=100,
    number_of_cores=1,
)

"""
__Analysis__
"""
analysis = ag.AnalysisInterferometer(dataset=dataset)

"""
__Model-Fit__
"""
result = search.fit(model=model, analysis=analysis)

"""
Finish.
"""
