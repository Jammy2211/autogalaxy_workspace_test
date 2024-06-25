"""
__PROFILING: Gaussian MGE__

This profiling script times how long it takes to fit `Imaging` data with a Multi-Gaussian Expansion (MGE) linear
light profile fit to datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoGalaxy** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import numpy as np
import time
import json

import autogalaxy as ag
import autogalaxy.plot as aplt


"""
The path all profiling results are output.
"""
profiling_path = path.dirname(path.realpath(__file__))

file_path = os.path.join(profiling_path, "times", ag.__version__, "light_basis")

"""
The number of repeats used to estimate the run time.
"""
repeats = conf.instance["general"]["profiling"]["repeats"]
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoGalaxy values.
"""
mask_radius = 3.5
psf_shape_2d = (21, 21)


print(f"circular mask mask_radius = {mask_radius}")
print(f"psf shape = {psf_shape_2d}")

"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
total_gaussians = 30

log10_sigma_list = np.linspace(-2, mask_radius, total_gaussians)

bulge_gaussian_list = []

for i in range(total_gaussians):
    gaussian = ag.lp.Gaussian(
        centre=(0.1, 0.1),
        ell_comps=ag.convert.ell_comps_from(axis_ratio=0.8, angle=60.0),
        sigma=10 ** log10_sigma_list[i],
    )

    bulge_gaussian_list.append(gaussian)

bulge = ag.lp_basis.Basis(profile_list=bulge_gaussian_list)

galaxy = ag.Galaxy(redshift=0.5, bulge=bulge)


"""
__Dataset__

Load and plot the galaxy dataset `light_asymmetric` via .fits files, which we will fit with the model.
"""
dataset_name = "light_mge"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.05,
)

"""
Apply the 2D mask, which for the settings above is representative of the masks we typically use to model strong lenses.
"""
mask = ag.Mask2D.circular(
    shape_native=dataset.shape_native,
    pixel_scales=dataset.pixel_scales,
    radius=mask_radius,
)

masked_dataset = dataset.apply_mask(mask=mask)

"""
__Numba Caching__

Call FitImaging once to get all numba functions initialized.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(
    dataset=masked_dataset,
    galaxies=galaxies,
    settings_inversion=ag.SettingsInversion(use_w_tilde=False),
)
print(fit.figure_of_merit)

"""
__Fit Time__

Time FitImaging by itself, to compare to profiling dict call.
"""
start = time.time()
for i in range(repeats):
    fit = ag.FitImaging(
        dataset=masked_dataset,
        galaxies=galaxies,
        settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    )
    print(fit.figure_of_merit)
fit_time = (time.time() - start) / repeats
print(f"Fit Time = {fit_time} \n")


"""
__Profiling Dict__

Apply mask, settings and profiling dict to fit, such that timings of every individiual function are provided.
"""
run_time_dict = {}

galaxies = ag.Galaxies(galaxies=[galaxy], run_time_dict=run_time_dict)

fit = ag.FitImaging(
    dataset=masked_dataset,
    galaxies=galaxies,
    settings_inversion=ag.SettingsInversion(use_w_tilde=False),
    run_time_dict=run_time_dict,
)
fit.figure_of_merit

run_time_dict = fit.run_time_dict

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Number of pixels = {masked_dataset.grid.shape_slim} \n")
print(f"Number of sub-pixels = {masked_dataset.grid.sub_shape_slim} \n")

"""
Print the profiling results of every step of the fit for command line output when running profiling scripts.
"""
for key, value in run_time_dict.items():
    print(key, value)

"""
__Predicted And Exccess Time__

The predicted time is how long we expect the fit should take, based on the individual profiling of functions above.

The excess time is the difference of this value from the fit time, and it indiciates whether the break-down above
has missed expensive steps.
"""
predicted_time = 0.0
predicted_time = sum(run_time_dict.values())
excess_time = fit_time - predicted_time

print(f"\nExcess Time = {excess_time} \n")

"""
__Output__

Output the profiling run times as a dictionary so they can be used in `profiling/graphs.py` to create graphs of the
profile run times.

This is stored in a folder using the **PyAutoGalaxy** version number so that profiling run times can be tracked through
**PyAutoGalaxy** development.
"""
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = f"run_time_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(run_time_dict, outfile)

"""
Output the profiling run time of the entire fit.
"""
filename = f"fit_time.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(fit_time, outfile)

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=file_path, filename=f"subplot_fit", format="png")
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_fit()

mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(path=file_path, filename=f"subplot_of_plane_1", format="png")
)
fit_plotter = aplt.FitImagingPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_of_galaxies(galaxy_index=0)

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = masked_dataset.grid.sub_shape_slim
info_dict["mask_radius"] = mask_radius
info_dict["psf_shape_2d"] = psf_shape_2d
info_dict["source_pixels"] = len(fit.inversion.reconstruction)
info_dict["excess_time"] = excess_time

print(info_dict)

with open(path.join(file_path, f"info.json"), "w+") as outfile:
    json.dump(info_dict, outfile, indent=4)
