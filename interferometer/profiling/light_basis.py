"""
__PROFILING: Interferometer Light Basis__

This profiling script times how long it takes to fit `interferometer` data with a Multi-Gaussian Expansion (MGE) linear
light profile fit to datasets of varying resolution.

This represents the time taken by a single iteration of the **PyAutoGalaxy** log likelihood function.
"""
import os
from os import path

cwd = os.getcwd()

from autoconf import conf

conf.instance.push(new_path=path.join(cwd, "config", "profiling"))

import json
import time

import autogalaxy as ag
import autogalaxy.plot as aplt

"""
The path all profiling results are output.
"""
profiling_path = path.dirname(path.realpath(__file__))

file_path = os.path.join(profiling_path, "times", ag.__version__, "light_mge")

"""
Whether w_tilde is used dictates the output folder.
"""
use_w_tilde = False

"""
The number of repeats used to estimate the `Inversion` run time.
"""
repeats = conf.instance["general"]["profiling"]["repeats"]
# repeats = 3
print("Number of repeats = " + str(repeats))
print()

"""
These settings control various aspects of how long a fit takes. The values below are default PyAutoLens values.
"""
sub_size = 1
mask_radius = 3.0
pixelization_shape_2d = (45, 45)

print(f"sub grid size = {sub_size}")
print(f"circular mask mask_radius = {mask_radius}")
print(f"pixelization shape = {pixelization_shape_2d}")


"""
The lens galaxy used to fit the data, which is identical to the lens galaxy used to simulate the data. 
"""
centre_y_list = [
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
    -0.00361289,
]

centre_x_list = [
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
    -0.00636064,
]

ell_comps_0_list = [
    0.05843285,
    0.0,
    0.05368621,
    0.05090395,
    0.0,
    0.25367341,
    0.01677313,
    0.03626733,
    0.15887384,
    0.02790297,
    0.12368768,
    0.38624915,
    -0.10490247,
    0.0385585,
]

ell_comps_1_list = [
    0.05932136,
    0.0,
    0.04267542,
    -0.06920487,
    -0.0,
    -0.15141799,
    0.01464508,
    0.03084128,
    -0.17983965,
    0.02215257,
    -0.16271084,
    -0.15945967,
    -0.3969543,
    -0.03808391,
]

sigma_list = [
    0.01607907,
    0.04039063,
    0.06734373,
    0.08471335,
    0.16048498,
    0.13531624,
    0.25649938,
    0.46096968,
    0.34492195,
    0.92418119,
    0.71803244,
    1.23547346,
    1.2574071,
    2.69979461,
]

gaussian_dict = {}

for gaussian_index in range(len(centre_x_list)):
    gaussian = ag.lp_linear.Gaussian(
        centre=(centre_y_list[gaussian_index], centre_x_list[gaussian_index]),
        ell_comps=(
            ell_comps_0_list[gaussian_index],
            ell_comps_1_list[gaussian_index],
        ),
        sigma=sigma_list[gaussian_index],
    )

    gaussian_dict[f"gaussian_{gaussian_index}"] = gaussian

galaxy = ag.Galaxy(redshift=0.5, **gaussian_dict)


gaussian_m = 1.0
gaussian_c = 1.0

gaussians = [ag.lp_linear.Gaussian() for i in range(10)]

for i, gaussian in enumerate(gaussians):
    gaussian.centre = gaussians[0].centre
    gaussian.ell_comps = gaussians[0].ell_comps
    gaussian.sigma = (gaussian_m * i) + gaussian_c

gaussian_dict = {f"gaussian_{i}": gaussian for i, gaussian in enumerate(gaussians)}

galaxy = ag.Galaxy(redshift=0.5, **gaussian_dict)

"""
__Dataset__

Set up the `Interferometer` dataset we fit. This includes the `real_space_mask` that the source galaxy's 
`Inversion` is evaluated using via mapping to Fourier space using the `Transformer`.
"""
instrument = "sma"
# instrument = "alma_low_res"
# instrument = "alma_high_res"

if instrument == "sma":
    real_shape_native = (64, 64)
    pixel_scales = (0.15625, 0.15625)

    real_space_mask = ag.Mask2D.circular_annular(
        shape_native=real_shape_native,
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        inner_radius=0.7,
        outer_radius=2.3,
    )

elif instrument == "alma_low_res":
    real_shape_native = (256, 256)
    pixel_scales = (0.0390625, 0.0390625)

    real_space_mask = ag.Mask2D.circular_annular(
        shape_native=real_shape_native,
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        inner_radius=0.25,
        outer_radius=1.15,
        centre=(0.0, 0.05),
    )

elif instrument == "alma_high_res":
    # real_shape_native = (1024, 1024)
    # pixel_scales = (0.0048828125, 0.0048828125)

    real_shape_native = (512, 512)
    pixel_scales = (0.027, 0.027)

    real_shape_native = (512, 512)
    pixel_scales = (0.009765625, 0.009765625)

    real_space_mask = ag.Mask2D.circular_annular(
        shape_native=real_shape_native,
        pixel_scales=pixel_scales,
        sub_size=sub_size,
        inner_radius=0.25,
        outer_radius=1.15,
        # inner_radius=0.5,
        # outer_radius=1.7,
        centre=(0.0, 0.05),
    )

else:
    raise Exception

"""
Load the strong lens dataset `mass_sie__source_sersic` `from .fits files.
"""

instrument = "sma"


dataset_name = "light_asymmetric"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
These settings control the run-time of the `Inversion` performed on the `Interferometer` data.
"""
transformer_class = ag.TransformerNUFFT

dataset = dataset.apply_settings(
    settings=ag.SettingsInterferometer(transformer_class=transformer_class)
)

"""
__Numba Caching__

Call FitInterferometer once to get all numba functions initialized.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitInterferometer(
    dataset=dataset,
    plane=plane,
    settings_inversion=ag.SettingsInversion(
        use_w_tilde=use_w_tilde,
    ),
)
print(fit.figure_of_merit)

"""
__Fit Time__

Time FitInterferometer by itself, to compare to profiling dict call.
"""
start = time.time()
for i in range(repeats):
    fit = ag.FitInterferometer(
        dataset=dataset,
        plane=plane,
        settings_inversion=ag.SettingsInversion(
            use_w_tilde=use_w_tilde,
        ),
    )
    fit.figure_of_merit
fit_time = (time.time() - start) / repeats
print(f"Fit Time = {fit_time} \n")

"""
__Profiling Dict__

Apply mask, settings and profiling dict to fit, such that timings of every individiual function are provided.
"""
run_time_dict = {}

galaxies = ag.Galaxies(galaxies=[galaxy], run_time_dict=run_time_dict)

fit = ag.FitInterferometer(
    dataset=dataset,
    plane=plane,
    settings_inversion=ag.SettingsInversion(
        use_w_tilde=use_w_tilde,
    ),
    run_time_dict=run_time_dict,
)
fit.figure_of_merit

run_time_dict = fit.run_time_dict

"""
__Results__

These two numbers are the primary driver of run time. More pixels = longer run time.
"""

print(f"Inversion fit run times for image type {instrument} \n")
print(f"Number of pixels = {dataset.grid.shape_slim} \n")
print(f"Number of sub-pixels = {dataset.grid.sub_shape_slim} \n")

"""
Print the profiling results of every step of the fit for command line output when running profiling scripts.
"""
for key, value in run_time_dict.items():
    print(key, value)

"""
__Predicted And Exccess Time__

The predicted time is how long we expect the fit should take, based on the individual profiling of functions above.

The excess time is the difference of this value from the fit time, and it indicates whether the break-down above
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

This is stored in a folder using the **PyAutoLens** version number so that profiling run times can be tracked through
**PyAutoLens** development.
"""
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = f"{instrument}_run_time_dict.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(run_time_dict, outfile)

"""
Output the profiling run time of the entire fit.
"""
filename = f"{instrument}_fit_time.json"

if os.path.exists(path.join(file_path, filename)):
    os.remove(path.join(file_path, filename))

with open(path.join(file_path, filename), "w") as outfile:
    json.dump(fit_time, outfile)

"""
Output an image of the fit, so that we can inspect that it fits the data as expected.
"""
mat_plot_2d = aplt.MatPlot2D(
    output=aplt.Output(
        path=file_path,
        filename=f"{instrument}_subplot_fit",
        format="png",
    )
)
fit_plotter = aplt.FitInterferometerPlotter(fit=fit, mat_plot_2d=mat_plot_2d)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()
fit_plotter.subplot_fit_real_space()

"""
The `info_dict` contains all the key information of the analysis which describes its run times.
"""
info_dict = {}
info_dict["repeats"] = repeats
info_dict["image_pixels"] = dataset.grid.sub_shape_slim
info_dict["sub_size"] = sub_size
info_dict["mask_radius"] = mask_radius
# info_dict["source_pixels"] = len(reconstruction)

print(info_dict)

with open(path.join(file_path, f"{instrument}_info.json"), "w") as outfile:
    json.dump(info_dict, outfile)
