"""
Overview: Interferometer
------------------------

Alongside CCD `Imaging` data, **PyAutoGalaxy** supports the modeling of interferometer data from submillimeter and radio
observatories.

The dataset is fitted directly in the uv-plane, circumventing issues that arise when fitting a `dirty image` such as
correlated noise.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt
from os import path
import numpy as np

import os

workspace_path = os.getcwd()

"""
__Real Space Mask__

To begin, we define a real-space mask. Although interferometer analysis is performed in the uv-plane and 
therefore Fourier space, we still need to define the grid of coordinates in real-space from which the galaxy's 
images are computed. It is this image that is mapped to Fourier space to compare to the uv-plane data.
"""
real_space_mask = ag.Mask2D.circular(
    shape_native=(400, 400), pixel_scales=0.025, radius=3.0
)

"""
__Dataset__

We next load an interferometer dataset from fits files, which follows the same API that we have seen for an `Imaging`
object.
"""
dataset_name = "light_sersic"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
)

"""
The PyAutoGalaxy plot module has tools for plotting interferometer datasets, including the visibilities, noise-map
and uv wavelength which represent the interferometer`s baselines. 

The data used in this tutorial contains 1 million visibilities and is representative of an ALMA dataset:
"""
dataset_plotter = aplt.InterferometerPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="visibilities", format="png")
    ),
)
dataset_plotter.figures_2d(data=True)

dataset_plotter = aplt.InterferometerPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="uv_wavelengths", format="png")
    ),
)
dataset_plotter.figures_2d(uv_wavelengths=True)

"""
This can also plot the dataset in real-space, using the fast Fourier transforms described below.
"""
dataset_plotter = aplt.InterferometerPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="dirty_image", format="png")
    ),
)

dataset_plotter.figures_2d(dirty_image=True)

dataset_plotter = aplt.InterferometerPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="dirty_signal_to_noise", format="png"
        )
    ),
)

dataset_plotter.figures_2d(dirty_signal_to_noise_map=True)

"""
__Galaxies__

To perform uv-plane modeling, **PyAutoGalaxy** generates an image of the galaxy system in real-space via a `Plane`. 

Lets quickly set up the `Plane` we'll use in this example.
"""
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

plane_plotter = aplt.GalaxiesPlotter(
    plane=plane,
    grid=real_space_mask.derive_grid.unmasked,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="image_pre_ft", format="png")
    ),
)
plane_plotter.figures_2d(image=True)

"""
__UV-Plane__

To perform uv-plane modeling, **PyAutoGalaxy** next Fourier transforms this image from real-space to the uv-plane.

This operation uses a *Transformer* object, of which there are multiple available in **PyAutoGalaxy**. This includes
a direct Fourier transform which performs the exact Fourier transformer without approximation.
"""
transformer_class = ag.TransformerDFT

"""
However, the direct Fourier transform is inefficient. For ~10 million visibilities, it requires thousands of seconds
to perform a single transform. This approach is therefore unfeasible for high quality ALMA and radio datasets.

For this reason, **PyAutoGalaxy** supports the non-uniform fast fourier transform algorithm
**PyNUFFT** (https://github.com/jyhmiinlin/pynufft), which is significantly faster, being able too perform a Fourier
transform of ~10 million in less than a second!
"""
transformer_class = ag.TransformerNUFFT

"""
The use this transformer in a fit, we use the `apply_settings` method.
"""
dataset = dataset.apply_settings(
    settings=ag.SettingsInterferometer(transformer_class=transformer_class)
)

"""
__Fitting__

The interferometer can now be used with a `FitInterferometer` object to fit it to a dataset:
"""
fit = ag.FitInterferometer(dataset=dataset, plane=plane)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)

"""
Visualization of the fit can be performed in the uv-plane or in real-space. 

Note that the fit is not performed in real-space, but plotting it in real-space is often more informative.
"""
fit_plotter = aplt.FitInterferometerPlotter(
    fit=fit,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="model_data", format="png")
    ),
)
fit_plotter.figures_2d(model_data=True)
fit_plotter = aplt.FitInterferometerPlotter(
    fit=fit,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(
            path=workspace_path, filename="fit_dirty_images", format="png"
        )
    ),
)
fit_plotter.subplot_fit_dirty_images()
"""
Interferometer data can also be modeled using pixelizations, which again perform the galaxy reconstruction by
directly fitting the visibilities in the uv-plane. 

The galaxy reconstruction can be visualized in real space:
"""
galaxy = ag.Galaxy(
    redshift=1.0,
    pixelization=ag.mesh.Rectangular(shape=(30, 30)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitInterferometer(
    dataset=dataset,
    plane=plane,
    settings_inversion=ag.SettingsInversion(use_linear_operators=True),
)

fit_plotter = aplt.FitInterferometerPlotter(fit=fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()
fit_plotter.subplot_fit_real_space()

"""
The combination of pixelizations with interferometer datasets therefore offers a compelling way to reconstruct
their emission following a fully Bayesian framework. 

This can allow one to determine whether star forming clumps are resolved in the data, with the fitting in the uv-plane
ensuring they are not spurious noise.

__Efficiency__

Computing this galaxy reconstruction would be extremely inefficient if **PyAutoGalaxy** used a traditional approach to
linear algebra which explicitly stored in memory the values required to solve for the source fluxes. In fact, for an
interferometer dataset of ~10 million visibilities this would require **hundreds of GB of memory**!

**PyAutoGalaxy** uses the library **PyLops** (https://pylops.readthedocs.io/en/latest/) to represent this 
calculation as a sequence of memory-light linear operators.
"""
inversion_plotter = aplt.InversionPlotter(inversion=fit.inversion)
inversion_plotter.figures_2d_of_pixelization(pixelization_index=0, reconstruction=True)

"""
__Modeling__

It is straight forward to fit a model to an interferometer dataset, using the same API that we saw for imaging
data in the `overview/modeling.py` example.

__Model__

We first compose the model, in the same way described in the `modeling.py` overview script:
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic)

galaxies = af.Collection(galaxy=galaxy)
model = af.Collection(galaxies=galaxies)

"""
__Non-linear Search__

We again choose the non-linear search `dynesty` (https://github.com/joshspeagle/dynesty).
"""
search = af.DynestyStatic(name="overview_interferometer")

"""
__Analysis__

Whereas we previously used an `AnalysisImaging` object, we instead use an `AnalysisInterferometer` object which fits 
the model in the correct way for an interferometer dataset. 

This includes mapping the model from real-space to the uv-plane via the Fourier transform discussed above.
"""
analysis = ag.AnalysisInterferometer(dataset=dataset)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

The results can be found in the `output/overview_interferometer` folder in the `autogalaxy_workspace`.
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The **PyAutoGalaxy** visualization library and `FitInterferometer` object includes specific methods for plotting the 
results.
"""
fit_plotter = aplt.FitInterferometerPlotter(fit=result.max_log_likelihood_fit)
fit_plotter.subplot_fit()
fit_plotter.subplot_fit_dirty_images()

"""
__Simulation__

Simulated interferometer datasets can be generated using the ``SimulatorInterferometer`` object, which includes adding
Gaussian noise to the visibilities:
"""
simulator = ag.SimulatorInterferometer(
    uv_wavelengths=dataset.uv_wavelengths, exposure_time=300.0, noise_sigma=0.01
)

real_space_grid_2d = ag.Grid2D.uniform(
    shape_native=real_space_mask.shape_native, pixel_scales=real_space_mask.pixel_scales
)

dataset = simulator.via_galaxies_from(plane=plane, grid=real_space_grid_2d)

"""
__Wrap Up__

The `interferometer` package of the `autogalaxy_workspace` contains numerous example scripts for performing 
interferometer modeling and simulating galaxy interferometer datasets.
"""
