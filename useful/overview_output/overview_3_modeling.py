"""
Overview: Modeling
------------------

Modeling is the process of taking data of a galaxy (e.g. imaging data from the Hubble Space Telescope or interferometer
data from ALMA) and fitting it with a model, to determine the `LightProfile`'s that best represent the observed galaxy.

Modeling uses the probabilistic programming language
`PyAutoFit <https://github.com/rhayes777/PyAutoFit>`_, an open-source Python framework that allows complex model
fitting techniques to be straightforwardly integrated into scientific modeling software. Check it out if you
are interested in developing your own software to perform advanced model-fitting!

We import **PyAutoFit** separately to **PyAutoGalaxy**
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path

import autogalaxy as ag
import autogalaxy.plot as aplt

import autofit as af

import os

workspace_path = os.getcwd()

"""
__Dataset__

In this example, we fit simulated imaging of a galaxy. 

First, lets load this imaging dataset and plot it.
"""
dataset_name = "light_sersic_exp"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__

We next mask the dataset, to remove the exterior regions of the image that do not contain emission from the galaxy.

Note how when we plot the `Imaging` below, the figure now zooms into the masked region.
"""
mask_2d = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask_2d)

dataset_plotter = aplt.ImagingPlotter(
    dataset=dataset,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="image", format="png")
    ),
)
dataset_plotter.figures_2d(data=True)

"""
__Model__

We compose the model that we fit to the data using PyAutoFit `Model` objects. 

These behave analogously to `Galaxy` objects but their  `LightProfile` parameters are not specified and are instead 
determined by a fitting procedure.

In this example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's disk is a parametric `Exponential` disk [6 parameters].
 
Note how we can easily extend the model below to include extra light profiles in the galaxy.
"""
galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=ag.lp.Sersic, disk=ag.lp.Exponential)

"""
We put the model galaxy above into a `Collection`, which is the model we will fit. Note how we could easily 
extend this object to compose complex models containing many galaxies.

The reason we create separate `Collection`'s for the `galaxies` and `model` is so that the `model`
can be extended to include other components than just galaxies.
"""
galaxies = af.Collection(galaxy=galaxy)
model = af.Collection(galaxies=galaxies)

"""
__Non-linear Search__

We now choose the non-linear search, which is the fitting method used to determine the set of `LightProfile` (e.g.
bulge and disk) parameters that best-fit our data.

In this example we use `dynesty` (https://github.com/joshspeagle/dynesty), a nested sampling algorithm that is
very effective at modeling.

**PyAutoGalaxy** supports many model-fitting algorithms, including maximum likelihood estimators and MCMC, which are
documented throughout the workspace.
"""
search = af.DynestyStatic(name="overview_modeling")

"""
__Analysis__

We next create an `AnalysisImaging` object, which contains the `log likelihood function` that the non-linear search 
calls to fit the model to the data.
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
__Model-Fit__

To perform the model-fit we pass the model and analysis to the search's fit method. This will output results (e.g.,
dynesty samples, model parameters, visualization) to hard-disk.

Once running you should checkout the `autogalaxy_workspace/output` folder, which is where the results of the search are 
written to hard-disk (in the `overview_modeling` folder) on-the-fly. This includes model parameter estimates with 
errors non-linear samples and the visualization of the best-fit model inferred by the search so far. 
"""
result = search.fit(model=model, analysis=analysis)

"""
__Results__

Whilst navigating the output folder, you may of noted the results were contained in a folder that appears as a random
collection of characters. 

This is the model-fit's unique identifier, which is generated based on the model, search and dataset used by the fit. 
Fitting an identical model, search and dataset will generate the same identifier, meaning that rerunning the script 
will use the existing results to resume the model-fit. In contrast, if you change the model, search or dataset, a new 
unique identifier will be generated, ensuring that the model-fit results are output into a separate folder.

The fit above returns a `Result` object, which includes lots of information on the model. Below, 
we print the maximum log likelihood bulge and disk models inferred.
"""
print(result.max_log_likelihood_instance.galaxies.galaxy.bulge)
print(result.max_log_likelihood_instance.galaxies.galaxy.disk)

"""
In fact, the result contains the full posterior information of our non-linear search, including all
parameter samples, log likelihood values and tools to compute the errors on the model. **PyAutoGalaxy** includes
visualization tools for plotting this.
"""
plotter = aplt.NestPlotter(
    samples=result.samples,
    output=aplt.Output(path=workspace_path, filename="corner", format="png"),
)
plotter.corner_cornerpy()

"""
The result also contains the maximum log likelihood `Galaxies` and `FitImaging` objects which can easily be plotted.
"""
plane_plotter = aplt.GalaxiesPlotter(
    galaxies=result.max_log_likelihood_galaxies, grid=dataset.grid
)
plane_plotter.subplot()

fit_plotter = aplt.FitImagingPlotter(
    fit=result.max_log_likelihood_fit,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="subplot_fit", format="png")
    ),
)
fit_plotter.subplot_fit()

"""
A full guide of result objects is contained in the `autogalaxy_workspace/*/results` package.

__Model Customization__

The `Model` can be fully customized, making it simple to parameterize and fit many different models
using any combination of light profiles and galaxies:
"""
galaxy_model = af.Model(
    ag.Galaxy,
    redshift=0.5,
    bulge=ag.lp.DevVaucouleurs,
    disk=ag.lp.Sersic,
    bar=ag.lp.Gaussian,
    extra_galaxy_0=ag.lp.ElsonFreeFall,
    extra_galaxy_1=ag.lp.ElsonFreeFall,
)

"""
This aligns the bulge and disk centres in the first galaxy of the model, reducing the
number of free parameter fitted for by Dynesty by 2.
"""
galaxy_model.bulge.centre = galaxy_model.disk.centre

"""
This fixes the galaxy bulge light profile's effective radius to a value of
0.8 arc-seconds, removing another free parameter.
"""
galaxy_model.bulge.effective_radius = 0.8

"""
This forces the light profile bulge's effective radius to be above 3.0.
"""
galaxy_model.bulge.add_assertion(galaxy_model.bulge.effective_radius > 3.0)

"""
__Wrap Up__

A more detailed description of modeling's is given in chapter 2 of the **HowToGalaxy** 
tutorials, which I strongly advise new users check out!
"""
