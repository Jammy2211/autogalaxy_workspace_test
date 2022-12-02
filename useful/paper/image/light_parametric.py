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
dataset_path = path.join("paper_image")

imaging = ag.Imaging.from_fits(
    image_path=path.join(dataset_path, "image_hdf_example.fits"),
    psf_path=path.join(dataset_path, "psf_hdf_example.fits"),
    noise_map_path=path.join(dataset_path, "noise_map_hdf_example.fits"),
    pixel_scales=0.03,
)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging, mat_plot_2d=mat_plot_2d)
imaging_plotter.subplot_imaging()

"""
__Masking__

The model-fit requires a `Mask2D` defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask_2d = ag.Mask2D.elliptical(
    shape_native=imaging.shape_native,
    pixel_scales=imaging.pixel_scales,
    major_axis_radius=9.0,
    angle=-10.0,
    axis_ratio=0.64,
)

imaging = imaging.apply_mask(mask=mask_2d)

imaging_plotter = aplt.ImagingPlotter(imaging=imaging, mat_plot_2d=mat_plot_2d)
imaging_plotter.subplot_imaging()

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's disk is a parametric `Exponential` disk, whose centre is aligned with the bulge [4 parameters].
 
The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.

NOTE: 

**PyAutoGalaxy** assumes that the galaxy centre is near the coordinates (0.0", 0.0"). 

If for your dataset the galaxy is not centred at (0.0", 0.0"), we recommend that you either: 

 - Reduce your data so that the centre is (`autogalaxy_workspace/*/preprocess`). 
 - Manually override the model priors (`autogalaxy_workspace/*/imaging/modeling/customize/priors.py`).
"""
bulge = af.Model(ag.lp.Sersic)
disk = af.Model(ag.lp.Exponential)
bulge.centre = disk.centre

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge, disk=disk)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))

"""
__Search__

The model is fitted to the data using a non-linear search. In this example, we use the nested sampling algorithm 
Dynesty (https://dynesty.readthedocs.io/en/latest/). We make the following changes to the Dynesty settings:

 - Increase the number of live points, `nlive`, from the default value of 50 to 100. 
 - Increase the number of random walks per live point, `walks` from the default value of 5 to 10. 
 
These changes are motivated by the higher dimensionality non-linear parameter space that including the lens light 
creates, which requires more thorough sampling by the non-linear search.

The folders: 

 - `autogalaxy_workspace/*/imaging/modeling/searches`.
 - `autogalaxy_workspace/*/imaging/modeling/customize`
  
Give overviews of the non-linear searches **PyAutoGalaxy** supports and more details on how to customize the
model-fit, including the priors on the model. 

If you are unclear of what a non-linear search is, checkout chapter 2 of the **HowToGalaxy** lectures.

The `name` and `path_prefix` below specify the path where results ae stored in the output folder:  

 `/autogalaxy_workspace/output/imaging/light_sersic/mass[sie]/unique_identifier`.

__Unique Identifier__

In the path above, the `unique_identifier` appears as a collection of characters, where this identifier is generated 
based on the model, search and dataset that are used in the fit.
 
An identical combination of model and search generates the same identifier, meaning that rerunning the script will use 
the existing results to resume the model-fit. In contrast, if you change the model or search, a new unique identifier 
will be generated, ensuring that the model-fit results are output into a separate folder.

We additionally want the unique identifier to be specific to the dataset fitted, so that if we fit different datasets
with the same model and search results are output to a different folder. We achieve this below by passing 
the `dataset_name` to the search's `unique_tag`.

__Number Of Cores__

We include an input `number_of_cores`, which when above 1 means that Dynesty uses parallel processing to sample multiple 
models at once on your CPU. When `number_of_cores=2` the search will run roughly two times as
fast, for `number_of_cores=3` three times as fast, and so on. The downside is more cores on your CPU will be in-use
which may hurt the general performance of your computer.

You should experiment to figure out the highest value which does not give a noticeable loss in performance of your 
computer. If you know that your processor is a quad-core processor you should be able to use `number_of_cores=4`. 

Above `number_of_cores=4` the speed-up from parallelization diminishes greatly. We therefore recommend you do not
use a value above this.

For users on a Windows Operating system, using `number_of_cores>1` may lead to an error, in which case it should be 
reduced back to 1 to fix it.
"""
search = af.DynestyStatic(
    path_prefix=path.join("imaging", "modeling"),
    name="light[bulge_disk]_2",
    nlive=100,
    walks=10,
    number_of_cores=1,
)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` used by the non-linear search to fit the model to 
the `Imaging` dataset.
"""
analysis = ag.AnalysisImaging(dataset=imaging)

"""
__Model-Fit__

We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
search to find which models fit the data with the highest likelihood.

Checkout the output folder for live outputs of the results of the fit, including on-the-fly visualization of the best 
fit model!
"""
result = search.fit(model=model, analysis=analysis)

"""
__Result__

The search returns a result object, which includes: 

 - The model corresponding to the maximum log likelihood solution in parameter space.
 - The corresponding maximum log likelihood `Plane` and `FitImaging` objects.
 - Information on the posterior as estimated by the `Dynesty` non-linear search. 
"""
print(result.max_log_likelihood_instance)

plane_plotter = aplt.PlanePlotter(
    plane=result.max_log_likelihood_plane, grid=result.grid
)
plane_plotter.subplot()

fit_imaging_plotter = aplt.FitImagingPlotter(fit=result.max_log_likelihood_fit)
fit_imaging_plotter.subplot_fit_imaging()

dynesty_plotter = aplt.DynestyPlotter(samples=result.samples)
dynesty_plotter.cornerplot()

"""
Checkout `autogalaxy_workspace/*/imaging/modeling/results.py` for a full description of the result object.
"""

model_image_2d = result.max_log_likelihood_fit.model_image.output_to_fits(
    file_path=path.join(dataset_path, "model_image.fits"), overwrite=True
)
residual_map_2d = result.max_log_likelihood_fit.residual_map.output_to_fits(
    file_path=path.join(dataset_path, "residual_map.fits"), overwrite=True
)
