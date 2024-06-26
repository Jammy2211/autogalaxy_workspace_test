"""
Overview: Pixelizations
-----------------------

Pixelizations reconstruct a galaxy's light on a pixel-grid.

Unlike `LightProfile`'s, they are able to reconstruct the light of non-symmetric and irregular galaxies.

To reconstruct the galaxy using a `Pixelization`, we impose a prior on the smoothness of the reconstructed
source, called the `Regularization`. The more we regularize the galaxy, the smoother the reconstruction.

The process of reconstructing a `Galaxy`'s light using a `Pixelization`  is called an `Inversion`,
and the term `inversion` is used throughout the **PyAutoGalaxy** example scripts to signify that their analysis
reconstructs the galaxy's light on a pixel-grid.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

from os import path
import autogalaxy as ag
import autogalaxy.plot as aplt

import os

workspace_path = os.getcwd()

"""
__Dataset__

Load the `Imaging` data that we'll reconstruct the galaxy's light of using a pixelization.

Note how complex the lensed source galaxy looks, with multiple clumps of light - this would be very difficult to 
represent using `LightProfile`'s!
"""
dataset_name = "light_complex"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__

We are going to fit this data, so we must create `Mask2D` and `Imaging` objects.
"""
mask_2d = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=2.5
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
__Pixelization + Regularization__

To reconstruct the galaxy on a pixel-grid, we simply pass it the `Mesh` class we want to reconstruct its 
light on as well as the `Regularization` scheme describing how we smooth the source reconstruction. 

We use a `Rectangular` mesh with resolution 40 x 40 and a `Constant` regularizaton scheme with a regularization
coefficient of 1.0. The higher this coefficient, the more our source reconstruction is smoothed.
"""
galaxy = ag.Galaxy(
    redshift=1.0,
    pixelization=ag.mesh.Rectangular(shape=(50, 50)),
    regularization=ag.reg.Constant(coefficient=1.0),
)

"""
__Fit__

Now that our galaxy has a `Pixelization`, we are able to fit the data using it in the 
same way as before, by simply passing the galaxy to a `Plane` and using this `Plane` to create a `FitImaging`
object.
"""
galaxies = ag.Galaxies(galaxies=[galaxy])

fit = ag.FitImaging(dataset=dataset, plane=plane)

"""
__Pixelization__

The fit has been performed using an `Inversion` for the galaxy.

We can see that the `model_image` of the fit subplot shows a reconstruction of the observed galaxy that is close 
to the data.
"""
fit_plotter = aplt.FitImagingPlotter(
    fit=fit,
    mat_plot_2d=aplt.MatPlot2D(
        output=aplt.Output(path=workspace_path, filename="rectangular", format="png")
    ),
)
fit_plotter.figures_2d_of_galaxies(galaxy_index=0, model_image=True)

"""
__Why Use Pixelizations?__

From the perspective of a scientific analysis, it may be unclear what the benefits of using an inversion to 
reconstruct a complex galaxy are.

When I fit a galaxy with light profiles, I learn about its brightness (`intensity`), size (`effective_radius`), 
compactness (`sersic_index`), etc.

What did I learn about the galaxy I reconstructed? Not a lot, perhaps.

Inversions are most useful when combined with light profiles. For the complex galaxy above, we can fit it with light 
profiles to quantify the properties of its `bulge` and `disk` components, whilst simultaneously fitting the clumps 
with the inversion so as to ensure they do not impact the fit.

The workspace contains examples of how to do this, as well as other uses for pixelizations.

__Wrap Up__

This script gives a brief overview of pixelizations. 

However, there is a lot more to using *Inversions* then presented here. 

In the `autogalaxy_workspace/*/modeling` folder you will find example scripts of how to fit a model to a 
galaxy using an `Inversion`. 

In chapter 4 of the **HowToGalaxy** lectures we fully cover all details of  *Inversions*, specifically:

 - How the inversion's reconstruction determines the flux-values of the galaxy it reconstructs.
 - The Bayesian framework employed to choose the appropriate level of `Regularization` and avoid overfitting noise.
 - Unphysical model solutions that often arise when using an `Inversion`.
"""
