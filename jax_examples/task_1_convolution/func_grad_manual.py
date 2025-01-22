"""
Func Grad 1: Convolution
========================

The examples so far have used an "operated light profile", via the import `lp_operated`.

This means the light profile assumes it has already had PSF convolution performed on it, and it was chosen as
the starting point because 2D convolution is not currently impleneted in the source code in a way that supports
JAX.

By switching from an operated light profile (`lp_oeprated`) to a standard light profile (`lp`) the PSF convolution
code will be called.

The 2D convolution is performed via the `Convolver` object:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/operators/convolver.py

This class performs 2D convolution is a fairly complicated way. For a given 2D mask and 2D PSF kernel, it stores
in memory indexes mapping every convolution step that is performed. This is quite a bit more complex than the standard
2D convolution routines available in Python libraries like NumPy.

The reason the `Convolver` is used is because pre computing the calculation in this way can give significiant efficiency
gains for calculations which use a pixelized grid and linear algebra to reconstruct the image, which is the standard
method for modeling sources in PyAutoLEns. Thus, whilst its a bit of overkill for parametric sources, it allowed us to
reuse existing functionality.

The `Convolver` uses a lot of in-plane memory manipulation and is therefore probably a no-no for JAX. However,
it should be simple enough to add a few extra convolution routines which use JAX convolve functions, and just
have them called when JAX is used. Lets not worry about how this will be extended to linear algebra calculations
for now

The current 2D convolution is performed in real space, we may wish to have both a real space option and FFT option,
for the size of images and kernels we normally deal with real space seems fine.

We need to:

 1) Extend the `Convolver` with JAX supported convolutions.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

import jax
from jax import grad
from os import path

import autofit as af
import autogalaxy as ag
import autogalaxy.plot as aplt
from autoconf import conf

"""
__Dataset__
"""
dataset_name = "operated"
dataset_path = path.join("dataset", "imaging", dataset_name)

dataset = ag.Imaging.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    psf_path=path.join(dataset_path, "psf.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    pixel_scales=0.1,
)

"""
__Mask__
"""
mask_2d = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask_2d)

"""
__Model__

The model now uses a standard light profile via `.lp` and not an operated light profile via `.lp_operated`.
"""
bulge = af.Model(ag.lp.Sersic)

# bulge.ell_comps.ell_comps_0 = af.UniformPrior(lower_limit=0.2, upper_limit=0.4)
# bulge.ell_comps.ell_comps_1 = af.UniformPrior(lower_limit=0.0, upper_limit=0.2)

bulge.centre.centre_0 = 0.1
bulge.centre.centre_1 = 0.1
bulge.ell_comps.ell_comps_0 = 0.1
bulge.ell_comps.ell_comps_1 = 0.2

galaxy = af.Model(ag.Galaxy, redshift=0.5, bulge=bulge)

model = af.Collection(galaxies=af.Collection(galaxy=galaxy))


"""
__Analysis__
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
The log likelihood function below has been adapted from the `start` example to include 2D convolution
"""
from jax import numpy as np


def log_likelihood_function(instance):
    """
    All Code illustrated in the `start` example:
    """
    galaxies = analysis.galaxies_via_instance_from(instance=instance)
    bulge = galaxies[0].bulge
    grid = np.array(dataset.grid)

    shifted_grid_2d = np.subtract(grid, np.array(bulge.centre))

    radius = np.sqrt(np.sum(shifted_grid_2d**2.0, 1))
    theta_coordinate_to_profile = np.arctan2(
        shifted_grid_2d[:, 0], shifted_grid_2d[:, 1]
    ) - np.radians(bulge.angle)
    grid = np.vstack(
        radius
        * np.array(
            (np.sin(theta_coordinate_to_profile), np.cos(theta_coordinate_to_profile))
        )
    ).T

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=bulge.ell_comps)
    grid_radii = np.sqrt(
        np.add(np.square(grid[:, 1]), np.square(np.divide(grid[:, 0], axis_ratio)))
    )
    grid_radii = np.multiply(np.sqrt(axis_ratio), grid_radii)

    image_2d = np.multiply(
        bulge._intensity,
        np.exp(
            np.multiply(
                -bulge.sersic_constant,
                np.add(
                    np.power(
                        np.divide(grid_radii, bulge.effective_radius),
                        1.0 / bulge.sersic_index,
                    ),
                    -1,
                ),
            )
        ),
    )

    """
    To perform 2D convultion, we also need a "blurring image", which is the image of all pixels outside the mask
    whose light blurs into the mask.
    
    This will be input into the Convolber object.
    
    The code below pretty much is the exact same as above, but uses the dataset's blurring grid, which only contains
    image pixels outside the mask but near enough to it that their light blurs into the mask.
    
    The first JAX bug arises, which is because when the blurring grid is made here:
    
    https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/dataset/imaging/dataset.py
    
    In the function
        
        `blurring_grid` -> `blurring_grid_via_kernel_shape_from`

    The grid is passed an `OverSampling` object, which JAX does not like.
        
    This object basically sets over sampling to only use a `sub_size` of 1.    
    """
    grid = np.array(dataset.grids.blurring)

    shifted_grid_2d = np.subtract(grid, np.array(bulge.centre))

    radius = np.sqrt(np.sum(shifted_grid_2d**2.0, 1))
    theta_coordinate_to_profile = np.arctan2(
        shifted_grid_2d[:, 0], shifted_grid_2d[:, 1]
    ) - np.radians(bulge.angle)
    grid = np.vstack(
        radius
        * np.array(
            (np.sin(theta_coordinate_to_profile), np.cos(theta_coordinate_to_profile))
        )
    ).T

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(ell_comps=bulge.ell_comps)
    grid_radii = np.sqrt(
        np.add(np.square(grid[:, 1]), np.square(np.divide(grid[:, 0], axis_ratio)))
    )
    grid_radii = np.multiply(np.sqrt(axis_ratio), grid_radii)

    blurring_image_2d = np.multiply(
        bulge._intensity,
        np.exp(
            np.multiply(
                -bulge.sersic_constant,
                np.add(
                    np.power(
                        np.divide(grid_radii, bulge.effective_radius),
                        1.0 / bulge.sersic_index,
                    ),
                    -1,
                ),
            )
        ),
    )

    """
    Once the issue above is fixed, here is how the 2D convolution is performed:
    """
    model_data = dataset.convolver.convolve_image(
        image=image_2d, blurring_image=blurring_image_2d
    )

    """
    A PSF can also be used to perform convolution (which is what the `Convolver` is derived from:
    """
    model_data = dataset.psf.convolved_array_with_mask_from(
        array=image_2d.native + blurring_image_2d.native,
        mask=image_2d.mask,
    )

    """
    Here is where the 2D convolution is performed
    """
    residual_map = dataset.data - model_data
    chi_squared_map = (residual_map / dataset.noise_map) ** 2.0
    chi_squared = sum(chi_squared_map)
    # noise_normalization = np.sum(np.log(2 * np.pi * dataset.noise_map ** 2.0))
    log_likelihood = -0.5 * (chi_squared)

    return log_likelihood


"""
Create a list of input parameters, which are representative of the parameters a non-linear search would input
to the `log_likelihood_function` during sampling.
"""
parameters = model.physical_values_from_prior_medians
instance = model.instance_from_vector(vector=parameters)

"""
By printing the instance, we can see the values of the model's parameters before the log likelihood function is
computed.

These are fairly arbitrary values, but it is good to know what they are before we compute the log likelihood.
"""
print(instance.galaxies.galaxy.bulge.centre)
print(instance.galaxies.galaxy.bulge.ell_comps)
print(instance.galaxies.galaxy.bulge.intensity)
print(instance.galaxies.galaxy.bulge.effective_radius)
print(instance.galaxies.galaxy.bulge.sersic_index)

"""
compile the gradient of the log likelihood function via JAX.

This is also the function we need to confirm runs on GPU with significant speed-up.
"""
print(parameters)
print(log_likelihood_function(instance))

"""
We now compile the gradient of the fitness function via JAX and pass this gradient the instance of the model,
thereby computing the gradient of every model parameter based on the log likelihood function.
"""
grad = jax.jit(grad(log_likelihood_function))
instance_grad = grad(instance)

"""
Printing each instance parameter's gradient shows that JAX has computed the gradient of the log likelihood.
"""
print(instance_grad.galaxies.galaxy.bulge.centre)
print(instance_grad.galaxies.galaxy.bulge.ell_comps)
print(instance_grad.galaxies.galaxy.bulge.intensity)
print(instance_grad.galaxies.galaxy.bulge.effective_radius)
print(instance_grad.galaxies.galaxy.bulge.sersic_index)
print(dir(instance_grad))


"""
Checkout `autogalaxy_workspace/*/imaging/modeling/results.py` for a full description of the result object.
"""
