"""
Func Grad 4: Over Sampling Uniform
==================================

Over sampling is explained in detail in the following notebook:

https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/guides/over_sampling.ipynb

This task converts the over sampling code to use JAX.

Source code
-----------

Over sampling is implemented in the following part of autoarray via the following decorators:

https://github.com/Jammy2211/PyAutoArray/tree/feature/jax_wrapper/autoarray/operators/over_sampling

https://github.com/Jammy2211/PyAutoArray/blob/feature/jax_wrapper/autoarray/operators/over_sampling/decorator.py

For now, we only care about uniform over sampling, which is implemented at:

https://github.com/Jammy2211/PyAutoArray/blob/feature/jax_wrapper/autoarray/operators/over_sampling/uniform.py

This example will first perform uniform over samplinmg, which does not adapt the level of over sampling
to the light profile being evaluated.

The follow up tutorial covers adaptive over sampling, which does adapt the level of over sampling to the light profile
being evaluated.

The `over_sampler` object introduced below, in this example, uses a single value of `sub_size` for all pixels.
`sub_size` is the number of sub-pixels that each pixel of the grid is divided into when computing the light profile,
so for `sub_size=2` each pixel is divided into 2 x 2 = 4 sub-pixels.

However, the `sub_size` input is converted to an ndarray whose shape matches the data, and all calculations using
the `sub_size` assume it is an ndarray. This means that the calculations support a different level of over sampling
in each pixel, which is what adaptive over described in the next example does.

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
The log likelihood function below has been adapted to include over sampling.
"""
from jax import numpy as np


def log_likelihood_function(instance):

    """
    Create the uniform over sampling object, which is used to create the over sampled grid on which the light profile
    is evaluated.
    """
    over_sampling = ag.OverSamplingUniform(sub_size=2)

    over_sampler = over_sampling.over_sampler_from(mask=dataset.mask)

    # The over sampled grid is used to evaluate the light profile, and it is a property of the over_sampler.
    # This called a function which uses numba, titled `grid_2d_slim_over_sampled_via_mask_from`, and is therefore
    # the first function that will need to be converted to JAX.

    over_sampled_grid = over_sampler.over_sampled_grid


    galaxies = analysis.galaxies_via_instance_from(instance=instance)
    bulge = galaxies[0].bulge
    grid = np.array(over_sampled_grid)

    """
    All Code illustrated in the `start` example,although it now uses the over sampled grid.
    """
    shifted_grid_2d = np.subtract(grid, np.array(bulge.centre))

    radius = np.sqrt(np.sum(shifted_grid_2d ** 2.0, 1))
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
    The `image_2d` is computed using the over sampled grid, and therefore the `image_2d` is over sampled
    and has a much higher resolution than the image data.
    
    The `over_sampler` object also has a function reduces the over sampled image to the image data's resolution.
    
    This calls the numba function `binned_array_2d_from` which will also need to be converted to JAX.
    """
    image_2d = over_sampler.binned_array_2d_from(array=image_2d)

    """
    The code below performs 2D convolution, which is now already converted to JAX and therefore should work.
    I've included it for completeness, but it is not the focus of this example and does not depend on over sampling.
    
    The `blurring_grid`, which is all pixels outside the mask whose light blurs into the mask, is not over sampled.
    """
    grid = np.array(dataset.grids.blurring)

    shifted_grid_2d = np.subtract(grid, np.array(bulge.centre))

    radius = np.sqrt(np.sum(shifted_grid_2d ** 2.0, 1))
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
