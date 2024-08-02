"""
Func Grad 4: Over Sampling
==========================

After implementing uniform over sampling, its now time to adapt the level of over sampling to the light profile being
evaluated.

Source code
-----------

Adaptive over sampling is pretty much all handled in the following decorator.

https://github.com/Jammy2211/PyAutoArray/blob/feature/jax_wrapper/autoarray/operators/over_sampling/decorator.py

This is the code that is important:

        if isinstance(grid, Grid2D):
            if grid.over_sampling is None:
                if grid.is_uniform:
                    over_sampling = OverSamplingUniform.from_adaptive_scheme(
                        grid=grid,
                        name=obj.__class__.__name__,
                        centre=obj.centre,
                    )

                    grid = Grid2D(
                        values=grid, mask=grid.mask, over_sampling=over_sampling
                    )

Basically, if a user inputs a 2D grid, has not manually set the over sampling, and the grid is uniform, then adaptive
over sampling is performed, which is the default behavior.

The method `from_adaptive_scheme` uses the light profile centre and  config files to determine the level of over
sampling that should be used for the light profile. Typically, inner 3 * pixels receive over sampling of 32 x 32,
whereas the outer regions receive 4 x 4, and very outer regions receive no over sampling.

This calls a numba function `sub_size_radial_bins_from` which will need to be converted to JAX.

In the previous example the `over_sampler` object was uniform meaning that the same `sub_size` was used for all pixels.
However, the calculations all supported the `sub_size` being an ndarray with different values for each pixel.
This means that adaptive over sampling is supported.

Array Sizes
-----------

This is the first example where a numpy array in a JAX calculation is not necessarily a constant size.

The size of the over sampling array is determined by the size of the data and the light profile centre. If the
centre moves close to the edge of the mask of the data, the number of sub pixels used to perform the calculation
decreases, because pixels with high numbers of over sampled pixels hit the edge.

For simple single galaxy analyses generalizing adaptive over sampling to JAX is ok, but requires some thought. One
option would be to have a check which ensures that the light profile centre is not too close to the edge of the mask
and raises an error if it is. An alternative is to not support adaptive over sampling in JAX, and instead require the
user to manually set the over sampling before modeling.

The picture becomes more complex when we consider multiple galaxies, where the light profile centre of one galaxy
may be quite different to the other. Adaptive over sampling can adapt the over sampling of each galaxy's light profile
independently, which is a powerful feature. However, ensuring the array sizes do not change becomes quite complex.
It is also a lot more difficult for a user to specify the over sampling of each galaxy manually.

I am open to suggestions on how to handle this, but for now we'll just focus on the single galaxy case.

Lensing
-------

Adaptive over sampling depends on the input 2D grid being uniform, so that its pixels can be uniformly sub-sampled
and evaluated.

For a strongly lensed source, this is not the case, as the source's light is deflected by the lens galaxy's mass.
Lensing calculations therefore do not support adaptive over sampling, however a user can manually set the over sampling
`sub_size` array before lens modeling and this will be used in the lensing calculations.

The lens galaxy's light is evaluate on a uniform grid (assuming no foreground deflectors), so adaptive over sampling
is used for the lens galaxy's light profile.
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

    """
    All Code illustrated in the `start` example,although it now uses the over sampled grid.
    """
    galaxies = analysis.galaxies_via_instance_from(instance=instance)
    bulge = galaxies[0].bulge

    grid = dataset.grid

    over_sampling = ag.OverSamplingUniform.from_adaptive_scheme(
        grid=grid,
        name=bulge.__class__.__name__,
        centre=bulge.centre,
    )

    grid = ag.Grid2D(
        values=grid, mask=grid.mask, over_sampling=over_sampling
    )

    over_sampled_grid = grid.over_sampler.over_sampled_grid
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
    As discussed previously, because the `sub_size` array is an ndarray of variable sizes, the same
    `binned_array_2d_from` function can be used to convert the 2D array to a 2D binned array.    
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
