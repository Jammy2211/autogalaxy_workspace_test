"""
Func Grad 0: Radial Grid
========================

When evaluating a light or mass profile, if the input grid value is (0.0, 0.0), meaning that its radial value r = 0.0,
an exception may be raised of an incorrect value may be computed. Values numerically very close to
zero (e.g. r < 1.0e-8) can also cause problems.

Whether or not this occurs depends on the mass or light profile, for example if it calls functions like sin,
cos, tan, hyper geometric functions, etc, which may break for an input near zero.

To mitigate this, the current implementation does the following:

 1) For an input grid, computes the radial values which are input into the function that evaluates the light or
    mass profile (e.g. image_2d_from()).

 2) If any value is less than a threshold value (e.g. 1.0e-8), it is rounded to this threshold value. The value is
 determined from the config file `config/grids.yaml`).

An example of the config file is found here:

 https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/config/grids.yaml

The code which performs this grid round is found here, implemted via a decorator which decorates nearly every
function of the light and mass profiles. Note how currently on the jax_wrapper branch this code is disabled:

https://github.com/Jammy2211/PyAutoArray/blob/feature/jax_wrapper/autoarray/structures/decorators/relocate_radial.py

We need to:

 1) Confirm that this rounding is a sane approach to take.
 2) Get this to support JAX.
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
bulge = af.Model(ag.lp_operated.Sersic)

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
The log likelihood function below has been adapted from the `start` example to include the grid relocation.
"""
from jax import numpy as np


def log_likelihood_function(instance):
    """
    All Code illustrated in the `start` example:
    """
    galaxies = analysis.galaxies_via_instance_from(instance=instance)
    bulge = galaxies[0].bulge
    grid = np.array(dataset.grid)

    """
    Here is where we attempt the grid relocation, which JAX currently does not like.
    """
    obj = bulge

    grid_radial_minimum = conf.instance["grids"]["radial_minimum"]["radial_minimum"][
        obj.__class__.__name__
    ]

    grid_radii = obj.radial_grid_from(grid=grid)

    grid_radial_scale = np.where(
        grid_radii < grid_radial_minimum, grid_radial_minimum / grid_radii, 1.0
    )
    moved_grid = np.multiply(grid, grid_radial_scale[:, None])

    if hasattr(grid, "with_new_array"):
        moved_grid = grid.with_new_array(moved_grid)

    grid = moved_grid

    """
    All Code illustrated in the `start` example:
    """
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

    model_data = image_2d
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
