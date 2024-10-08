"""
Func Grad 5: Linear Solver
==========================

A linear light profile is a light profile whose `intensity` is not manually input by the user nor is it a
parameter that is fitted for by the non-linear search.

Instead, the `intensity` of a linear light profile is determined by solving a linear inversion, which finds
the values of the `intensity` that best-fit the observed image.

A step-by-step guide of the likelihood function of a linear light profile is given here:

https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/linear_light_profile/log_likelihood_function.ipynb

This task converts the linear light profile code to use JAX.

__Linear Solver__

The main challenge with the linear inversion is that after setting up the problem as a few matrices, a linear
inversion is performed to solve the system of equations (e.g. the `intensity` values that best-fit the image).

This can be performed using `np.linalg.solve`, which is supported by JAX and therefore would be an easy conversion.

However, this method allows for both positive and negative values of `intensity` in the solution, which is
unphysical for a light profile. For linear light profiles with good parameters this nearly always works fine and
infers only positive values of `intensity`.

However, more advanced features which will require JAX conversion in the future (e.g. multi-Gaussian expansion,
pixelized source reconstruction) do not give reliable results if negative solutions are allowed. For example, the
MGE may create a positive-negative oscillation in the solution, which is unphysical.

The source code therefore uses a "non negative least squares" method, which is a method that only allows positive
solutions. An example of such a method is found in scipy here:

https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

This method is not supported by JAX, and therefore poses a challenge for the conversion.

However, the source code does not even use the `nnls` method, but instead uses a custom method that is faster,
found at this GitHub repository:

https://github.com/jvendrow/fnnls

This could be even more difficult to convert to JAX.

This example is written assuming N linear light profiles are fitted, we should ensure our JAX
implementation scales to the case of 300 linear light profiles, which is the maximum number of linear light profiles
we realistically expect to fit in a model (e.g. an MGE).

Source code
-----------

The following guide explains where all the relevant functions are located in the source code:

https://github.com/Jammy2211/autogalaxy_workspace/blob/main/notebooks/imaging/advanced/log_likelihood_function/linear_light_profile/contributor_guide.ipynb
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

First, note how we now set up the model using the `lp_linear` module, which means we use a linear light profile
istead of a standard light profile. This light profile does not have an `intensity` parameter.
"""
bulge = af.Model(ag.lp_linear.Sersic)

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
    galaxies = analysis.galaxies_via_instance_from(instance=instance)
    bulge = galaxies[0].bulge

    """
    Create the `LightProfileLinearObjFuncList` that is used to perform the linear inversion.
    """
    lp_linear_func = ag.LightProfileLinearObjFuncList(
        grid=dataset.grids.uniform,
        blurring_grid=dataset.grids.blurring,
        convolver=dataset.convolver,
        light_profile_list=[bulge],
        regularization=None,
    )

    # """
    # Once the issue above is fixed, here is how the 2D convolution is performed:
    # """
    # model_data =
    #
    # """
    # Here is where the 2D convolution is performed
    # """
    # residual_map = dataset.data - model_data
    # chi_squared_map = (residual_map / dataset.noise_map) ** 2.0
    # chi_squared = sum(chi_squared_map)
    # # noise_normalization = np.sum(np.log(2 * np.pi * dataset.noise_map ** 2.0))
    # log_likelihood = -0.5 * (chi_squared)

    # return log_likelihood


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
print(instance.galaxies.galaxy.bulge.effective_radius)
print(instance.galaxies.galaxy.bulge.sersic_index)

"""
The `intensity` of the bulge is set to a default value of 1.0, which will be updated by the linear inversion.
"""
print(instance.galaxies.galaxy.bulge.intensity)

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
