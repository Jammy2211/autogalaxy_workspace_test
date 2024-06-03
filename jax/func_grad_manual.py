"""
Func Grad: Light Parametric Operated
====================================

This script test if JAX can successfully compute the gradient of the log likelihood of an `Imaging` dataset with a
model which uses operated light profiles.

It uses a manually written log likelihood function, which is used to compute the log likelihood of the model-fit.
By writing the log likelihood function manually, we can see and edit every step of the calculation, without
having to navigate and understand the source-code.

 __Operated Fitting__

It is common for galaxies to have point-source emission, for example bright emission right at their centre due to
an active galactic nuclei or very compact knot of star formation.

This point-source emission is subject to blurring during data accquisiton due to the telescope optics, and therefore
is not seen as a single pixel of light but spread over multiple pixels as a convolution with the telescope
Point Spread Function (PSF).

It is difficult to model this compact point source emission using a point-source light profile (or an extremely
compact Gaussian / Sersic profile). This is because when the model-image of a compact point source of light is
convolved with the PSF, the solution to this convolution is extremely sensitive to which pixel (and sub-pixel) the
compact model emission lands in.

Operated light profiles offer an alternative approach, whereby the light profile is assumed to have already been
convolved with the PSF. This operated light profile is then fitted directly to the point-source emission, which as
discussed above shows the PSF features.
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

Load and plot the galaxy dataset `operated` via .fits files, which we will fit with 
the model.
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

The model-fit requires a `Mask2D` defining the regions of the image we fit the model to the data, which we define
and use to set up the `Imaging` object that the model fits.
"""
mask_2d = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

dataset = dataset.apply_mask(mask=mask_2d)

"""
__Model__

We compose our model using `Model` objects, which represent the galaxies we fit to our data. In this 
example we fit a model where:

 - The galaxy's bulge is a parametric `Sersic` bulge [7 parameters]. 
 - The galaxy's point source emission is a parametric operated `Gaussian` centred on the bulge [4 parameters].

The number of free parameters and therefore the dimensionality of non-linear parameter space is N=11.
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
The `info` attribute shows the model in a readable format.
"""
print(model.info)

"""
__Analysis__

The `AnalysisImaging` object defines the `log_likelihood_function` which will be used to determine if JAX
can compute its gradient.
"""
analysis = ag.AnalysisImaging(dataset=dataset)

"""
We are now going to maually write the `log_likelihood_function` that we will use JAX to compute the gradient of.

We from here on import JAX numpy, so that all array-like structures used in the log likelihood function come from
JAX and are therefore compatible with JAX operations.

This is what makes JAX actually get used in the function, so this import is very important. The source code
has a wrapper which automatically switches between JAX and NumPy, depending on whether JAX is installed
and the `USE_JAX` environment variable is set to `1`:


"""
from jax import numpy as np

"""
This is the simplest possible log likelihood function extracted from PyAutoGalaxy, which fits an image of a 
galaxy with a Sersic bulge.
"""
def log_likelihood_function(instance):

    """
    For the input instance, consiting of a Sersic bulge with the 7 free parameters listed by `model.info` above,
    convert this instance in a list of galaxies (a `Galaxies` object).
    """
    galaxies = analysis.galaxies_via_instance_from(instance=instance)

    """
    Extract the `bulge`, whose image we will compute via its light profile.
    """
    bulge = galaxies[0].bulge

    """
    The calculation uses the `Grid2D` object contained in the `Imaging` dataset, which is a Numpy array of
    shape (total_masked_pixels, 2).
    """
    grid = np.array(dataset.grid)

    """
    The code below performs the steps in the function:    
    
        bulge.transformed_from_reference_frame_grid_from(grid=dataset.grid)
    
    Which is found here:
    
    https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py
    
    This function transforms the (y,x) coordinates of the grid to the reference frame of the bulge, which is
    defined by its centre and elliptical geometry. It shifts the coordinates to the centre of the bulge (by subtracting
    the centre) and then rotates them to the geometry of the bulge (by using the elliptical components).    
    
    Uncommenting the code below raises a few exceptions, because it cannot subtract the centre of the bulge
    (which is a tuple of floats) from the grid (which is a JAX Numpy array) and due to another weird issue. These
    would be good first issues to fix to learn how JAX works.
    
    The Grid2D object inherits from the following object:
    
    https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/abstract_ndarray.py
    
    This has lots of Python methods which I believe Rich, the other software developer, wrote in order to fix
    these types of issues (e.g. make it work with tuple subtraction). This is worth a look in order to start
    understanding how this works, and I will get Rich to look in more detail next week.
    """

    shifted_grid_2d = np.subtract(grid, bulge.centre)

    # shifted_grid_2d = grid

    radius = np.sqrt(np.sum(shifted_grid_2d**2.0, 1))
    theta_coordinate_to_profile = np.arctan2(
        shifted_grid_2d[:, 0], shifted_grid_2d[:, 1]

    ) - np.radians(bulge.angle)
    grid = np.vstack(
        radius
        * (np.sin(theta_coordinate_to_profile), np.cos(theta_coordinate_to_profile))
    ).T

    axis_ratio, angle = ag.convert.axis_ratio_and_angle_from(
        ell_comps=bulge.ell_comps
    )

    """
    The code below performs the steps in the function:   

        bulge.elliptical_radii_grid_from 
        
    Which is found here:
    
        https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py
        
    This basically just converts the (y,x) coordinates of the grid to elliptical radii as:
    
    R_ell = y^2 + x^2 / (1 - q^2) 
    
    q is the axis ratio of the bulge.
    """
    grid_radii = np.sqrt(
            np.add(
                np.square(grid[:, 1]), np.square(np.divide(grid[:, 0], axis_ratio))
            )
        )

    """
    The code below performs the steps in the function:   

        bulge.eccentric_radii_grid_from(grid=grid)

    Which is found here:

        https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/geometry_profiles.py

    This basically just converts the elliptical radii grid to the eccentric radii grid as:
    
    R_ecc = sqrt(q^2) * R_ell

    R = y^2 + x^2 / (1 - q^2) 
    """
    grid_radii = np.multiply(np.sqrt(axis_ratio), grid_radii)

    """
    The code below performs the steps in the function:   

        bulge.eccentric_radii_grid_from(grid=grid)

    Which is found here:
    
        https://github.com/Jammy2211/PyAutoGalaxy/blob/main/autogalaxy/profiles/light/standard/sersic.py
        
    The expression below is the 2D Sersic profile, which is the intensity at a given radius R from the centre of the
    bulge. This is computed as:
    
    I(R) = I_0 * exp^(-k * (R / R_e)^(1/n))
    """

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
    The code below computes the log likelihood of the model image, given the dataset and noise-map. This is done
    by computing the residual-map between the model image and observed image and squaring it, summing the
    chi-squareds and computing the overall likelihood.
    
    The code below is taken from the following modules:
    
    https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_dataset.py
    https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/fit/fit_util.py
    """
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
