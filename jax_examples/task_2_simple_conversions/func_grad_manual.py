"""
Func Grad 2: Simple Conversions
===============================

There are a number of functions which use numba for loops, which we need to convert to numpy arithmitic in order
to support JAX.

This is a good way to learn some of the basics of JAX, as well as how to interact with the source code and put up a
a pull request to have code merged into the main branch.

I will start with a function which computes a 2D mask, explaining it in detial, and then point you to
a few more functions which it would be useful to have converted.

Once familair with this from this tutorial we can move on to something more interesting!
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

This function, which creates the 2D mask, is the first function we will target speeding up.

Currently, with JAX, the code below runs fine. This is because when the USE_JAX enviroment variable is 1,
numba is disabled and the function runs as normal.

However, it will be significantly slower than normal because numba is disabled, and it is impleneted
in a slow way (see below).
"""
mask_2d = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)

"""
We can print the mask to see it is a 2D array of numpy bools, wherw a `False` value indicates it is within the 3.0"
circle:
"""
print(mask_2d)

"""
Here is the function that is actually called to create the mask, which I have copy and pasted from the following
part of the source code:

https://github.com/Jammy2211/PyAutoArray/blob/main/autoarray/mask/mask_2d_util.py
"""
import numpy as np
from typing import Tuple

from autoarray import type as ty

# @numba_util.jit()
def mask_2d_centres_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    centre: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Returns the (y,x) scaled central coordinates of a mask from its shape, pixel-scales and centre.

    The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the scaled centre is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D array.
    centre : (float, flloat)
        The (y,x) centre of the 2D mask.

    Returns
    -------
    tuple (float, float)
        The (y,x) scaled central coordinates of the input array.

    Examples
    --------
    centres_scaled = centres_from(shape=(5,5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    """
    y_centre_scaled = (float(shape_native[0] - 1) / 2) - (centre[0] / pixel_scales[0])
    x_centre_scaled = (float(shape_native[1] - 1) / 2) + (centre[1] / pixel_scales[1])

    return (y_centre_scaled, x_centre_scaled)

# @numba_util.jit()
def mask_2d_circular_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    radius: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns a circular mask from the 2D mask array shape and radius of the circle.

    This creates a 2D array where all values within the mask radius are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native: Tuple[int, int]
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    radius
        The radius (in scaled units) of the circle within which pixels unmasked.
    centre
            The centre of the circle used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a circle.

    Examples
    --------
    mask = mask_circular_from(
        shape=(10, 10), pixel_scales=0.1, radius=0.5, centre=(0.0, 0.0))
    """

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)

            if r_scaled <= radius:
                mask_2d[y, x] = False

    return mask_2d

"""
We can immediately see that this function will be quite slow in native Python, because it uses a double for loop.

For the current source code implementation, it would not be slow, because it uses `numba` via the 
`@numba_util.jit()` decorator I have commented out above.

If we try and use `jax.np` apply the JAX function `grad` on the above function we get an error:
"""

from jax import numpy as np

# @numba_util.jit()
def mask_2d_centres_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    centre: Tuple[float, float],
) -> Tuple[float, float]:
    """
    Returns the (y,x) scaled central coordinates of a mask from its shape, pixel-scales and centre.

    The coordinate system is defined such that the positive y axis is up and positive x axis is right.

    Parameters
    ----------
    shape_native
        The (y,x) shape of the 2D array the scaled centre is computed for.
    pixel_scales
        The (y,x) scaled units to pixel units conversion factor of the 2D array.
    centre : (float, flloat)
        The (y,x) centre of the 2D mask.

    Returns
    -------
    tuple (float, float)
        The (y,x) scaled central coordinates of the input array.

    Examples
    --------
    centres_scaled = centres_from(shape=(5,5), pixel_scales=(0.5, 0.5), centre=(0.0, 0.0))
    """
    y_centre_scaled = (float(shape_native[0] - 1) / 2) - (centre[0] / pixel_scales[0])
    x_centre_scaled = (float(shape_native[1] - 1) / 2) + (centre[1] / pixel_scales[1])

    return (y_centre_scaled, x_centre_scaled)

# @numba_util.jit()
def mask_2d_circular_from(
    shape_native: Tuple[int, int],
    pixel_scales: ty.PixelScales,
    radius: float,
    centre: Tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Returns a circular mask from the 2D mask array shape and radius of the circle.

    This creates a 2D array where all values within the mask radius are unmasked and therefore `False`.

    Parameters
    ----------
    shape_native: Tuple[int, int]
        The (y,x) shape of the mask in units of pixels.
    pixel_scales
        The scaled units to pixel units conversion factor of each pixel.
    radius
        The radius (in scaled units) of the circle within which pixels unmasked.
    centre
            The centre of the circle used to mask pixels.

    Returns
    -------
    ndarray
        The 2D mask array whose central pixels are masked as a circle.

    Examples
    --------
    mask = mask_circular_from(
        shape=(10, 10), pixel_scales=0.1, radius=0.5, centre=(0.0, 0.0))
    """

    mask_2d = np.full(shape_native, True)

    centres_scaled = mask_2d_centres_from(
        shape_native=mask_2d.shape, pixel_scales=pixel_scales, centre=centre
    )

    for y in range(mask_2d.shape[0]):
        for x in range(mask_2d.shape[1]):
            y_scaled = (y - centres_scaled[0]) * pixel_scales[0]
            x_scaled = (x - centres_scaled[1]) * pixel_scales[1]

            r_scaled = np.sqrt(x_scaled**2 + y_scaled**2)

            if r_scaled <= radius:
                mask_2d[y, x] = False

    return mask_2d


grad = jax.jit(grad(mask_2d_circular_from))
grad(shape_native=(100, 100), pixel_scales=(0.1, 0.1), radius=3.0)


"""
Basically, the task is as follows:

1) Convert the above function to only use numpy arithmitic that is supported in 
JAX (see https://jax.readthedocs.io/en/latest/jax.numpy.html)

2) Implemented this in the source code and make sure that the following code below runs without error.

3) Raise a PR on the feature/jax_wrapper branch with the update.
"""
mask_2d = ag.Mask2D.circular(
    shape_native=dataset.shape_native, pixel_scales=dataset.pixel_scales, radius=3.0
)