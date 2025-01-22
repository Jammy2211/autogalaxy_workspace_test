"""
Func Grad 3: Interferometer
===========================

One of the biggest bottlenecks is the multiplication of 3 matrices, with a central matrix of size N x N,
which is multiplied by a matrix of size M x N and its transpose.

For a typical use case, N = 10000 (or more) and M = 1000 (or more), which is a huge matrix multiplication.
Both matrices are fully dense, meaning the matrix multiplication is computationally expensive.

The matrix is especially important for interferometer datasets, with this task being about JAX-ifying
the whole likelihood calculation for an interferometer dataset.

This task profile and speed up the matrix multiplication using JAX, and then JAX-ify's other functions which
make up the likelihood calculation.
"""
# %matplotlib inline
# from pyprojroot import here
# workspace_path = str(here())
# %cd $workspace_path
# print(f"Working Directory has been set to `{workspace_path}`")

try:
    import jax
    from jax import grad
    from jax import numpy as np
except ImportError:
    import numpy as np

from os import path

import autogalaxy as ag


"""
__Mask__

We define the ‘real_space_mask’ which defines the grid the image the galaxy is evaluated using.
"""
real_space_mask = ag.Mask2D.circular(
    shape_native=(128, 128),
    pixel_scales=(0.03625, 0.03625),
    radius=3.0,
)

"""
__Dataset__

Load and plot the galaxy `Interferometer` dataset `simple__sersic` from .fits files, which we will fit
with the model.

This includes the method used to Fourier transform the real-space image of the galaxy to the uv-plane and compare
directly to the visibilities. We use a non-uniform fast Fourier transform, which is the most efficient method for
interferometer datasets containing ~1-10 million visibilities. We will discuss how the calculation of the likelihood
function changes for different methods of Fourier transforming in this guide.
"""
dataset_name = "simple"
dataset_path = path.join("dataset", "interferometer", dataset_name)

dataset = ag.Interferometer.from_fits(
    data_path=path.join(dataset_path, "data.fits"),
    noise_map_path=path.join(dataset_path, "noise_map.fits"),
    uv_wavelengths_path=path.join(dataset_path, "uv_wavelengths.fits"),
    real_space_mask=real_space_mask,
    transformer_class=ag.TransformerDFT,
    over_sample_size_lp=1,
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
__W Tilde__

The calculation uses a fully dense matrix, which is the central matrix in the multiplication of 3 matrices.
We refer to it as `w_tilde`, which is the algebraic representation of the matrix in papers.

The calculation of this matrix uses the function below.

However, this task IS NOT about JAX-ifying the function below. This is because the function is called once, before
we begin a fitting procedure, and therefore it does not need to support JAX in order for our fitting procedure to
be fast.

Furthermore, the function is quite computationally expensive and currently is only fast when it is decorated with
numba. Because your enviroment is running JAX, `numba` is disabled, meaning this function could take hours to run.


I have therefore instead computed w_tilde using `np.random`. Whilst this is not the correct `w_tilde` matrix, it
will allow us to profile the matrix multiplication below without having to wait hours for the function to run.

It is worth inspecting the function below though, because converting it to JAX will, in the long term, be a task
we will want to do. This is because in the long term the goal is to drop numba support and have all functions
support JAX.
"""


def w_tilde_curvature_interferometer_from(
    noise_map_real: np.ndarray,
    uv_wavelengths: np.ndarray,
    grid_radians_slim: np.ndarray,
) -> np.ndarray:
    """
    The matrix w_tilde is a matrix of dimensions [image_pixels, image_pixels] that encodes the NUFFT of every pair of
    image pixels given the noise map. This can be used to efficiently compute the curvature matrix via the mappings
    between image and source pixels, in a way that omits having to perform the NUFFT on every individual source pixel.
    This provides a significant speed up for inversions of interferometer datasets with large number of visibilities.

    The limitation of this matrix is that the dimensions of [image_pixels, image_pixels] can exceed many 10s of GB's,
    making it impossible to store in memory and its use in linear algebra calculations extremely. The method
    `w_tilde_preload_interferometer_from` describes a compressed representation that overcomes this hurdles. It is
    advised `w_tilde` and this method are only used for testing.

    Parameters
    ----------
    noise_map_real
        The real noise-map values of the interferometer data.
    uv_wavelengths
        The wavelengths of the coordinates in the uv-plane for the interferometer dataset that is to be Fourier
        transformed.
    grid_radians_slim
        The 1D (y,x) grid of coordinates in radians corresponding to real-space mask within which the image that is
        Fourier transformed is computed.

    Returns
    -------
    ndarray
        A matrix that encodes the NUFFT values between the noise map that enables efficient calculation of the curvature
        matrix.
    """

    w_tilde = np.zeros((grid_radians_slim.shape[0], grid_radians_slim.shape[0]))

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            y_offset = grid_radians_slim[i, 1] - grid_radians_slim[j, 1]
            x_offset = grid_radians_slim[i, 0] - grid_radians_slim[j, 0]

            for vis_1d_index in range(uv_wavelengths.shape[0]):
                w_tilde[i, j] += noise_map_real[vis_1d_index] ** -2.0 * np.cos(
                    2.0
                    * np.pi
                    * (
                        y_offset * uv_wavelengths[vis_1d_index, 0]
                        + x_offset * uv_wavelengths[vis_1d_index, 1]
                    )
                )

    for i in range(w_tilde.shape[0]):
        for j in range(i, w_tilde.shape[1]):
            w_tilde[j, i] = w_tilde[i, j]

    return w_tilde


"""
Here is how the function above is called, although it is commented out because we instead use `np.random`.
"""
# w_tilde = w_tilde_curvature_interferometer_from(
#     noise_map_real=dataset.noise_map,
#     uv_wavelengths=dataset.uv_wavelengths,
#     grid_radians_slim=dataset.grid.in_radians
# )

"""
We now use `np.random` to create a random `w_tilde` matrix, which will allow us to profile the matrix multiplication.

We define N, which is the number of image pixels in the `real_space_mask`, which for the starting dataset is 16132.

Note that the JAX numpy wrapper does not support `np.random`, so we use normal numpy to create the random matrix
and then convert it to a JAX array.
"""
N = real_space_mask.pixels_in_mask

import numpy as np_not_jax

w_tilde = np_not_jax.random.normal(size=(N, N))

w_tilde = np.array(w_tilde)

print(w_tilde)

"""
Printing the shape of `w_tilde` confirms it is a 2D matrix of shape [N, N].
"""
print(w_tilde.shape)

"""
__Light Profile__

The above matrix is multiplied either side by the Sersic light profile evaluated using the real-space grid.
"""
light = ag.lp.Sersic(
    centre=(0.0, 0.0),
    ell_comps=(0.2, 0.2),
    intensity=1.0,
    effective_radius=1.0,
    sersic_index=1.0,
)

image = light.image_2d_from(grid=dataset.grid)

"""
__linear Algebra__

Below, I rename the `image` to `mapping_matrix`, which seems like a strange name given the `image` is not a
`matrix` and it is not mapping anything.

The reason for this name change is because when `w_tilde` is used in the context of the full linear algebra
calculation, it is multiplied by the `mapping_matrix` (which is the image of the light profile).

I have so far written the examples so far to avoid making the linear algebra calculation explicit, as it is
conceptually complex and will be introduced once you are more familiar with the code base.

However, I use `mapping_matrix` below to make it clear that when you begin using the linear algebra calculation
this is where it fits in.
"""
mapping_matrix = image

"""
__Dimensions__

Lets quickly consider the shapes of our matrices:

- In this example, the `mapping_matrix` is a 1D matrix of shape [M, N].
- The `w_tilde` matrix is a 2D matrix of shape [N, N].

Where:

- M = 1, corresponding to a single Serisc light profile.
- N=16132 and is the number of image pixels in the `real_space_mask`.

It should be noted that for realistic science cases, N could go as large as 50 0000 and M could go as large as 1000.
This is a huge dimensionality and the matrix multiplication of `w_tilde` with the `mapping_matrix`!
"""
print(mapping_matrix.shape)
print(w_tilde.shape)

"""
__Matrix Multiplication__

We now perform the triple matrix multiplication of `w_tilde` with the `mapping_matrix` and its transpose.

Again, keep in mind how if N and M are large this is a very computationally expensive calculation.
"""
mapping_matrix = np.array(mapping_matrix.array)
w_tilde = np.array(w_tilde)


def matrix_multiplication(matrix, w_tilde):
    return np.dot(matrix.T, np.dot(w_tilde, matrix))


curvature_matrix = matrix_multiplication(mapping_matrix, w_tilde)

"""
The function above just uses `np.dot` to perform the matrix multiplication, which is natively supported by JAX.

The function can therefore be JAX-ified using the `jax.jit` decorator.
"""
try:
    jitted = jax.jit(matrix_multiplication)
    jitted(mapping_matrix, w_tilde)
except ImportError:
    pass

"""
__First Task__

The first task is to profile the matrix multiplication above and understand how the run times scale with JAX.

Profiling should also account for higher dimension `mapping_matrix`, which can be achieved stacking the `mapping_matrix`
to create a 2D matrix of shape [M, N].

One thing to note about JAX is that when it uses `jit` to speed up a function, this can be performed on many functions
at once. In the case of the matrix multiplication above, the `jax.jit` is only used on the `matrix_multiplication`,
meaning it has limited scope for speeding up the overall calculation.

The implementation we hope to have at the end of this will be to use `jax.jit` on a sequence of ~10 functions where
the matrix multiplication is just one of them. This will allow JAX to optimize the entire sequence of functions
and not just the matrix multiplication, producing a much faster calculation.

Therefore, do not worry too much about the speed-up of the matrix multiplication above yet.

The full seuqence of functions will be introduced in the next task.
"""
M = 100
mapping_matrix = np.stack([mapping_matrix for _ in range(M)])

print(mapping_matrix.T.shape)

curvature_matrix = matrix_multiplication(mapping_matrix.T, w_tilde)

"""
__Second Task__

The second task brings in a number of other functions that in the source code calculation are used before
and after the w_tilde matrix multiplication above.

These all need to be JAX-ified and profiled to understand how they scale with JAX.

They look simple to JAX-ify -- they just use `np.multiply` and `np.dot` which are natively supported by JAX.
"""

# NOTE:
chi_squared_term_1 = np.linalg.multi_dot(
    [
        mapping_matrix,  # NOTE: shape = (N, )
        w_tilde,  # NOTE: shape = (N, N)
        mapping_matrix,
    ]
)

# NOTE: This array is pre-computed and then loaded. Use np.random for now
d = np_not_jax.random.normal(size=(N,))

# NOTE:
chi_squared_term_2 = -np.multiply(2.0, np.dot(mapping_matrix, d))

"""
Basically you just need to put your JAX implementation of the functions above into the functions below
and then make sure they run correctly after jit.
"""


def chi_squared_term_1():
    pass


def chi_squared_term_2():
    pass


try:
    jitted = jax.jit(chi_squared_term_1)
    jitted()
except ImportError:
    pass

try:
    jitted = jax.jit(chi_squared_term_1)
    jitted()
except ImportError:
    pass


"""
Once all functions are jitted we should chat and begin to put them together into a single jitted function.
"""
