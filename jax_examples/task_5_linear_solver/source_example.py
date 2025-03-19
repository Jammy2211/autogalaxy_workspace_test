"""
This example uses the fast non negative least squares method to solve for a linear system that includes
both a Multi-Gaussian Expansion (MGE) and a pixelized source reconstruction.

It allows one to profile the run times for a realistic linear inversion, which fits both the lens light with an
MGE and source with a pixelized source reconstruction. It allows you to do this without fully understanding
the source pixelization calculation fully yet, which is complex and we'll get to later.

You'll need to download the following .npy files for the linear matrices used in this example from the following
Google Drive folder:

https://drive.google.com/file/d/1UboXEakk3TX58VcX9CScswsZ7l-nyaQn/view?usp=sharing
"""
import numpy as np
import pathlib
import matplotlib.pyplot as plt

import autolens as al

matrices_path = pathlib.Path("jax_examples/task_5_linear_solver/matrices")

"""
Load the data and noise-map that was fitted, which is used to create the `data_vector` that is solved for.

The `shape` of the `data` and `noise_map` show there are 11304 image pixels in the data that was fitted.
"""
data = np.load(matrices_path / "data.npy")
noise_map = np.load(matrices_path / "noise_map.npy")

"""
Load the `blurred_mapping_matrix` and inspect its shape.
"""
blurred_mapping_matrix = np.load(matrices_path / "blurred_mapping_matrix.npy")

print(blurred_mapping_matrix.shape)

"""
The shape is (11304, 1145), which indicates there are 11304 image pixels in the data that was fitted and 1145 linear 
parameters.

The model fitted included 180 Gaussians and 965 source pixels, which a plot of the `blurred_mapping_matrix` makes
clear because the first 180 columns are fully dense (corresponding to the light value of every Gaussian) and the
remaining columns are sparsely populated (corresponding to the source pixel values).
"""
plt.figure(figsize=(18, 6))
plt.imshow(blurred_mapping_matrix, aspect="auto")
plt.show()
plt.savefig("blurred_mapping_matrix.png")

"""
The `data_vector` contains the data that was fitted, and it has dimensions 1145 corresponding to the number of linear
parameters in the model.
"""
data_vector = np.load(matrices_path / "data_vector.npy")

print(data_vector.shape)

"""
We can optionally validate that the `data_vector` computed via the source code is the same as the `data_vector` computed
via the JAX code.
"""
# data_vector = al.util.inversion_imaging.data_vector_via_blurred_mapping_matrix_from(
#     blurred_mapping_matrix=blurred_mapping_matrix,
#     image=np.array(data),
#     noise_map=np.array(noise_map),
# )

"""
We can optionally validate that the `data_vector` computed via the numpy artimetic is the same as the `data_vector`
computed above.
"""
# data_vector = np.dot(blurred_mapping_matrix.T, data / noise_map ** 2.0)

"""
The `curvature_reg_matrix` contains the curvature matrix of the linear system, which is used to solve for the linear
parameters. It has dimensions 1145 again corresponding to the number of linear parameters in the model.

Whereas before we have referred to this matrix as the `curvature_matrix`, we have added the `_reg_` tag to indicate
that it is the curvature matrix with regularization applied. This regularization is necessary to ensure the matrix is
positive-definite and thus solvable.
"""
curvature_reg_matrix = np.load(matrices_path / "curvature_reg_matrix.npy")

#print(curvature_reg_matrix.shape)

"""
We can optionally validate that the `curvature_reg_matrix` computed via the source code is the same as the
`curvature_reg_matrix` computed via the JAX code.

Note that this also requires us to load and apply the regularization matrix associated with the pixelization,
which we have not described in the other example scripts for simplicity. Basically, just ignore it for now
but know it exists.
"""
# curvature_reg_matrix = al.util.inversion.curvature_matrix_via_mapping_matrix_from(
#     mapping_matrix=blurred_mapping_matrix,
#     noise_map=noise_map,
#     add_to_curvature_diag=True,
#     no_regularization_index_list=list(range(180)),
# )

# curvature_reg_matrix += regularization_matrix

"""
We can optionally validate that the `curvature_reg_matrix` computed via the numpy artimetic is the same as the
`curvature_reg_matrix` computed above.

This is actually the same as the function `curvature_matrix_via_mapping_matrix_from` above.
"""
# array = blurred_mapping_matrix / noise_map[:, None]
# curvature_reg_matrix = np.dot(array.T, array)
# curvature_reg_matrix[0:180, 0:180] += np.diag(np.full(180, 0.001))
#
# regularization_matrix = np.load(matrices_path / "regularization_matrix.npy")
#
# curvature_reg_matrix += regularization_matrix

"""
The purpose of this script is to profile the run-time of the fast non negative least squares method now implemented
in JAX.

Below, I use the current implementation of the fast non negative least squares method in the source code to solve
the linear system and profile its run-time.
"""
reconstruction = al.util.inversion.reconstruction_positive_only_from(
    data_vector=data_vector,
    curvature_reg_matrix=curvature_reg_matrix,  # ignore _reg_ tag in this guide
)

print(reconstruction)

"""
__Reconstruction__

Compuare to the reconstruction computed via the source code.
"""
reconstruction_truth = np.load(matrices_path / "reconstruction.npy")

print(reconstruction_truth)

print(np.max(np.abs(reconstruction - reconstruction_truth)))


"""
__Likelihood__

The following code computes the likelihood of the model data given the reconstruction, which is the overall
figure-of-merit the `fnnls` is trying to maximize.

It is composed of three terms:

- `chi_squared` is the difference between the model data and the data, which is weighted by the noise-map.

- `noise_normalization` is the overall normalization term for the noise in the data.

- `regularization_term` is the regularization term that is applied to the solution to prevent over-fitting, which is
  the sum of the difference between fluxes of all source-pixels.
"""
# Chi Squared

mapped_reconstructed_image_2d = (
    al.util.inversion.mapped_reconstructed_data_via_mapping_matrix_from(
        mapping_matrix=blurred_mapping_matrix, reconstruction=reconstruction
    )
)

model_image = mapped_reconstructed_image_2d

residual_map = data - model_image
normalized_residual_map = residual_map / noise_map
chi_squared_map = normalized_residual_map**2.0

chi_squared = np.sum(chi_squared_map)

# Regularization

regularization_matrix = np.load(matrices_path / "regularization_matrix.npy")

regularization_term = np.matmul(
    reconstruction.T, np.matmul(regularization_matrix, reconstruction)
)

# Noise Normalization

noise_normalization = float(np.sum(np.log(2 * np.pi * noise_map**2.0)))

# Overall vluoe of the likelihood

log_likelihood = -0.5 * (chi_squared + regularization_term + noise_normalization)


print(log_likelihood)
