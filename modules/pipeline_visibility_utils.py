from typing import Tuple, Union

import numpy as np


def transmission(depth: np.ndarray, beta: Union[float, np.ndarray]):
    """
    Compute the transmission map for a given image and depth map.
    """
    return np.exp(-beta * depth)


def haze(image: np.ndarray, depth: np.ndarray, beta: Union[float, np.ndarray], alpha=None):
    """
    Compute the haze image for a given image and depth map.
    """
    alpha = alpha or np.max(image)
    trans = transmission(depth, beta)
    return np.nan_to_num(image * trans + alpha * (1 - trans))


def sample_covariance(smooth=True):
    """
    Sample a random covariance matrix that is roughly isotropic.
    """
    A = np.random.rand(2, 2)
    if smooth:
        A = (A + A.T) / 2
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvalues_mean = np.mean(eigenvalues)
        A = eigenvectors @ np.diag([eigenvalues_mean, eigenvalues_mean]) @ eigenvectors.T
    L = np.linalg.cholesky(np.dot(A, A.T))
    return np.dot(L, L.T)


def sample_gaussian(image_shape: Tuple[int, int], max_scale=0.1, covariance_scale=512):
    """
    Sample a Gaussian distribution with a random mean and covariance matrix.
    """
    # sample attributes
    pixel = np.random.randint(0, image_shape[1]), \
            np.random.randint(0, image_shape[0])
    scale = np.random.random() * max_scale
    covariance_matrix = sample_covariance() * covariance_scale

    # Gaussian synthesis
    x = np.linspace(0, image_shape[1], image_shape[1])
    y = np.linspace(0, image_shape[0], image_shape[0])
    d = np.dstack(np.meshgrid(x, y))

    mean = np.array(pixel)
    gaussian = np.exp(-0.5 * np.sum((d - mean) @ np.linalg.inv(covariance_matrix) * (d - mean), axis=2))
    gaussian = gaussian * scale
    return gaussian


def gaussian_source_sink(image_shape: Tuple[int, int], beta: float, num_gaussians=32, source_sink_ratio=0.5, max_scale=0.1):
    """
    Add Gaussian noise at source locations and subtract Gaussian noise at sink locations.
    """
    if isinstance(beta, float):
        beta = np.ones(image_shape) * beta
    
    transformed_beta = beta
    for i in range(int(num_gaussians * source_sink_ratio)):
        transformed_beta += sample_gaussian(image_shape, max_scale=max_scale)
    for i in range(int(num_gaussians * (1 - source_sink_ratio))):
        transformed_beta -= sample_gaussian(image_shape, max_scale=max_scale)
    transformed_beta = np.clip(transformed_beta, 0, 10)
    return transformed_beta
    

if __name__ == '__main__':
    from PIL import Image

    image = Image.open('/home/gtangg12/auto-augment/tests/example_output1.png')
    image = np.array(image).transpose(2, 0, 1)
    image = image / 255
    depth = np.load('/home/gtangg12/auto-augment/tests/example_output_depth.npy')

    betas = gaussian_source_sink(image.shape[1:], beta=0.1, num_gaussians=128, source_sink_ratio=0.5, max_scale=0.025)
    image_haze = haze(image, depth, beta=betas)
    image_haze = image_haze.transpose(1, 2, 0)
    image_haze = Image.fromarray((image_haze * 255).astype('uint8'))
    image_haze.save('/home/gtangg12/auto-augment/tests/example_output1_haze.png')