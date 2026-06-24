import torch
import numpy as np


def gaussian_heat_source_torch(
    x,
    y,
    Q0=500.0,
    x_c=0.5,
    y_c=0.5,
    sigma=0.10
):
    """
    Gaussian heat source for PyTorch tensors.

    Q(x,y) = Q0 * exp(-((x-xc)^2 + (y-yc)^2)/(2*sigma^2))
    """

    return Q0 * torch.exp(
        -((x - x_c) ** 2 + (y - y_c) ** 2)
        / (2 * sigma ** 2)
    )


def gaussian_heat_source_numpy(
    X,
    Y,
    Q0=500.0,
    x_c=0.5,
    y_c=0.5,
    sigma=0.10
):
    """
    Gaussian heat source for NumPy arrays.
    """

    return Q0 * np.exp(
        -((X - x_c) ** 2 + (Y - y_c) ** 2)
        / (2 * sigma ** 2)
    )


def uniform_heat_source_torch(x, Q0=1.0):
    """
    Uniform heat source for PyTorch tensors.
    """

    return torch.ones_like(x) * Q0


def uniform_heat_source_numpy(X, Q0=1.0):
    """
    Uniform heat source for NumPy arrays.
    """

    return np.ones_like(X) * Q0
