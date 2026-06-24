import numpy as np


def mean_absolute_error(prediction, reference):
    """
    Mean Absolute Error.
    """

    return np.mean(np.abs(prediction - reference))


def root_mean_squared_error(prediction, reference):
    """
    Root Mean Squared Error.
    """

    return np.sqrt(np.mean((prediction - reference) ** 2))


def max_absolute_error(prediction, reference):
    """
    Maximum absolute error.
    """

    return np.max(np.abs(prediction - reference))


def relative_l2_error(prediction, reference):
    """
    Relative L2 error.
    """

    return np.linalg.norm(prediction - reference) / np.linalg.norm(reference)


def compute_all_metrics(prediction, reference):
    """
    Compute common benchmark metrics.
    """

    return {
        "MAE": mean_absolute_error(prediction, reference),
        "RMSE": root_mean_squared_error(prediction, reference),
        "Max Error": max_absolute_error(prediction, reference),
        "Relative L2": relative_l2_error(prediction, reference),
    }


def temperature_summary(T):
    """
    Summary statistics for a temperature field.
    """

    return {
        "T_min": np.min(T),
        "T_max": np.max(T),
        "T_mean": np.mean(T),
        "T_std": np.std(T),
    }
