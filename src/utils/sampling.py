import torch


def sample_1d(n_points, device="cpu"):
    """
    Uniform sampling in [0,1].
    """

    return torch.rand((n_points, 1), device=device)


def sample_2d(n_points, device="cpu"):
    """
    Uniform sampling in [0,1]x[0,1].
    """

    x = torch.rand((n_points, 1), device=device)
    y = torch.rand((n_points, 1), device=device)

    return x, y


def sample_space_time_1d(n_points, device="cpu"):
    """
    Sample (x,t).
    """

    x = torch.rand((n_points, 1), device=device)
    t = torch.rand((n_points, 1), device=device)

    return x, t


def sample_space_time_2d(n_points, device="cpu"):
    """
    Sample (x,y,t).
    """

    x = torch.rand((n_points, 1), device=device)
    y = torch.rand((n_points, 1), device=device)
    t = torch.rand((n_points, 1), device=device)

    return x, y, t


def sample_initial_condition_1d(n_points, device="cpu"):
    """
    Sample x with t=0.
    """

    x = torch.rand((n_points, 1), device=device)
    t = torch.zeros_like(x)

    return x, t


def sample_boundary_1d(n_points, device="cpu"):
    """
    Sample transient 1D boundaries.
    """

    t = torch.rand((n_points, 1), device=device)

    x_left = torch.zeros_like(t)
    x_right = torch.ones_like(t)

    return x_left, x_right, t
