import torch


def boundary_error_1d(model, T_left=400.0, T_right=300.0, device="cpu"):
    """
    Monitor boundary error for 1D steady-state models.

    Checks:
        T(0) = T_left
        T(1) = T_right
    """

    x_left = torch.tensor([[0.0]], device=device)
    x_right = torch.tensor([[1.0]], device=device)

    with torch.no_grad():
        T_left_pred = model(x_left)
        T_right_pred = model(x_right)

    left_error = torch.abs(T_left_pred - T_left).item()
    right_error = torch.abs(T_right_pred - T_right).item()

    return left_error, right_error


def boundary_error_2d(model, T_boundary=300.0, n_points=100, device="cpu"):
    """
    Monitor boundary error for 2D steady-state models.

    Checks all four boundaries:
        x = 0
        x = 1
        y = 0
        y = 1
    """

    vals = torch.linspace(0, 1, n_points).view(-1, 1).to(device)

    zeros = torch.zeros_like(vals)
    ones = torch.ones_like(vals)

    with torch.no_grad():
        T_left = model(zeros, vals)
        T_right = model(ones, vals)
        T_bottom = model(vals, zeros)
        T_top = model(vals, ones)

    error = (
        torch.mean(torch.abs(T_left - T_boundary))
        + torch.mean(torch.abs(T_right - T_boundary))
        + torch.mean(torch.abs(T_bottom - T_boundary))
        + torch.mean(torch.abs(T_top - T_boundary))
    ) / 4

    return error.item()


def initial_condition_error_1d(
    model,
    initial_condition_fn,
    n_points=100,
    device="cpu"
):
    """
    Monitor initial-condition error for 1D transient models.

    Checks:
        u(x,0) = initial_condition_fn(x)
    """

    x = torch.linspace(0, 1, n_points).view(-1, 1).to(device)
    t = torch.zeros_like(x).to(device)

    with torch.no_grad():
        u_pred = model(x, t)

    u_true = initial_condition_fn(x)

    error = torch.mean(torch.abs(u_pred - u_true))

    return error.item()


def boundary_error_transient_1d(model, n_points=100, device="cpu"):
    """
    Monitor boundary error for 1D transient models.

    Checks:
        u(0,t) = 0
        u(1,t) = 0
    """

    t = torch.linspace(0, 1, n_points).view(-1, 1).to(device)

    x_left = torch.zeros_like(t)
    x_right = torch.ones_like(t)

    with torch.no_grad():
        u_left = model(x_left, t)
        u_right = model(x_right, t)

    error = (
        torch.mean(torch.abs(u_left))
        + torch.mean(torch.abs(u_right))
    ) / 2

    return error.item()


def constraint_error_transient_2d(
    model,
    T0=300.0,
    n_points=80,
    device="cpu"
):
    """
    Monitor hard-constrained IC and BC errors for 2D transient models.

    Checks:
        T = T0 on all boundaries
        T(x,y,0) = T0
    """

    vals = torch.linspace(0, 1, n_points).view(-1, 1).to(device)

    zeros = torch.zeros_like(vals)
    ones = torch.ones_like(vals)

    with torch.no_grad():
        t_vals = vals

        T_left = model(zeros, vals, t_vals)
        T_right = model(ones, vals, t_vals)
        T_bottom = model(vals, zeros, t_vals)
        T_top = model(vals, ones, t_vals)

        T_init = model(vals, vals, zeros)

    bc_error = (
        torch.mean(torch.abs(T_left - T0))
        + torch.mean(torch.abs(T_right - T0))
        + torch.mean(torch.abs(T_bottom - T0))
        + torch.mean(torch.abs(T_top - T0))
    ) / 4

    ic_error = torch.mean(torch.abs(T_init - T0))

    return bc_error.item(), ic_error.item()
