import torch


def second_derivative_1d(output, x):
    """
    Compute d²(output)/dx² using automatic differentiation.
    """

    first_derivative = torch.autograd.grad(
        output,
        x,
        grad_outputs=torch.ones_like(output),
        create_graph=True
    )[0]

    second_derivative = torch.autograd.grad(
        first_derivative,
        x,
        grad_outputs=torch.ones_like(first_derivative),
        create_graph=True
    )[0]

    return second_derivative


def steady_state_residual_1d(model, x):
    """
    Residual for:
        d²T/dx² = 0
    """

    x.requires_grad_(True)

    T = model(x)

    T_xx = second_derivative_1d(T, x)

    residual = T_xx

    return torch.mean(residual ** 2)


def heat_generation_residual_1d(model, x, Q=1000.0, k=10.0):
    """
    Residual for:
        d²T/dx² + Q/k = 0
    """

    x.requires_grad_(True)

    T = model(x)

    T_xx = second_derivative_1d(T, x)

    residual = T_xx + Q / k

    return torch.mean(residual ** 2)


def transient_residual_1d(model, x, t, alpha=0.1):
    """
    Residual for:
        u_t - alpha*u_xx = 0
    """

    x.requires_grad_(True)
    t.requires_grad_(True)

    u = model(x, t)

    u_t = torch.autograd.grad(
        u,
        t,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_x = torch.autograd.grad(
        u,
        x,
        grad_outputs=torch.ones_like(u),
        create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x,
        x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True
    )[0]

    residual = u_t - alpha * u_xx

    return torch.mean(residual ** 2)


def steady_state_residual_2d(model, x, y, heat_source_fn, k=10.0):
    """
    Residual for:
        T_xx + T_yy + Q(x,y)/k = 0
    """

    x.requires_grad_(True)
    y.requires_grad_(True)

    T = model(x, y)

    T_x = torch.autograd.grad(
        T,
        x,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    T_y = torch.autograd.grad(
        T,
        y,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    T_xx = torch.autograd.grad(
        T_x,
        x,
        grad_outputs=torch.ones_like(T_x),
        create_graph=True
    )[0]

    T_yy = torch.autograd.grad(
        T_y,
        y,
        grad_outputs=torch.ones_like(T_y),
        create_graph=True
    )[0]

    Q = heat_source_fn(x, y)

    residual = T_xx + T_yy + Q / k

    return torch.mean(residual ** 2)


def transient_residual_2d(model, x, y, t, heat_source_fn, alpha=0.1):
    """
    Residual for:
        T_t - alpha*(T_xx + T_yy) - S(x,y) = 0
    """

    x.requires_grad_(True)
    y.requires_grad_(True)
    t.requires_grad_(True)

    T = model(x, y, t)

    T_t = torch.autograd.grad(
        T,
        t,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    T_x = torch.autograd.grad(
        T,
        x,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    T_y = torch.autograd.grad(
        T,
        y,
        grad_outputs=torch.ones_like(T),
        create_graph=True
    )[0]

    T_xx = torch.autograd.grad(
        T_x,
        x,
        grad_outputs=torch.ones_like(T_x),
        create_graph=True
    )[0]

    T_yy = torch.autograd.grad(
        T_y,
        y,
        grad_outputs=torch.ones_like(T_y),
        create_graph=True
    )[0]

    S = heat_source_fn(x, y)

    residual = T_t - alpha * (T_xx + T_yy) - S

    return torch.mean(residual ** 2)
