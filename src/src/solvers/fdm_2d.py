import numpy as np


def solve_steady_state_2d(
    heat_source_fn,
    nx=100,
    ny=100,
    k=10.0,
    T_boundary=300.0,
    max_iter=10000,
    tolerance=1e-7
):
    """
    Finite Difference Method solver for:

        T_xx + T_yy + Q(x,y)/k = 0

    Parameters
    ----------
    heat_source_fn : callable
        Function returning Q(X,Y).

    nx : int
        Number of x grid points.

    ny : int
        Number of y grid points.

    k : float
        Thermal conductivity.

    T_boundary : float
        Fixed boundary temperature.

    max_iter : int
        Maximum iterations.

    tolerance : float
        Convergence tolerance.

    Returns
    -------
    X : ndarray
        x-grid mesh.

    Y : ndarray
        y-grid mesh.

    T : ndarray
        Temperature field.
    """

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y)

    Q = heat_source_fn(X, Y)

    T = np.ones((ny, nx)) * T_boundary

    for iteration in range(max_iter):

        T_old = T.copy()

        T[1:-1, 1:-1] = (
            (T[1:-1, 2:] + T[1:-1, :-2]) * dy**2
            + (T[2:, 1:-1] + T[:-2, 1:-1]) * dx**2
            + (Q[1:-1, 1:-1] / k) * dx**2 * dy**2
        ) / (2.0 * (dx**2 + dy**2))

        # Dirichlet boundaries
        T[0, :] = T_boundary
        T[-1, :] = T_boundary
        T[:, 0] = T_boundary
        T[:, -1] = T_boundary

        max_change = np.max(np.abs(T - T_old))

        if max_change < tolerance:
            print(
                f"FDM converged in "
                f"{iteration} iterations"
            )
            break

    return X, Y, T


def compute_temperature_metrics(T):
    """
    Basic statistics for a temperature field.
    """

    return {
        "T_min": np.min(T),
        "T_max": np.max(T),
        "T_mean": np.mean(T),
        "T_std": np.std(T)
    }
