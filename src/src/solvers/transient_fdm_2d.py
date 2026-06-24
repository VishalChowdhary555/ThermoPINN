import numpy as np


def solve_transient_2d(
    source_fn,
    nx=80,
    ny=80,
    alpha=0.1,
    T0=300.0,
    t_final=1.0,
    safety=0.8,
    snapshot_times=None
):
    """
    Stable explicit finite difference solver for:

        T_t = alpha * (T_xx + T_yy) + S(x,y)

    Boundary condition:
        T = T0 on all boundaries

    Initial condition:
        T(x,y,0) = T0

    Parameters
    ----------
    source_fn : callable
        Function returning S(X,Y).

    nx, ny : int
        Number of grid points.

    alpha : float
        Thermal diffusivity.

    T0 : float
        Initial and boundary temperature.

    t_final : float
        Final simulation time.

    safety : float
        Safety factor for explicit time-step stability.

    snapshot_times : list or None
        Times at which temperature snapshots are stored.

    Returns
    -------
    X, Y : ndarray
        Spatial meshgrid.

    snapshots : dict
        Dictionary of temperature fields at selected times.

    time_history : ndarray
        Time values for center temperature history.

    center_history : ndarray
        Center temperature history.
    """

    if snapshot_times is None:
        snapshot_times = [0.0, 0.25, 0.5, 0.75, 1.0]

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)

    dx = x[1] - x[0]
    dy = y[1] - y[0]

    stability_limit = 1.0 / (
        2.0 * alpha * (1.0 / dx**2 + 1.0 / dy**2)
    )

    dt = safety * stability_limit
    nt = int(np.ceil(t_final / dt))
    dt = t_final / nt

    print("Stable FDM settings:")
    print(f"dx = {dx:.4e}, dy = {dy:.4e}")
    print(f"dt = {dt:.4e}, nt = {nt}")
    print(f"stability limit = {stability_limit:.4e}")

    X, Y = np.meshgrid(x, y)

    S = source_fn(X, Y)

    T = np.ones((ny, nx)) * T0

    snapshots = {0.0: T.copy()}

    center_y = ny // 2
    center_x = nx // 2

    time_history = [0.0]
    center_history = [T[center_y, center_x]]

    for step in range(1, nt + 1):

        T_new = T.copy()

        T_new[1:-1, 1:-1] = (
            T[1:-1, 1:-1]
            + alpha * dt * (
                (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dx**2
                + (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dy**2
            )
            + dt * S[1:-1, 1:-1]
        )

        T_new[0, :] = T0
        T_new[-1, :] = T0
        T_new[:, 0] = T0
        T_new[:, -1] = T0

        if np.isnan(T_new).any() or np.isinf(T_new).any():
            raise RuntimeError("FDM became unstable: NaN or Inf detected.")

        T = T_new

        current_time = step * dt

        time_history.append(current_time)
        center_history.append(T[center_y, center_x])

        for snapshot_time in snapshot_times:
            if snapshot_time not in snapshots and current_time >= snapshot_time:
                snapshots[snapshot_time] = T.copy()

    snapshots[t_final] = T.copy()

    return (
        X,
        Y,
        snapshots,
        np.array(time_history),
        np.array(center_history)
    )
