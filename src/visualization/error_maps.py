import matplotlib.pyplot as plt


def plot_error_map(
    X,
    Y,
    error,
    title="Error Map",
    levels=50
):
    """
    Plot spatial error heatmap.
    """

    plt.figure(figsize=(7, 5))

    contour = plt.contourf(
        X,
        Y,
        error,
        levels=levels
    )

    plt.colorbar(
        contour,
        label="Error"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(title)

    plt.tight_layout()
    plt.show()


def plot_centerline_comparison(
    x,
    reference,
    prediction,
    title="Centerline Comparison"
):
    """
    Plot PINN vs reference centerline comparison.
    """

    plt.figure(figsize=(7, 5))

    plt.plot(
        x,
        reference,
        label="Reference",
        linewidth=2
    )

    plt.plot(
        x,
        prediction,
        "--",
        label="PINN",
        linewidth=2
    )

    plt.xlabel("x")
    plt.ylabel("Temperature")

    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_time_series_comparison(
    time,
    reference,
    prediction,
    title="Time Series Comparison"
):
    """
    Plot PINN vs reference time-series comparison.
    """

    plt.figure(figsize=(7, 5))

    plt.plot(
        time,
        reference,
        label="Reference",
        linewidth=2
    )

    plt.plot(
        time,
        prediction,
        "--",
        label="PINN",
        linewidth=2
    )

    plt.xlabel("Time")
    plt.ylabel("Temperature")

    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
