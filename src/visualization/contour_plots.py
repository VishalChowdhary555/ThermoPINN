import matplotlib.pyplot as plt


def contour_temperature(
    X,
    Y,
    T,
    title="Temperature Field",
    levels=50
):
    """
    Plot temperature contour map.
    """

    plt.figure(figsize=(7, 5))

    contour = plt.contourf(
        X,
        Y,
        T,
        levels=levels
    )

    plt.colorbar(
        contour,
        label="Temperature"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(title)

    plt.tight_layout()
    plt.show()


def contour_heat_source(
    X,
    Y,
    Q,
    title="Heat Source",
    levels=50
):
    """
    Plot heat source distribution.
    """

    plt.figure(figsize=(7, 5))

    contour = plt.contourf(
        X,
        Y,
        Q,
        levels=levels
    )

    plt.colorbar(
        contour,
        label="Heat Generation"
    )

    plt.xlabel("x")
    plt.ylabel("y")

    plt.title(title)

    plt.tight_layout()
    plt.show()
