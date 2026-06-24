import matplotlib.pyplot as plt


def plot_training_loss(
    loss_history,
    title="Training Loss"
):
    """
    Plot training loss.
    """

    plt.figure(figsize=(7, 5))

    plt.semilogy(loss_history, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)

    plt.grid(True)
    plt.tight_layout()

    plt.show()


def plot_multiple_losses(
    total_loss,
    pde_loss,
    constraint_loss,
    labels=("Total", "PDE", "Constraint")
):
    """
    Plot multiple training losses.
    """

    plt.figure(figsize=(7, 5))

    plt.semilogy(total_loss, label=labels[0])
    plt.semilogy(pde_loss, label=labels[1])
    plt.semilogy(constraint_loss, label=labels[2])

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
