import torch
import torch.nn as nn


class TransientPINN1D(nn.Module):
    """
    PINN model for the 1D transient heat equation.

    Governing equation:
        ∂u/∂t = α ∂²u/∂x²

    Boundary conditions:
        u(0,t) = 0
        u(1,t) = 0

    Hard boundary enforcement:
        u(x,t) = x(1-x)N(x,t)
    """

    def __init__(
        self,
        hidden_dim=64,
        num_hidden_layers=5
    ):
        super().__init__()

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]

        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ]

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x, t):
        """
        Parameters
        ----------
        x : torch.Tensor
            Spatial coordinate.

        t : torch.Tensor
            Time coordinate.

        Returns
        -------
        torch.Tensor
            Predicted temperature field u(x,t).
        """

        inputs = torch.cat([x, t], dim=1)

        raw_output = self.network(inputs)

        # Hard boundary condition enforcement
        u = x * (1 - x) * raw_output

        return u
