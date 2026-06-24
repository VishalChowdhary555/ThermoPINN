import torch
import torch.nn as nn


class SteadyStatePINN2D(nn.Module):
    """
    PINN model for 2D steady-state heat conduction.

    Governing equation:
        T_xx + T_yy + Q(x,y)/k = 0

    Boundary conditions:
        T = T_boundary on all edges

    Hard boundary enforcement:
        T(x,y) =
        T_boundary
        +
        x(1-x)y(1-y)N(x,y)
    """

    def __init__(
        self,
        T_boundary=300.0,
        hidden_dim=80,
        num_hidden_layers=6
    ):
        super().__init__()

        self.T_boundary = T_boundary

        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]

        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ]

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        Parameters
        ----------
        x : torch.Tensor
            x-coordinate.

        y : torch.Tensor
            y-coordinate.

        Returns
        -------
        torch.Tensor
            Predicted temperature field T(x,y).
        """

        inputs = torch.cat([x, y], dim=1)

        raw_output = self.network(inputs)

        boundary_factor = (
            x * (1 - x)
            * y * (1 - y)
        )

        T = (
            self.T_boundary
            + boundary_factor * raw_output
        )

        return T
