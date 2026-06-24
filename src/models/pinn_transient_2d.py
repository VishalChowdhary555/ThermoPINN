import torch
import torch.nn as nn


class TransientPINN2D(nn.Module):
    """
    PINN model for 2D transient heat diffusion.

    Governing equation:
        T_t = α(T_xx + T_yy) + S(x,y)

    Initial condition:
        T(x,y,0) = T0

    Boundary conditions:
        T = T0 on all edges

    Hard constraint:
        T(x,y,t) =
        T0 + (1 - exp(-t))x(1-x)y(1-y)N(x,y,t)
    """

    def __init__(
        self,
        T0=300.0,
        hidden_dim=80,
        num_hidden_layers=6
    ):
        super().__init__()

        self.T0 = T0

        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]

        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh()
            ]

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x, y, t):
        """
        Parameters
        ----------
        x : torch.Tensor
            x-coordinate.

        y : torch.Tensor
            y-coordinate.

        t : torch.Tensor
            time coordinate.

        Returns
        -------
        torch.Tensor
            Predicted temperature field T(x,y,t).
        """

        inputs = torch.cat([x, y, t], dim=1)

        raw_output = self.network(inputs)

        boundary_factor = (
            x * (1 - x)
            * y * (1 - y)
        )

        time_factor = 1 - torch.exp(-t)

        T = (
            self.T0
            + time_factor * boundary_factor * raw_output
        )

        return T
