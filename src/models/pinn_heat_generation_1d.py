import torch
import torch.nn as nn


class HeatGenerationPINN1D(nn.Module):
    """
    PINN model for 1D steady-state heat conduction
    with internal heat generation.

    Governing equation:
        d²T/dx² + Q/k = 0

    Boundary conditions:
        T(0) = T_left
        T(1) = T_right

    Hard boundary enforcement:
        T(x) = T_left(1-x) + T_right*x + x(1-x)N(x)
    """

    def __init__(
        self,
        T_left=400.0,
        T_right=300.0,
        hidden_dim=64,
        num_hidden_layers=4
    ):
        super().__init__()

        self.T_left = T_left
        self.T_right = T_right

        layers = [nn.Linear(1, hidden_dim), nn.Tanh()]

        for _ in range(num_hidden_layers):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]

        layers.append(nn.Linear(hidden_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.network(x)

        boundary_solution = self.T_left * (1 - x) + self.T_right * x
        correction = x * (1 - x) * raw

        return boundary_solution + correction
