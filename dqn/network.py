"""Q-networks for low-dimensional DQN agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class DQNMLP(nn.Module):
    """MLP Q-network for vector observations."""

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer size.")

        self.input_dim = int(input_dim)
        self.num_actions = int(num_actions)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)

        layers: list[nn.Module] = []
        last_dim = self.input_dim
        for hidden_dim in self.hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU())
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, self.num_actions))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return self.model(x)

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        return self.forward(state)


def build_q_network(
    network_type: str,
    input_dim: int,
    num_actions: int,
    hidden_sizes: tuple[int, ...],
) -> nn.Module:
    if network_type != "mlp":
        raise ValueError(f"Unsupported network_type: {network_type}")
    return DQNMLP(input_dim=input_dim, num_actions=num_actions, hidden_sizes=hidden_sizes)


if __name__ == "__main__":
    model = DQNMLP(input_dim=4, num_actions=2)
    dummy_input = torch.randn(32, 4)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
