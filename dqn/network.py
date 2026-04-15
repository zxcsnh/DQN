"""Convolutional Q-network used by DQN."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNCNN(nn.Module):
    """Nature DQN convolutional backbone for Atari observations."""

    def __init__(self, input_channels: int = 4, num_actions: int = 6):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # For 84x84 input: 84 -> 20 -> 9 -> 7
        self.conv_output_size = 64 * 7 * 7

        self.fc1 = nn.Linear(self.conv_output_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float() / 255.0

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def get_q_values(self, state: torch.Tensor) -> torch.Tensor:
        """Alias kept for readability at call sites."""
        return self.forward(state)


if __name__ == "__main__":
    model = DQNCNN(input_channels=4, num_actions=6)
    print(f"Network architecture:\n{model}")

    dummy_input = torch.randint(0, 255, (32, 4, 84, 84), dtype=torch.float32)
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
