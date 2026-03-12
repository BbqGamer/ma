"""Neural network architectures for function approximation.

This module contains all model definitions used in the thesis experiments:
- MLP: Residual MLP with configurable activation
- SIREN: Sinusoidal Representation Networks for high-frequency functions
- FourierFeatureMLP: MLP with random Fourier feature encoding

All models expect 2D input (x1, x2) and produce scalar output.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def init_weights_lecun(m: nn.Module) -> None:
    """
    Matches Flax's default 'lecun_normal' initialization.
    Flax Dense uses truncated normal with stddev = 1/sqrt(fan_in).
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


ACTIVATIONS: dict[str, type[nn.Module]] = {
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "relu": nn.ReLU,
}


class ResBlock(nn.Module):
    """Residual block with two linear layers and configurable activation."""

    def __init__(self, dim: int, activation: str = "silu") -> None:
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            act_cls(),
            nn.Linear(dim, dim),
        )
        self.act = act_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class MLP(nn.Module):
    """Residual MLP with configurable depth and activation.

    Architecture:
        input (2D) -> Linear + Act -> [ResBlock x num_blocks] -> Linear -> output (1D)

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for all layers
    num_blocks : int
        Number of residual blocks
    activation : str
        Activation function name (must be in ACTIVATIONS dict)
    """

    def __init__(self, hidden_dim: int = 256, num_blocks: int = 4, activation: str = "silu") -> None:
        super().__init__()
        act_cls = ACTIVATIONS[activation]
        self.input_proj = nn.Sequential(
            nn.Linear(2, hidden_dim),
            act_cls(),
        )
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, activation) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.apply(init_weights_lecun)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output_proj(x)


class SirenLayer(nn.Module):
    """Single sine layer with SIREN-specific weight initialisation."""

    def __init__(
        self, in_dim: int, out_dim: int, omega_0: float = 30.0, is_first: bool = False
    ) -> None:
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_dim, out_dim)
        with torch.no_grad():
            if is_first:
                bound = 1.0 / in_dim
            else:
                bound = np.sqrt(6.0 / in_dim) / omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """Sinusoidal Representation Network.

    Particularly effective for functions with high-frequency components
    (e.g. Eggholder, which contains sin(sqrt(...)) terms).

    Reference: Sitzmann et al., "Implicit Neural Representations with
    Periodic Activation Functions", NeurIPS 2020.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for all layers
    num_layers : int
        Number of sine layers
    omega_0 : float
        Frequency multiplier for sine activations
    """

    def __init__(
        self, hidden_dim: int = 256, num_layers: int = 4, omega_0: float = 30.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [SirenLayer(2, hidden_dim, omega_0=omega_0, is_first=True)]
        for _ in range(num_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega_0=omega_0))
        self.net = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output(self.net(x))


class FourierFeatureMLP(nn.Module):
    """MLP preceded by random Fourier feature encoding.

    Maps inputs through [sin(2π B x), cos(2π B x)] before the MLP,
    giving the network a head-start at representing high-frequency patterns.
    B is sampled once at init and kept fixed (not trained).

    Reference: Tancik et al., "Fourier Features Let Networks Learn High
    Frequency Functions in Low Dimensional Domains", NeurIPS 2020.

    Parameters
    ----------
    hidden_dim : int
        Hidden dimension for MLP layers
    num_blocks : int
        Number of residual blocks
    activation : str
        Activation function name
    num_fourier : int
        Number of random Fourier features (B is 2 × num_fourier)
    sigma : float
        Standard deviation for random Fourier feature matrix B
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        activation: str = "silu",
        num_fourier: int = 128,
        sigma: float = 10.0,
    ) -> None:
        super().__init__()
        B = torch.randn(2, num_fourier) * sigma
        self.register_buffer("B", B)
        input_dim = 2 * num_fourier  # sin + cos channels
        act_cls = ACTIVATIONS[activation]
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), act_cls())
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim, activation) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.apply(init_weights_lecun)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * np.pi * (x @ self.B)
        x_enc = torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
        x_enc = self.input_proj(x_enc)
        x_enc = self.blocks(x_enc)
        return self.output_proj(x_enc)


def build_model(hp: dict[str, Any], device: torch.device) -> nn.Module:
    """Construct a model from a hyperparameter dict (as produced by the sweep).

    Parameters
    ----------
    hp : dict
        Hyperparameter dictionary containing:
        - model_arch: "mlp", "siren", or "fourier"
        - hidden_dim: hidden layer dimension
        - For MLP: num_blocks, activation
        - For SIREN: num_layers, omega_0
        - For Fourier: num_blocks, activation, num_fourier, sigma
    device : torch.device
        Device to place the model on

    Returns
    -------
    nn.Module
        Constructed model ready for training
    """
    arch = hp.get("model_arch", "mlp")
    if arch == "siren":
        model = SIREN(
            hidden_dim=hp["hidden_dim"],
            num_layers=hp.get("num_layers", 4),
            omega_0=hp.get("omega_0", 30.0),
        )
    elif arch == "fourier":
        model = FourierFeatureMLP(
            hidden_dim=hp["hidden_dim"],
            num_blocks=hp.get("num_blocks", 4),
            activation=hp.get("activation", "silu"),
            num_fourier=hp.get("num_fourier", 128),
            sigma=hp.get("sigma", 10.0),
        )
    else:
        model = MLP(
            hidden_dim=hp["hidden_dim"],
            num_blocks=hp.get("num_blocks", 4),
            activation=hp.get("activation", "silu"),
        )
    return model.to(device)
