import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
import typer

from ma_thesis.config import PROCESSED_DATA_DIR

# Create necessary directories
os.makedirs("reports/figures", exist_ok=True)

app = typer.Typer()


def ackley(x: torch.Tensor) -> torch.Tensor:
    """PyTorch implementation of the Ackley function."""
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = x.shape[1]

    sum_sq = torch.sum(x**2, dim=1, keepdim=True)
    sum_cos = torch.sum(torch.cos(c * x), dim=1, keepdim=True)

    term1 = -a * torch.exp(-b * torch.sqrt(sum_sq / d))
    term2 = -torch.exp(sum_cos / d)

    return term1 + term2 + a + np.e


def generate_data(n_samples=2000, seed=0, device="cpu"):
    """Generates training data directly on the device."""
    # Use a specific generator for reproducibility
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    # Generate on device to save transfer time
    x_train = (10.0 * torch.rand(n_samples, 2, device=device, generator=g)) - 5.0
    y_train = ackley(x_train)

    return x_train, y_train


def init_weights_lecun(m):
    """
    Matches Flax's default 'lecun_normal' initialization.
    Flax Dense uses truncated normal with stddev = 1/sqrt(fan_in).
    """
    if isinstance(m, nn.Linear):
        # LeCun Normal: mean=0, std=sqrt(1/fan_in)
        # PyTorch equivalent is roughly kaiming_normal_ with nonlinearity='linear'
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        # Apply the Flax-like initialization
        self.apply(init_weights_lecun)

    def forward(self, x):
        return self.net(x)


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "ackley.csv",
    patience: int = 10,
    min_delta: float = 1e-4,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    df = pl.read_csv(input_path)
    X = torch.from_numpy(df[:, :2].to_numpy()).float().to(device)

    # Prepare hard targets for evaluation
    if "hard" in df.columns:
        y_hard_all = torch.from_numpy(df["hard"].to_numpy()).float().unsqueeze(1).to(device)
    else:
        print("Warning: 'hard' column not found. Hard validation loss will be skipped.")
        y_hard_all = None

    # Split into train and validation (80/20) - consistent across levels
    num_samples = X.shape[0]
    n_val = int(0.2 * num_samples)
    n_train = num_samples - n_val

    # Shuffle indices once for splitting
    perm = torch.randperm(num_samples, device=device)
    train_indices = perm[:n_train]
    val_indices = perm[n_train:]

    X_train = X[train_indices]
    X_val = X[val_indices]

    if y_hard_all is not None:
        y_hard_val = y_hard_all[val_indices]

    levels = df.columns[2:]
    for level in levels:
        print("Training level:", level)
        y = torch.from_numpy(df[level].to_numpy()).float().unsqueeze(1).to(device)
        y_train = y[train_indices]
        y_val = y[val_indices]

        model = MLP().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        epochs = 1000
        batch_size = 64
        steps_per_epoch = n_train // batch_size

        print(f"Starting training on {n_train} samples (validation: {n_val})...")

        train_loss_history = []
        val_loss_history = []
        hard_val_loss_history = []

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            perm = torch.randperm(n_train, device=device)
            x_perm = X_train[perm]
            y_perm = y_train[perm]

            epoch_loss = 0.0
            model.train()

            for i in range(steps_per_epoch):
                start = i * batch_size
                end = start + batch_size

                x_batch = x_perm[start:end]
                y_batch = y_perm[start:end]

                optimizer.zero_grad(set_to_none=True)  # Slightly faster than zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / steps_per_epoch
            train_loss_history.append(avg_loss)

            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val).item()
                val_loss_history.append(val_loss)

                hard_msg = ""
                if y_hard_all is not None:
                    hard_val_loss = criterion(val_outputs, y_hard_val).item()
                    hard_val_loss_history.append(hard_val_loss)
                    hard_msg = f" | Hard Val Loss: {hard_val_loss:.5f}"

            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model state (copy to CPU to avoid keeping graph)
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            print(
                f"Epoch {epoch} | Train Loss: {avg_loss:.5f} | Val Loss: {val_loss:.5f} {hard_msg}"
            )

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print("Training complete.")

        if best_model_state is not None:
            print(f"Restoring best model with val loss: {best_val_loss:.5f}")
            model.load_state_dict(best_model_state)

        model.eval()

        # Plotting Learning Curves
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss_history[1:], label="Train Loss")
        plt.plot(val_loss_history[1:], label="Val Loss")
        if y_hard_all is not None and level != "hard":
            plt.plot(hard_val_loss_history[1:], label="Hard Val Loss", linestyle="--")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(f"Learning Curves - {level}")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"reports/figures/learning_curve_{level}.png")
        plt.close()

        # Plotting Surface
        fig = plt.figure(figsize=(12, 5))

        grid_x = torch.linspace(-5, 5, 100)
        grid_y = torch.linspace(-5, 5, 100)
        XX, YY = torch.meshgrid(grid_x, grid_y, indexing="xy")
        grid_points = torch.stack([XX.ravel(), YY.ravel()], dim=-1).to(device)

        with torch.no_grad():
            pred_values = model(grid_points).cpu().numpy().reshape(100, 100)
            true_values = ackley(grid_points).cpu().numpy().reshape(100, 100)

        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(XX.numpy(), YY.numpy(), true_values, cmap="viridis", alpha=0.8)
        ax1.set_title("True Ackley Function")

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        ax2.plot_surface(XX.numpy(), YY.numpy(), pred_values, cmap="plasma", alpha=0.8)
        ax2.set_title("Neural Network Approximation")

        plt.tight_layout()
        plt.savefig(f"reports/figures/ackley_learned_{level}.png")


if __name__ == "__main__":
    app()
