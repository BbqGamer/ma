from functools import partial

from flax import linen as nn
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

from ma_thesis.data import ackley


def generate_data(n_samples=2000, key=None):
    if key is None:
        key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    x_train = jax.random.uniform(subkey, (n_samples, 2), minval=-5.0, maxval=5.0)
    y_train = ackley(x_train).reshape(-1, 1)

    return x_train, y_train


class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(64)(x)
        x = nn.tanh(x)
        x = nn.Dense(1)(x)
        return x


def mse_loss(params, model, x_batch, y_batch):
    preds = model.apply(params, x_batch)
    return jnp.mean((preds - y_batch) ** 2)


@partial(jax.jit, static_argnames=["model", "optimizer"])
def train_step(params, opt_state, x_batch, y_batch, model, optimizer):
    """
    Performs a single training step.
    Returns: new_params, new_opt_state, loss_value
    """
    loss_val, grads = jax.value_and_grad(mse_loss)(params, model, x_batch, y_batch)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss_val


def main():
    key = jax.random.PRNGKey(42)
    x_train, y_train = generate_data(n_samples=5000, key=key)

    model = MLP()
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.ones((1, 2)))

    learning_rate = 1e-3
    optimizer = optax.adamw(learning_rate)
    opt_state = optimizer.init(params)

    epochs = 1000
    batch_size = 64
    num_samples = x_train.shape[0]
    steps_per_epoch = num_samples // batch_size

    print(f"Starting training on {num_samples} samples...")

    loss_history = []

    for epoch in range(epochs):
        key, shuffle_key = jax.random.split(key)
        perms = jax.random.permutation(shuffle_key, num_samples)
        x_perm = x_train[perms]
        y_perm = y_train[perms]

        epoch_loss = 0.0

        for i in range(steps_per_epoch):
            start = i * batch_size
            end = start + batch_size
            x_batch = x_perm[start:end]
            y_batch = y_perm[start:end]

            params, opt_state, loss_val = train_step(
                params, opt_state, x_batch, y_batch, model, optimizer
            )
            epoch_loss += loss_val

        avg_loss = epoch_loss / steps_per_epoch
        loss_history.append(avg_loss)

        if epoch % 25 == 0:
            print(f"Epoch {epoch} | MSE Loss: {avg_loss:.5f}")

    print("Training complete.")

    fig = plt.figure(figsize=(12, 5))

    grid_x = jnp.linspace(-5, 5, 100)
    grid_y = jnp.linspace(-5, 5, 100)
    XX, YY = jnp.meshgrid(grid_x, grid_y)
    grid_points = jnp.stack([XX.ravel(), YY.ravel()], axis=-1)

    true_values = ackley(grid_points).reshape(100, 100)
    pred_values = model.apply(params, grid_points).reshape(100, 100)

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(np.array(XX), np.array(YY), np.array(true_values), cmap="viridis", alpha=0.8)
    ax1.set_title("True Ackley Function")

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    ax2.plot_surface(np.array(XX), np.array(YY), np.array(pred_values), cmap="plasma", alpha=0.8)
    ax2.set_title("Neural Network Approximation")

    plt.tight_layout()
    plt.savefig("reports/figures/ackley_learned.png")


if __name__ == "__main__":
    main()
