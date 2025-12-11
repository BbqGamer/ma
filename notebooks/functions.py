import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    from ma_thesis.data import ackley, smooth, bukin_function_6
    from functools import partial
    return ackley, bukin_function_6, np, partial, smooth


@app.cell
def _(np):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D


    def plot_surface(func, x_range, y_range, resolution=100):
        x = np.linspace(x_range[0], x_range[1], resolution)
        y = np.linspace(y_range[0], y_range[1], resolution)
        X, Y = np.meshgrid(x, y)
        Z = func(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, cmap='viridis')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")
        plt.show()
    return plot_surface, plt


@app.cell
def _(ackley, plot_surface):
    plot_surface(ackley, x_range=(-5, 5), y_range=(-5, 5), resolution=200)
    return


@app.cell
def _(ackley, plot_surface, smooth):
    smoothed_ackley = smooth(ackley, window_size=0.2, samples=100)
    plot_surface(smoothed_ackley, (-5, 5), (-5, 5))
    return


@app.cell
def _(ackley, plot_surface, smooth):
    plot_surface(smooth(ackley, window_size=0.5, samples=100), (-5, 5), (-5, 5))
    return


@app.cell
def _(bukin_function_6, plot_surface):
    plot_surface(bukin_function_6, (-15, -5), (-3, 3))
    return


@app.cell
def _(np, plt):
    def sample_function(func, x_range, y_range, num_samples=1000):
        x_samples = np.random.uniform(x_range[0], x_range[1], num_samples)
        y_samples = np.random.uniform(y_range[0], y_range[1], num_samples)
        X = np.c_[x_samples, y_samples]
        y = func(X)
        return X, y

    def scatter3D(X, y):
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap='viridis', marker='.')
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")
        plt.show()
    return sample_function, scatter3D


@app.cell
def _(ackley, sample_function, scatter3D):
    X, y = sample_function(ackley, (-5, 5), (-5, 5), num_samples=10000)
    scatter3D(X, y)
    return X, y


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from flax import linen as nn
    import optax
    return jax, jnp, nn, optax


@app.cell
def _(jax, jnp, nn, optax, partial):
    class MLP(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(64)(x)
            x = nn.tanh(x)
            x = nn.Dense(64)(x)
            x = nn.tanh(x)
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



    @partial(jax.jit, static_argnames=['model', 'optimizer'])
    def train_step(params, opt_state, x_batch, y_batch, model, optimizer):
        """
        Performs a single training step.
        Returns: new_params, new_opt_state, loss_value
        """
        loss_val, grads = jax.value_and_grad(mse_loss)(params, model, x_batch, y_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_val
    return MLP, train_step


@app.cell
def _(MLP, X, jax, jnp, optax, train_step, y):
    key = jax.random.PRNGKey(42)

    X_train = jnp.array(X)
    y_train = jnp.array(y)

    model = MLP()
    key, init_key = jax.random.split(key)
    params = model.init(init_key, jnp.ones((1, 2))) # Dummy input to infer shapes
    

    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    
    epochs = 10
    batch_size = 64
    num_samples = X_train.shape[0]
    steps_per_epoch = num_samples // batch_size

    print(f"Starting training on {num_samples} samples...")
    
    loss_history = []

    for epoch in range(epochs):
        # Shuffle data at the start of epoch
        key, shuffle_key = jax.random.split(key)
        perms = jax.random.permutation(shuffle_key, num_samples)
        x_perm = X_train[perms]
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
    
        print(f"Epoch {epoch} | MSE Loss: {avg_loss:.5f}")
    return model, params, y_train


@app.cell
def _(model, np, params, y_train):
    # Create a grid for inference
    _x = np.linspace(-5, 5, 200)
    _y = np.linspace(-5, 5, 200)
    _Xg, _Yg = np.meshgrid(_x, _y)
    _Xin = np.c_[_Xg.ravel(), _Yg.ravel()]

    _pred = np.array(model.apply(params, np.array(_Xin))).reshape(_Xg.shape)
    y_train
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
