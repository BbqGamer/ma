import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Ackley function
    def ackley(x, a=20, b=0.2, c=2*np.pi):
        x = np.asarray(x)
        d = x.shape[1]
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=1) / d))
        term2 = -np.exp(np.sum(np.cos(c * x), axis=1) / d)
        return term1 + term2 + a + np.e

    # Sample points (2D so we can visualize)
    n_samples = 10000
    d = 2
    X = np.random.uniform(-5, 5, size=(n_samples, d))
    y = ackley(X)
    noise_std = 0.1
    y_noisy = y + np.random.normal(0, noise_std, size=y.shape)

    # 3D scatter plot
    fig = plt.figure(figsize=(10,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], y_noisy, s=5)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_zlabel("f(x)")
    plt.show()

    return


if __name__ == "__main__":
    app.run()
