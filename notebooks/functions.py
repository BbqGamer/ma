import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np

    def ackley(x, a=20, b=0.2, c=2*np.pi):
        """https://www.sfu.ca/~ssurjano/ackley.html"""
        x = np.asarray(x)
        d = x.shape[1]
        term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2, axis=1) / d))
        term2 = -np.exp(np.sum(np.cos(c * x), axis=1) / d)
        return term1 + term2 + a + np.e

    return ackley, np


@app.cell
def _(np):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def visualize_function(func, x_range, y_range, resolution=100):
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
    return (visualize_function,)


@app.cell
def _(ackley, visualize_function):
    visualize_function(ackley, (-5, 5), (-5, 5))
    return


@app.cell
def _(np):
    def bukin_function_6(x):
        """https://www.sfu.ca/~ssurjano/bukin6.html"""
        x = np.asarray(x)
        x1 = x[:, 0]
        x2 = x[:, 1]
    
        term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
        term2 = 0.01 * np.abs(x1 + 10)
    
        return term1 + term2
    return (bukin_function_6,)


@app.cell
def _(bukin_function_6, visualize_function):
    visualize_function(bukin_function_6, (-15, -5), (-3, 3))
    return


if __name__ == "__main__":
    app.run()
