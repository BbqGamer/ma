from pathlib import Path

from loguru import logger
import numpy as np
import polars as pl
from sklearn.neighbors import NearestNeighbors
import typer

from ma_thesis.config import PROCESSED_DATA_DIR
from ma_thesis.data import (
    ackley,
    bukin_function_6,
    eggholder,
    franke,
    friedman1_2d,
    friedman2_2d,
    levy,
    peaks,
)

app = typer.Typer()

# Function configurations: (function, x_range, y_range)
FUNCTIONS = {
    "ackley": (ackley, (-5, 5), (-5, 5)),
    "levy": (levy, (-10, 10), (-10, 10)),
    "eggholder": (eggholder, (-512, 512), (-512, 512)),
    "bukin": (bukin_function_6, (-15, -5), (-3, 3)),
    "franke": (franke, (0, 1), (0, 1)),
    "peaks": (peaks, (-3, 3), (-3, 3)),
    "friedman1_2d": (friedman1_2d, (0, 1), (0, 1)),
    "friedman2_2d": (friedman2_2d, (0, 1), (0, 1)),
}


def generate_and_smooth(func, x_range, y_range, N=2000, K=6, sigma_scale=5, seed=42):
    """
    Generate dataset and apply Gaussian continuation smoothing.

    Args:
        func: Function to evaluate (e.g., ackley)
        x_range: Tuple of (min, max) for x dimension
        y_range: Tuple of (min, max) for y dimension
        N: Number of samples
        K: Number of smoothing levels (sigmas)
        sigma_scale: Scale factor for maximum sigma
        seed: Random seed

    Returns:
        X: Input points (N, 2)
        Y_raw: Raw function values
        sigmas: Array of sigma values used
        datasets: List of (X, Y_sigma) tuples for each sigma
    """
    np.random.seed(seed)
    X = np.random.uniform(x_range[0], x_range[1], size=(N, 2))
    Y_raw = func(X)

    # Compute pairwise distance matrix
    neigh = NearestNeighbors(n_neighbors=N, algorithm="auto")
    neigh.fit(X)
    D_dist = neigh.kneighbors_graph(X, mode="distance").toarray()

    # Determine sigma sequence based on nearest neighbor distances
    nearest_distances, _ = neigh.kneighbors(X, n_neighbors=2)
    mean_nn_dist = np.mean(nearest_distances[:, 1])
    sigma_max = sigma_scale * mean_nn_dist
    sigmas = np.linspace(sigma_max, 0, K)

    # Generate smoothed datasets for each sigma
    datasets = []
    for sigma in sigmas:
        if sigma > 0:
            W = np.exp(-(D_dist**2) / (2 * sigma**2))
        else:
            W = np.eye(N)
        W_norm = W / W.sum(axis=1, keepdims=True)
        Y_sigma = W_norm @ Y_raw
        datasets.append((X.copy(), Y_sigma.copy()))

    return X, Y_raw, sigmas, datasets


@app.command()
def main(
    function: str = typer.Option(
        "ackley",
        help=(
            "Function to use: ackley, levy, eggholder, bukin, franke, peaks, "
            "friedman1_2d, friedman2_2d, or 'all' for all."
        ),
    ),
    output_dir: Path = PROCESSED_DATA_DIR,
    num_samples: int = 4000,
    num_sigmas: int = 3,
    sigma_scale: float = 5.0,
    seed: int = 42,
    output_name: str | None = typer.Option(
        None,
        help="Optional output filename for single function generation (e.g. ackley_n4000_k3.parquet).",
    ),
):
    """
    Generate function dataset(s) with Gaussian continuation smoothing.

    Creates a dataset with multiple smoothing levels (curriculum) using
    Gaussian kernel smoothing based on pairwise distances.
    """
    # Determine which functions to process
    if function.lower() == "all":
        functions_to_process = FUNCTIONS.items()
    elif function.lower() in FUNCTIONS:
        functions_to_process = [(function.lower(), FUNCTIONS[function.lower()])]
    else:
        logger.error(
            f"Unknown function '{function}'. Available: {', '.join(FUNCTIONS.keys())}, 'all'"
        )
        raise typer.Exit(code=1)

    if function.lower() == "all" and output_name:
        logger.error("--output-name can only be used with a single function, not with 'all'.")
        raise typer.Exit(code=1)

    # Process each function
    for func_name, (func, x_range, y_range) in functions_to_process:
        logger.info(
            f"Generating {func_name} dataset with {num_samples} samples and {num_sigmas} smoothing levels..."
        )

        X, Y_raw, sigmas, datasets = generate_and_smooth(
            func=func,
            x_range=x_range,
            y_range=y_range,
            N=num_samples,
            K=num_sigmas,
            sigma_scale=sigma_scale,
            seed=seed,
        )

        # Build dataframe with x1, x2, and y_sigma_i columns
        data = {
            "x1": X[:, 0],
            "x2": X[:, 1],
        }

        for i, sigma in enumerate(sigmas):
            _, Y_sigma = datasets[i]
            data[f"y_sigma_{i}"] = Y_sigma
            logger.info(f"  Sigma {i}: {sigma:.4f}")

        df = pl.DataFrame(data)
        filename = output_name if output_name else f"{func_name}.parquet"
        output_path = output_dir / filename
        df.write_parquet(output_path)
        logger.success(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    app()
