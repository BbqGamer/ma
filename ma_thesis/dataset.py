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


def generate_and_smooth(
    func,
    x_range,
    y_range,
    N=2000,
    K=6,
    sigma_scale=5,
    seed=42,
    noise_ratio=0.02,
):
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
    Y_raw_clean = func(X)
    y_std = float(np.std(Y_raw_clean))
    noise_sigma = max(y_std, 1e-8) * noise_ratio
    noise = np.random.normal(loc=0.0, scale=noise_sigma, size=Y_raw_clean.shape)
    Y_raw = Y_raw_clean + noise

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

    return X, Y_raw, sigmas, datasets, noise_sigma


def _derive_split_filenames(func_name: str, output_name: str | None) -> tuple[str, str]:
    """Return (train_filename, test_filename) using *_train/*_test suffixes."""
    base = output_name if output_name else f"{func_name}.parquet"
    if not base.endswith(".parquet"):
        base = f"{base}.parquet"
    stem = base.removesuffix(".parquet")
    if stem.endswith("_train"):
        stem = stem.removesuffix("_train")
    if stem.endswith("_test"):
        stem = stem.removesuffix("_test")
    return f"{stem}_train.parquet", f"{stem}_test.parquet"


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
    num_samples: int = 20000,
    num_sigmas: int = 3,
    sigma_scale: float = 5.0,
    train_samples: int = 10000,
    noise_ratio: float = 0.02,
    seed: int = 42,
    output_name: str | None = typer.Option(
        None,
        help="Optional output base filename for single function generation.",
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
            f"Generating {func_name} dataset with {num_samples} samples and "
            f"{num_sigmas} smoothing levels..."
        )
        if train_samples <= 0 or train_samples >= num_samples:
            logger.error(
                f"Invalid split: train_samples={train_samples}, num_samples={num_samples}. "
                "Need 0 < train_samples < num_samples."
            )
            raise typer.Exit(code=1)
        test_samples = num_samples - train_samples

        X, Y_raw, sigmas, datasets, noise_sigma = generate_and_smooth(
            func=func,
            x_range=x_range,
            y_range=y_range,
            N=num_samples,
            K=num_sigmas,
            sigma_scale=sigma_scale,
            seed=seed,
            noise_ratio=noise_ratio,
        )
        logger.info(f"  Added Gaussian label noise with σ={noise_sigma:.6f} (ratio={noise_ratio})")

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
        rng = np.random.default_rng(seed)
        perm = rng.permutation(num_samples)
        train_idx = perm[:train_samples]
        test_idx = perm[train_samples:]

        df_train = df[train_idx.tolist()]
        df_test = df[test_idx.tolist()]
        train_filename, test_filename = _derive_split_filenames(func_name, output_name)
        train_path = output_dir / train_filename
        test_path = output_dir / test_filename
        df_train.write_parquet(train_path)
        df_test.write_parquet(test_path)
        logger.success(
            f"Datasets saved to {train_path} (train={train_samples}) "
            f"and {test_path} (test={test_samples})"
        )


if __name__ == "__main__":
    app()
