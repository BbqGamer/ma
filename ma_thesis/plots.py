from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.interpolate import griddata
import typer

from ma_thesis.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_dir: Path = PROCESSED_DATA_DIR.parent / "processed",
    output_dir: Path = FIGURES_DIR,
    grid_res: int = 80,
    # -----------------------------------------
):
    logger.info("Generating plots from parquet files...")

    # Get all parquet files in the input directory
    parquet_files = list(input_dir.glob("*.parquet"))

    if not parquet_files:
        logger.error(f"No parquet files found in {input_dir}")
        return

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    for parquet_file in parquet_files:
        func_name = parquet_file.stem
        logger.info(f"Processing {func_name}...")

        # Read the parquet file
        df = pl.read_parquet(parquet_file)

        # Extract X coordinates
        X = df.select(["x1", "x2"]).to_numpy()

        # Get all y columns (y_sigma_0, y_sigma_1, etc.)
        y_cols = [col for col in df.columns if col.startswith("y_sigma_")]
        n_sigmas = len(y_cols)

        # Determine x and y ranges for gridding
        x_range = (X[:, 0].min(), X[:, 0].max())
        y_range = (X[:, 1].min(), X[:, 1].max())

        # Create figure with surface and scatter plots for each sigma
        fig = plt.figure(figsize=(12, 4 * n_sigmas))
        fig.suptitle(func_name, fontsize=16)

        for i, y_col in enumerate(y_cols):
            Y = df[y_col].to_numpy()

            # Surface plot
            ax1 = fig.add_subplot(n_sigmas, 2, 2 * i + 1, projection="3d")
            xg = np.linspace(x_range[0], x_range[1], grid_res)
            yg = np.linspace(y_range[0], y_range[1], grid_res)
            Xg, Yg = np.meshgrid(xg, yg)
            Zg = griddata(X, Y, (Xg, Yg), method="cubic", fill_value=np.nan)
            ax1.plot_surface(Xg, Yg, Zg, cmap="viridis", edgecolor="none", alpha=0.95)
            ax1.set_title(f"{y_col}\nSurface plot")
            ax1.set_xlabel("x1")
            ax1.set_ylabel("x2")
            ax1.set_zlabel("f(x)")

            # Scatter plot
            ax2 = fig.add_subplot(n_sigmas, 2, 2 * i + 2, projection="3d")
            ax2.scatter(X[:, 0], X[:, 1], Y, c=Y, cmap="viridis", s=10)
            ax2.set_title(f"{y_col}\nScatter plot")
            ax2.set_xlabel("x1")
            ax2.set_ylabel("x2")
            ax2.set_zlabel("f(x)")

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save the figure
        output_path = output_dir / f"{func_name}_gaussian_continuation.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.success(f"Saved plot to {output_path}")

    logger.success("All plots generated successfully.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
