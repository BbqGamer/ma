from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import polars as pl
import typer

from ma_thesis.config import FIGURES_DIR, PROCESSED_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    input_path: Path = PROCESSED_DATA_DIR / "ackley.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
    # -----------------------------------------
):
    logger.info("Generating plot from data...")
    df = pl.read_csv(input_path)
    X = df[:, :2]
    ycols = df.columns[2:]
    print(ycols)

    fig = plt.figure(figsize=(30, 10))
    for i, col in enumerate(ycols):
        y = df[col]
        ax = fig.add_subplot(1, len(ycols), i + 1, projection="3d")
        ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap="viridis", marker=".")
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_zlabel("f(x)")
        ax.title(col)

    fig.savefig(output_path)

    logger.success("Plot generation complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
