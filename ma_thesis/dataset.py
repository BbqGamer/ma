from pathlib import Path

from loguru import logger
import numpy as np
import polars as pl
import typer

from ma_thesis.config import PROCESSED_DATA_DIR
from ma_thesis.data import ackley, smooth

app = typer.Typer()

XRANGE = (-5, 5)
YRANGE = (-5, 5)

CURRICULUM = {"easy": ackley, "medium": smooth(ackley, 0.2, 100), "hard": smooth(ackley, 0.5, 100)}


@app.command()
def main(
    output_path: Path = PROCESSED_DATA_DIR / "ackley.csv",
    num_samples: int = 5000,
):
    logger.info("Generating dataset...")
    x_samples = np.random.uniform(XRANGE[0], XRANGE[1], num_samples)
    y_samples = np.random.uniform(YRANGE[0], YRANGE[1], num_samples)
    X = np.c_[x_samples, y_samples]
    data = {"X": x_samples, "Y": y_samples}
    for title, func in CURRICULUM.items():
        data[title] = func(X)

    df = pl.DataFrame(data)
    df.write_csv(output_path)
    logger.success("Processing dataset complete.")
    # -----------------------------------------


if __name__ == "__main__":
    app()
