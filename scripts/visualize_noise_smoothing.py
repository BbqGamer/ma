#!/usr/bin/env python3
"""Visualize how label noise interacts with Gaussian continuation smoothing.

For each benchmark function, this script generates one figure with:
- rows = noise levels
- columns = clean surface + smoothed targets for each sigma level

The underlying targets are produced with the same dataset generation logic used in
experiments, i.e. Gaussian label noise is added first and smoothing is applied after.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import typer

from ma_thesis.config import REPORTS_DIR
from ma_thesis.dataset import FUNCTIONS, generate_and_smooth

app = typer.Typer(add_completion=False)


ROOT_OUT = REPORTS_DIR / "analysis" / "noise_smoothing_viz"


def _parse_csv_str(values: str) -> list[str]:
    return [v.strip() for v in values.split(",") if v.strip()]


def _parse_csv_float(values: str) -> list[float]:
    return [float(v.strip()) for v in values.split(",") if v.strip()]


def _surface_from_samples(
    X: np.ndarray,
    y: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    grid_res: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xg = np.linspace(x_range[0], x_range[1], grid_res)
    yg = np.linspace(y_range[0], y_range[1], grid_res)
    Xg, Yg = np.meshgrid(xg, yg)
    Zg = griddata(X, y, (Xg, Yg), method="cubic")
    if np.isnan(Zg).any():
        Zg_linear = griddata(X, y, (Xg, Yg), method="linear")
        Zg = np.where(np.isnan(Zg), Zg_linear, Zg)
    if np.isnan(Zg).any():
        Zg_nearest = griddata(X, y, (Xg, Yg), method="nearest")
        Zg = np.where(np.isnan(Zg), Zg_nearest, Zg)
    return Xg, Yg, Zg


@app.command()
def main(
    functions: str = typer.Option(
        "levy,ackley,eggholder",
        help="Comma-separated function list.",
    ),
    noise_ratios: str = typer.Option(
        "0.0,0.01,0.02,0.05,0.1",
        help="Comma-separated noise ratios to visualize.",
    ),
    num_samples: int = typer.Option(
        5000,
        help="Number of random samples used to build each visualization dataset.",
    ),
    num_sigmas: int = typer.Option(
        5,
        help="Number of smoothing levels to visualize.",
    ),
    sigma_scale: float = typer.Option(5.0, help="Maximum sigma scale."),
    seed: int = typer.Option(42, help="Random seed."),
    grid_res: int = typer.Option(120, help="Surface rendering resolution."),
    output_dir: Path = typer.Option(ROOT_OUT, help="Output directory."),
) -> None:
    function_values = _parse_csv_str(functions)
    noise_values = _parse_csv_float(noise_ratios)
    if not function_values:
        raise typer.BadParameter("At least one function is required.")
    if not noise_values:
        raise typer.BadParameter("At least one noise ratio is required.")

    output_dir.mkdir(parents=True, exist_ok=True)

    for function_name in function_values:
        if function_name not in FUNCTIONS:
            raise typer.BadParameter(
                f"Unknown function '{function_name}'. Allowed: {', '.join(sorted(FUNCTIONS))}."
            )

        func, x_range, y_range = FUNCTIONS[function_name]
        fig, axes = plt.subplots(
            len(noise_values),
            num_sigmas + 1,
            figsize=(3.8 * (num_sigmas + 1), 3.3 * len(noise_values)),
            squeeze=False,
            constrained_layout=True,
        )

        # Clean reference surface on a regular grid.
        xg = np.linspace(x_range[0], x_range[1], grid_res)
        yg = np.linspace(y_range[0], y_range[1], grid_res)
        Xg_clean, Yg_clean = np.meshgrid(xg, yg)
        grid_points = np.column_stack([Xg_clean.ravel(), Yg_clean.ravel()])
        Z_clean = func(grid_points).reshape(grid_res, grid_res)

        all_surfaces: list[np.ndarray] = [Z_clean]
        row_payloads: list[tuple[float, np.ndarray, list[float], list[np.ndarray], float]] = []

        for noise_ratio in noise_values:
            X, _Y_raw, sigmas, datasets, noise_sigma = generate_and_smooth(
                func=func,
                x_range=x_range,
                y_range=y_range,
                N=num_samples,
                K=num_sigmas,
                sigma_scale=sigma_scale,
                seed=seed,
                noise_ratio=noise_ratio,
            )
            surfaces: list[np.ndarray] = []
            sigma_values = [float(s) for s in sigmas.tolist()]
            for _, y_sigma in datasets:
                _Xg, _Yg, Zg = _surface_from_samples(X, y_sigma, x_range, y_range, grid_res)
                surfaces.append(Zg)
                all_surfaces.append(Zg)
            row_payloads.append((noise_ratio, X, sigma_values, surfaces, noise_sigma))

        finite_values = np.concatenate([z[np.isfinite(z)] for z in all_surfaces if np.isfinite(z).any()])
        vmin = float(np.quantile(finite_values, 0.02))
        vmax = float(np.quantile(finite_values, 0.98))

        for row_idx, (noise_ratio, _X, sigma_values, surfaces, noise_sigma) in enumerate(row_payloads):
            ax0 = axes[row_idx, 0]
            im = ax0.imshow(
                Z_clean,
                extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            if row_idx == 0:
                ax0.set_title("clean function")
            ax0.set_ylabel(f"noise={noise_ratio:g}\nσ_noise={noise_sigma:.3g}")
            ax0.set_xlabel("x1")

            for col_idx, (sigma_value, Zg) in enumerate(zip(sigma_values, surfaces, strict=False), start=1):
                ax = axes[row_idx, col_idx]
                ax.imshow(
                    Zg,
                    extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                    origin="lower",
                    aspect="auto",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                if row_idx == 0:
                    ax.set_title(f"y_sigma_{col_idx - 1}\nσ={sigma_value:.3g}")
                ax.set_xlabel("x1")

        for ax in axes.ravel():
            ax.set_yticks([])
            ax.set_xticks([])

        fig.suptitle(
            f"{function_name}: noise added before smoothing, then Gaussian continuation targets",
            fontsize=14,
        )
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.92, pad=0.01)
        cbar.set_label("target value")
        out_path = output_dir / f"{function_name}_noise_smoothing_grid.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    app()
