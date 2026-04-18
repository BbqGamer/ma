#!/usr/bin/env python3
"""Visualize noise and Gaussian smoothing as 3D surfaces.

For each function and noise level, this script writes one PNG with:
- clean function surface
- one 3D surface per y_sigma_* target

This uses the same target generation logic as the training pipeline:
noise is added first, then Gaussian continuation smoothing is applied.
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

ROOT_OUT = REPORTS_DIR / "analysis" / "noise_smoothing_viz_3d"


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
        4000,
        help="Number of random samples used to build each visualization dataset.",
    ),
    num_sigmas: int = typer.Option(
        5,
        help="Number of smoothing levels to visualize.",
    ),
    sigma_scale: float = typer.Option(5.0, help="Maximum sigma scale."),
    seed: int = typer.Option(42, help="Random seed."),
    grid_res: int = typer.Option(90, help="Surface rendering resolution."),
    elev: float = typer.Option(28.0, help="3D elevation angle."),
    azim: float = typer.Option(-60.0, help="3D azimuth angle."),
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
        xg = np.linspace(x_range[0], x_range[1], grid_res)
        yg = np.linspace(y_range[0], y_range[1], grid_res)
        Xg_clean, Yg_clean = np.meshgrid(xg, yg)
        grid_points = np.column_stack([Xg_clean.ravel(), Yg_clean.ravel()])
        Z_clean = func(grid_points).reshape(grid_res, grid_res)

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

            surfaces: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = [
                ("clean function", Xg_clean, Yg_clean, Z_clean)
            ]
            for idx, ((_, y_sigma), sigma_value) in enumerate(zip(datasets, sigmas, strict=False)):
                Xg, Yg, Zg = _surface_from_samples(X, y_sigma, x_range, y_range, grid_res)
                surfaces.append((f"y_sigma_{idx}\nσ={float(sigma_value):.3g}", Xg, Yg, Zg))

            finite_values = np.concatenate([
                Z[np.isfinite(Z)] for _, _, _, Z in surfaces if np.isfinite(Z).any()
            ])
            zmin = float(np.quantile(finite_values, 0.02))
            zmax = float(np.quantile(finite_values, 0.98))

            fig = plt.figure(figsize=(22, 10), constrained_layout=True)
            for panel_idx, (title, Xg, Yg, Zg) in enumerate(surfaces, start=1):
                ax = fig.add_subplot(2, 3, panel_idx, projection="3d")
                ax.plot_surface(
                    Xg,
                    Yg,
                    Zg,
                    cmap="viridis",
                    edgecolor="none",
                    antialiased=True,
                    vmin=zmin,
                    vmax=zmax,
                )
                ax.set_title(title)
                ax.set_xlabel("x1")
                ax.set_ylabel("x2")
                ax.set_zlabel("target")
                ax.view_init(elev=elev, azim=azim)
                ax.set_zlim(zmin, zmax)

            fig.suptitle(
                f"{function_name} | noise_ratio={noise_ratio:g} | noise added before smoothing "
                f"(σ_noise={noise_sigma:.3g})",
                fontsize=16,
            )

            out_path = output_dir / f"{function_name}_noise{noise_ratio:g}_surfaces_3d.png"
            fig.savefig(out_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    app()
