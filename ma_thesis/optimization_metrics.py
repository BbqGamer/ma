"""Optimization and landscape diagnostics for training loops."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

EPS = 1e-12


@dataclass
class StepMetricsState:
    """State for cheap step-wise gradient diagnostics."""

    beta: float = 0.98
    prev_grad: torch.Tensor | None = None
    ema_grad_mean: float = 0.0
    ema_grad_sq_mean: float = 0.0
    initialized: bool = False


def _flatten_grads(model: nn.Module) -> torch.Tensor | None:
    grads = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    if not grads:
        return None
    return torch.cat(grads)


def compute_step_metrics(model: nn.Module, state: StepMetricsState) -> dict[str, float]:
    """Compute cheap step-wise metrics from current mini-batch gradients."""
    grad = _flatten_grads(model)
    if grad is None:
        return {}

    grad_mean = float(grad.mean().item())
    grad_sq_mean = float((grad * grad).mean().item())
    grad_var = max(grad_sq_mean - grad_mean * grad_mean, 0.0)

    if not state.initialized:
        state.ema_grad_mean = grad_mean
        state.ema_grad_sq_mean = grad_sq_mean
        state.initialized = True
    else:
        b = state.beta
        state.ema_grad_mean = b * state.ema_grad_mean + (1.0 - b) * grad_mean
        state.ema_grad_sq_mean = b * state.ema_grad_sq_mean + (1.0 - b) * grad_sq_mean

    ema_var = max(state.ema_grad_sq_mean - state.ema_grad_mean * state.ema_grad_mean, 0.0)
    gsnr = (state.ema_grad_mean * state.ema_grad_mean) / (ema_var + EPS)

    out = {
        "grad_variance": grad_var,
        "grad_noise_scale": ema_var,
        "gsnr": gsnr,
    }

    if state.prev_grad is not None:
        num = torch.dot(grad, state.prev_grad)
        den = grad.norm() * state.prev_grad.norm() + EPS
        out["grad_cosine_sim"] = float((num / den).item())

    state.prev_grad = grad.clone()
    return out


def _probe_loss(
    model: nn.Module,
    criterion: nn.Module,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
) -> torch.Tensor:
    pred = model(x_probe)
    return criterion(pred, y_probe)


def hutchinson_hessian_trace(
    model: nn.Module,
    criterion: nn.Module,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
) -> float:
    """Estimate Hessian trace via one Rademacher probe."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return float("nan")

    loss = _probe_loss(model, criterion, x_probe, y_probe)
    grads = torch.autograd.grad(loss, params, create_graph=True)
    vecs = [torch.randint_like(p, low=0, high=2, dtype=torch.int64).float() * 2.0 - 1.0 for p in params]
    gv = sum((g * v).sum() for g, v in zip(grads, vecs, strict=False))
    hv = torch.autograd.grad(gv, params, retain_graph=False)
    trace_est = sum((h * v).sum() for h, v in zip(hv, vecs, strict=False))
    return float(trace_est.item())


def critical_sharpness(
    model: nn.Module,
    criterion: nn.Module,
    x_probe: torch.Tensor,
    y_probe: torch.Tensor,
    lr_ref: float,
    n_binary_steps: int = 5,
) -> float:
    """Estimate critical step size where loss starts increasing along -grad direction."""
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return float("nan")

    with torch.no_grad():
        base_loss = float(_probe_loss(model, criterion, x_probe, y_probe).item())

    loss = _probe_loss(model, criterion, x_probe, y_probe)
    grads = torch.autograd.grad(loss, params, create_graph=False)
    direction = [-g.detach() for g in grads]
    norm = torch.sqrt(sum((d * d).sum() for d in direction)).item()
    if norm <= EPS:
        return 0.0

    direction = [d / (norm + EPS) for d in direction]

    def eval_eta(eta: float) -> float:
        with torch.no_grad():
            for p, d in zip(params, direction, strict=False):
                p.add_(eta * d)
            val = float(_probe_loss(model, criterion, x_probe, y_probe).item())
            for p, d in zip(params, direction, strict=False):
                p.add_(-eta * d)
        return val

    lo = 0.0
    hi = max(lr_ref, 1e-6)
    max_hi = max(1024.0 * max(lr_ref, 1e-6), 1.0)

    while hi < max_hi and eval_eta(hi) <= base_loss:
        lo = hi
        hi *= 2.0

    if hi >= max_hi and eval_eta(hi) <= base_loss:
        return hi

    for _ in range(n_binary_steps):
        mid = 0.5 * (lo + hi)
        if eval_eta(mid) > base_loss:
            hi = mid
        else:
            lo = mid
    return hi


def layerwise_spectral_alpha(model: nn.Module) -> tuple[float, dict[str, float]]:
    """Estimate power-law slope alpha from singular values of 2D weights."""
    per_layer: dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue
        w = param.detach()
        s = torch.linalg.svdvals(w).float()
        s = s[s > 0]
        if s.numel() < 4:
            continue
        s_sorted, _ = torch.sort(s, descending=True)
        idx = torch.arange(1, s_sorted.numel() + 1, device=s_sorted.device, dtype=torch.float32)
        x = torch.log(idx)
        y = torch.log(s_sorted)
        x_mean = x.mean()
        y_mean = y.mean()
        denom = ((x - x_mean) ** 2).sum()
        if float(denom.item()) <= EPS:
            continue
        slope = ((x - x_mean) * (y - y_mean)).sum() / denom
        alpha = float((-slope).item())
        per_layer[name.replace(".", "_")] = alpha

    if not per_layer:
        return float("nan"), {}
    alpha_mean = float(sum(per_layer.values()) / len(per_layer))
    return alpha_mean, per_layer

