from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader

from ctf.data import DatasetBundle, load_dataset, seed_worker
from ctf.hierarchy import compute_hierarchy, singleton_clusters
from ctf.models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch reproduction of Coarse-to-Fine Curriculum Learning"
    )
    parser.add_argument("--mode", choices=["baseline", "curriculum"], required=True)
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "cifar100", "shapes", "tiny-imagenet"],
        default="cifar100",
    )
    parser.add_argument("--model", choices=["cnn", "resnet18", "resnet50"], default="cnn")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--curriculum_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--shapes_test_ratio", type=float, default=0.2)
    parser.add_argument(
        "--distance_source",
        choices=["classifier_weights", "confusion"],
        default="classifier_weights",
    )
    parser.add_argument(
        "--curriculum_target_fraction",
        type=float,
        default=0.9,
        help="Auto curriculum length target: first baseline epoch reaching this fraction of best val acc.",
    )
    parser.add_argument("--data_dir", type=str, default="/workspace/data")
    parser.add_argument("--output_dir", type=str, default="/workspace/runs")
    parser.add_argument("--run_id", type=str, default="run")
    parser.add_argument("--reference_run_dir", type=str, default=None)
    parser.add_argument("--shapes_path", type=str, default=None)
    parser.add_argument("--tiny_imagenet_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--augmentation", action="store_true", default=None)
    parser.add_argument("--no-augmentation", action="store_false", dest="augmentation")
    return parser.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.epochs is None:
        args.epochs = 200 if args.model != "cnn" else 400
    if args.batch_size is None:
        args.batch_size = 128 if args.model != "cnn" else 512
    if args.lr is None:
        args.lr = 0.1 if args.model != "cnn" else 1e-3
    if args.weight_decay is None:
        args.weight_decay = 5e-4 if args.model != "cnn" else 0.0
    if args.augmentation is None:
        args.augmentation = args.model != "cnn" and args.dataset != "shapes"
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(output_dir: Path, filename: str) -> logging.Logger:
    logger = logging.getLogger(str(output_dir))
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(output_dir / filename)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def save_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    save_rows_csv(path, [row])


def build_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(bundle.train_dataset, shuffle=True, **loader_kwargs)
    eval_loader_kwargs = {k: v for k, v in loader_kwargs.items() if k != "generator"}
    val_loader = DataLoader(bundle.val_dataset, shuffle=False, **eval_loader_kwargs)
    test_loader = DataLoader(bundle.test_dataset, shuffle=False, **eval_loader_kwargs)
    return train_loader, val_loader, test_loader


def create_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
) -> tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler | None]:
    if args.model == "cnn":
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        return optimizer, None

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[round(args.epochs * 0.37), round(args.epochs * 0.75)],
        gamma=0.1,
    )
    return optimizer, scheduler


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
    }


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    preds_all: list[np.ndarray] = []
    true_probs_all: list[np.ndarray] = []
    margins_all: list[np.ndarray] = []

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        sorted_probs, _ = probs.sort(dim=1, descending=True)
        margins = sorted_probs[:, 0] - sorted_probs[:, 1]
        true_probs = probs.gather(1, labels.to(device, non_blocking=True).unsqueeze(1)).squeeze(1)

        probabilities.append(probs.cpu().numpy())
        labels_all.append(labels.numpy())
        preds_all.append(preds.cpu().numpy())
        true_probs_all.append(true_probs.cpu().numpy())
        margins_all.append(margins.cpu().numpy())

    return {
        "probabilities": np.concatenate(probabilities, axis=0),
        "labels": np.concatenate(labels_all, axis=0),
        "predictions": np.concatenate(preds_all, axis=0),
        "true_probabilities": np.concatenate(true_probs_all, axis=0),
        "top1_margins": np.concatenate(margins_all, axis=0),
    }


@torch.inference_mode()
def collect_probabilities(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(probs, axis=0), np.concatenate(labels_all, axis=0)


@torch.inference_mode()
def classifier_weight_distance(model: nn.Module) -> np.ndarray:
    weights = model.classifier_weight.detach().cpu()
    weights = F.normalize(weights, p=2, dim=1)
    sim = weights @ weights.T
    dist = 1.0 - sim.numpy()
    np.fill_diagonal(dist, 0.0)
    return dist


@torch.inference_mode()
def confusion_distance(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    probs, labels = collect_probabilities(model, loader, device)
    predictions = np.argmax(probs, axis=1)
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for label in range(num_classes):
        indices = np.where(labels == label)[0]
        if indices.size == 0:
            continue
        predicted = predictions[indices]
        counts = np.bincount(predicted, minlength=num_classes)
        confusion[label] = counts / max(indices.size, 1)
    return 1.0 - confusion


def clusters_to_membership(
    clusters: list[list[int]],
    num_classes: int,
    device: torch.device,
) -> torch.Tensor:
    membership = torch.zeros((num_classes, num_classes), dtype=torch.bool, device=device)
    for cluster in clusters:
        cluster_tensor = torch.tensor(cluster, dtype=torch.long, device=device)
        cluster_mask = torch.zeros((num_classes,), dtype=torch.bool, device=device)
        cluster_mask[cluster_tensor] = True
        membership[cluster_tensor] = cluster_mask
    return membership


def marginalized_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    membership: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    mask = membership[targets]
    selected = log_probs.masked_fill(~mask, float("-inf"))
    return -torch.logsumexp(selected, dim=1).mean()


def build_curriculum_schedule(
    num_classes: int,
    hierarchy_levels: list[list[list[int]]],
    curriculum_epochs: int,
    total_epochs: int,
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    effective_levels = [clusters for clusters in hierarchy_levels if len(clusters) < num_classes]
    num_levels = len(effective_levels)
    if num_levels > 0 and curriculum_epochs > 0:
        base_epochs = curriculum_epochs // num_levels
        remainder = curriculum_epochs % num_levels
        for level_idx, clusters in enumerate(effective_levels):
            epochs_this_level = base_epochs + (1 if level_idx < remainder else 0)
            if epochs_this_level <= 0:
                continue
            schedule.append(
                {
                    "name": f"level_{level_idx + 1}",
                    "clusters": clusters,
                    "epochs": epochs_this_level,
                }
            )

    remaining = max(total_epochs - curriculum_epochs, 0)
    if remaining > 0 or not schedule:
        schedule.append(
            {
                "name": "fine_tune",
                "clusters": singleton_clusters(num_classes),
                "epochs": remaining if remaining > 0 else total_epochs,
            }
        )
    return schedule


def epoch_to_stage(schedule: list[dict[str, Any]], epoch: int) -> dict[str, Any]:
    cursor = 0
    for stage in schedule:
        next_cursor = cursor + int(stage["epochs"])
        if epoch < next_cursor:
            return stage
        cursor = next_cursor
    return schedule[-1]


def compute_confusion_counts(
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for label, pred in zip(labels, predictions, strict=False):
        confusion[int(label), int(pred)] += 1
    return confusion


def normalize_confusion(confusion: np.ndarray) -> np.ndarray:
    row_sums = confusion.sum(axis=1, keepdims=True)
    return confusion / np.clip(row_sums, 1, None)


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or np.allclose(values.sum(), 0.0):
        return 0.0
    values = np.sort(values)
    n = values.size
    cumulative = np.cumsum(values)
    return float((n + 1 - 2 * (cumulative.sum() / cumulative[-1])) / n)


def one_way_anova(values: np.ndarray, group_ids: list[int]) -> dict[str, float]:
    groups: dict[int, list[float]] = {}
    for value, group_id in zip(values.tolist(), group_ids, strict=False):
        groups.setdefault(int(group_id), []).append(float(value))

    non_empty = [np.asarray(group, dtype=np.float64) for group in groups.values() if group]
    if len(non_empty) < 2:
        return {"f": 0.0, "eta2": 0.0}

    grand = np.concatenate(non_empty)
    grand_mean = grand.mean()
    ss_between = sum(group.size * float((group.mean() - grand_mean) ** 2) for group in non_empty)
    ss_within = sum(float(((group - group.mean()) ** 2).sum()) for group in non_empty)
    df_between = len(non_empty) - 1
    df_within = grand.size - len(non_empty)
    if df_between <= 0 or df_within <= 0 or ss_within <= 0:
        return {"f": 0.0, "eta2": 0.0}
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    f_stat = ms_between / ms_within if ms_within > 0 else 0.0
    eta2 = ss_between / (ss_between + ss_within) if (ss_between + ss_within) > 0 else 0.0
    return {"f": float(f_stat), "eta2": float(eta2)}


def build_class_metrics(
    confusion: np.ndarray,
    class_names: list[str] | None,
    class_group_ids: list[int] | None,
    class_group_names: list[str] | None,
) -> list[dict[str, Any]]:
    row_sums = confusion.sum(axis=1)
    class_acc = np.divide(
        np.diag(confusion),
        np.clip(row_sums, 1, None),
        dtype=np.float64,
    )
    rows: list[dict[str, Any]] = []
    for class_idx in range(confusion.shape[0]):
        row = confusion[class_idx].copy()
        row[class_idx] = 0
        hardest_confuser = int(np.argmax(row)) if row.sum() > 0 else class_idx
        hardest_confuser_rate = float(row[hardest_confuser] / max(row_sums[class_idx], 1))
        rows.append(
            {
                "class_idx": class_idx,
                "class_name": class_names[class_idx] if class_names is not None else str(class_idx),
                "group_idx": class_group_ids[class_idx] if class_group_ids is not None else None,
                "group_name": class_group_names[class_group_ids[class_idx]]
                if class_group_ids is not None and class_group_names is not None
                else None,
                "support": int(row_sums[class_idx]),
                "accuracy": float(class_acc[class_idx]),
                "error_rate": float(1.0 - class_acc[class_idx]),
                "hardest_confuser_idx": hardest_confuser,
                "hardest_confuser_name": class_names[hardest_confuser]
                if class_names is not None
                else str(hardest_confuser),
                "hardest_confuser_rate": hardest_confuser_rate,
            }
        )
    return rows


def build_difficulty_metrics(
    predictions: dict[str, np.ndarray],
    confusion: np.ndarray,
    bundle: DatasetBundle,
) -> dict[str, Any]:
    labels = predictions["labels"]
    preds = predictions["predictions"]
    true_probs = predictions["true_probabilities"]
    margins = predictions["top1_margins"]
    correct = preds == labels
    row_sums = confusion.sum(axis=1)
    class_acc = np.divide(np.diag(confusion), np.clip(row_sums, 1, None), dtype=np.float64)
    confusion_norm = normalize_confusion(confusion)
    entropy_rows = []
    offdiag_mass = []
    for idx in range(confusion.shape[0]):
        row = confusion_norm[idx].copy()
        row[idx] = 0.0
        row_sum = row.sum()
        offdiag_mass.append(float(row_sum))
        if row_sum > 0:
            row = row / row_sum
            entropy = -np.sum(np.where(row > 0, row * np.log(row + 1e-12), 0.0))
            entropy_rows.append(float(entropy / np.log(max(confusion.shape[0] - 1, 2))))
        else:
            entropy_rows.append(0.0)

    metrics = {
        "overall_accuracy": float(correct.mean()),
        "class_accuracy_mean": float(class_acc.mean()),
        "class_accuracy_std": float(class_acc.std()),
        "class_accuracy_variance": float(class_acc.var()),
        "class_accuracy_min": float(class_acc.min()),
        "class_accuracy_max": float(class_acc.max()),
        "class_accuracy_gini": gini(1.0 - class_acc),
        "true_class_probability_mean": float(true_probs.mean()),
        "true_class_probability_std": float(true_probs.std()),
        "top1_margin_mean": float(margins.mean()),
        "top1_margin_std": float(margins.std()),
        "top1_margin_correct_mean": float(margins[correct].mean()) if correct.any() else 0.0,
        "top1_margin_error_mean": float(margins[~correct].mean()) if (~correct).any() else 0.0,
        "confusion_entropy_mean": float(np.mean(entropy_rows)),
        "offdiag_confusion_mass_mean": float(np.mean(offdiag_mass)),
    }

    if bundle.class_group_ids is not None:
        anova = one_way_anova(class_acc, bundle.class_group_ids)
        metrics["group_accuracy_anova_f"] = anova["f"]
        metrics["group_accuracy_anova_eta2"] = anova["eta2"]

    return metrics


def save_evaluation_artifacts(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    bundle: DatasetBundle,
    output_dir: Path,
    split_name: str,
) -> dict[str, Any]:
    predictions = collect_predictions(model, loader, device)
    confusion = compute_confusion_counts(predictions["labels"], predictions["predictions"], bundle.num_classes)
    confusion_norm = normalize_confusion(confusion)
    class_metrics = build_class_metrics(
        confusion,
        class_names=bundle.class_names,
        class_group_ids=bundle.class_group_ids,
        class_group_names=bundle.class_group_names,
    )
    difficulty_metrics = build_difficulty_metrics(predictions, confusion, bundle)

    np.savetxt(output_dir / f"confusion_{split_name}_counts.csv", confusion, delimiter=",", fmt="%d")
    np.savetxt(
        output_dir / f"confusion_{split_name}_normalized.csv",
        confusion_norm,
        delimiter=",",
        fmt="%.8f",
    )
    save_rows_csv(output_dir / f"class_metrics_{split_name}.csv", class_metrics)
    save_json(output_dir / f"difficulty_metrics_{split_name}.json", difficulty_metrics)
    save_single_row_csv(output_dir / f"difficulty_metrics_{split_name}.csv", difficulty_metrics)
    return {
        "confusion_counts_path": str(output_dir / f"confusion_{split_name}_counts.csv"),
        "confusion_normalized_path": str(output_dir / f"confusion_{split_name}_normalized.csv"),
        "class_metrics_path": str(output_dir / f"class_metrics_{split_name}.csv"),
        "difficulty_metrics_path": str(output_dir / f"difficulty_metrics_{split_name}.json"),
        **difficulty_metrics,
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    best_val_acc: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_val_acc": best_val_acc,
            "args": vars(args),
        },
        path,
    )


def load_reference_model(
    run_dir: Path,
    args: argparse.Namespace,
    bundle: DatasetBundle,
    device: torch.device,
) -> tuple[nn.Module, list[dict[str, Any]]]:
    checkpoint_path = run_dir / "best_model.pt"
    history_path = run_dir / "history.json"
    if not checkpoint_path.exists() or not history_path.exists():
        raise FileNotFoundError(f"Reference run not found at {run_dir}")
    model = build_model(
        model_name=args.model,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        dropout=args.dropout,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    history = json.loads(history_path.read_text())
    return model, history


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
    run_dir: Path,
    schedule: list[dict[str, Any]] | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_val_acc = -float("inf")
    best_epoch = -1
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"
    current_stage_name = None

    for epoch in range(args.epochs):
        stage = epoch_to_stage(schedule, epoch) if schedule is not None else None
        if stage is not None and stage["name"] != current_stage_name:
            current_stage_name = str(stage["name"])
            logger.info("Switching to curriculum stage %s", current_stage_name)
        membership = None
        if stage is not None:
            membership = clusters_to_membership(stage["clusters"], num_classes, device)

        model.train()
        total_train_loss = 0.0
        total_train_samples = 0
        total_train_correct = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext
            with autocast_ctx():
                logits = model(inputs)
                if membership is None:
                    loss = F.cross_entropy(logits, labels)
                else:
                    loss = marginalized_loss(logits, labels, membership)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total_train_loss += loss.item() * labels.size(0)
            total_train_samples += labels.size(0)
            total_train_correct += (logits.argmax(dim=1) == labels).sum().item()

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        train_loss = total_train_loss / max(total_train_samples, 1)
        record = {
            "epoch": epoch + 1,
            "stage": current_stage_name or "baseline",
            "train_loss": train_loss,
            "train_acc": total_train_correct / max(total_train_samples, 1),
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        logger.info(
            "Epoch [%d/%d] | stage=%s | train_loss=%.4f | train_acc=%.4f | val_acc=%.4f | test_acc=%.4f | lr=%.6f",
            epoch + 1,
            args.epochs,
            record["stage"],
            train_loss,
            record["train_acc"],
            val_metrics["acc"],
            test_metrics["acc"],
            record["lr"],
        )

        if val_metrics["acc"] > best_val_acc:
            best_val_acc = val_metrics["acc"]
            best_epoch = epoch
            save_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, best_val_acc, args)

        save_checkpoint(last_path, model, optimizer, scheduler, epoch + 1, best_val_acc, args)
        save_json(run_dir / "history.json", history)
        save_rows_csv(run_dir / "history.csv", history)

        if args.patience > 0 and epoch - best_epoch >= args.patience:
            logger.info("Early stopping triggered after %d epochs without val improvement", args.patience)
            break

    best_checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])
    final_test_metrics = evaluate(model, test_loader, device)
    return model, history, final_test_metrics


def infer_curriculum_epochs(history: list[dict[str, Any]], target_fraction: float) -> int:
    best_val_acc = max(float(entry["val_acc"]) for entry in history)
    threshold = best_val_acc * target_fraction
    for entry in history:
        if float(entry["val_acc"]) >= threshold:
            return int(entry["epoch"])
    return int(history[-1]["epoch"])


def baseline_run(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    run_dir: Path,
) -> dict[str, Any]:
    model = build_model(
        model_name=args.model,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        dropout=args.dropout,
    ).to(device)
    _, history, test_metrics = train_loop(
        model=model,
        train_loader=loaders[0],
        val_loader=loaders[1],
        test_loader=loaders[2],
        num_classes=bundle.num_classes,
        args=args,
        device=device,
        logger=logger,
        run_dir=run_dir,
        schedule=None,
    )
    val_artifacts = save_evaluation_artifacts(model, loaders[1], device, bundle, run_dir, "val")
    test_artifacts = save_evaluation_artifacts(model, loaders[2], device, bundle, run_dir, "test")
    result = {
        "mode": "baseline",
        "dataset": bundle.name,
        "model": args.model,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_val_acc": max(float(entry["val_acc"]) for entry in history),
        "best_test_acc": max(float(entry["test_acc"]) for entry in history),
        "final_test_acc": float(test_metrics["acc"]),
        "val_difficulty_class_accuracy_variance": float(val_artifacts["class_accuracy_variance"]),
        "test_difficulty_class_accuracy_variance": float(test_artifacts["class_accuracy_variance"]),
        "test_difficulty_confusion_entropy_mean": float(test_artifacts["confusion_entropy_mean"]),
    }
    save_json(run_dir / "results.json", result)
    save_single_row_csv(run_dir / "summary.csv", result)
    return result


def curriculum_run(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    run_dir: Path,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = loaders

    if args.reference_run_dir is not None:
        reference_dir = Path(args.reference_run_dir)
        logger.info("Loading reference baseline from %s", reference_dir)
        reference_model, reference_history = load_reference_model(reference_dir, args, bundle, device)
    else:
        reference_dir = run_dir / "reference_baseline"
        reference_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Training reference baseline for hierarchy construction")
        reference_logger = setup_logger(reference_dir, "training_log_reference.txt")
        reference_model = build_model(
            model_name=args.model,
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
            dropout=args.dropout,
        ).to(device)
        reference_model, reference_history, _ = train_loop(
            model=reference_model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            num_classes=bundle.num_classes,
            args=args,
            device=device,
            logger=reference_logger,
            run_dir=reference_dir,
            schedule=None,
        )
        save_evaluation_artifacts(reference_model, val_loader, device, bundle, reference_dir, "val")
        save_evaluation_artifacts(reference_model, test_loader, device, bundle, reference_dir, "test")

    if args.distance_source == "classifier_weights":
        dist_matrix = classifier_weight_distance(reference_model)
    else:
        dist_matrix = confusion_distance(reference_model, val_loader, bundle.num_classes, device)
    np.save(run_dir / "distance_matrix.npy", dist_matrix)

    hierarchy_levels = compute_hierarchy(dist_matrix)
    save_json(
        run_dir / "hierarchy.json",
        {
            "distance_source": args.distance_source,
            "levels": hierarchy_levels,
            "class_names": bundle.class_names,
        },
    )

    if args.curriculum_epochs is None:
        curriculum_epochs = infer_curriculum_epochs(reference_history, args.curriculum_target_fraction)
        logger.info(
            "Auto curriculum length: %d epochs (target fraction %.2f)",
            curriculum_epochs,
            args.curriculum_target_fraction,
        )
    else:
        curriculum_epochs = args.curriculum_epochs
        logger.info("Manual curriculum length: %d epochs", curriculum_epochs)

    curriculum_epochs = min(curriculum_epochs, args.epochs)
    schedule = build_curriculum_schedule(
        num_classes=bundle.num_classes,
        hierarchy_levels=hierarchy_levels,
        curriculum_epochs=curriculum_epochs,
        total_epochs=args.epochs,
    )
    save_json(run_dir / "schedule.json", schedule)

    model = build_model(
        model_name=args.model,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        dropout=args.dropout,
    ).to(device)
    _, history, test_metrics = train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=bundle.num_classes,
        args=args,
        device=device,
        logger=logger,
        run_dir=run_dir,
        schedule=schedule,
    )
    val_artifacts = save_evaluation_artifacts(model, val_loader, device, bundle, run_dir, "val")
    test_artifacts = save_evaluation_artifacts(model, test_loader, device, bundle, run_dir, "test")
    result = {
        "mode": "curriculum",
        "dataset": bundle.name,
        "model": args.model,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "curriculum_epochs": curriculum_epochs,
        "num_hierarchy_levels": len(hierarchy_levels),
        "best_val_acc": max(float(entry["val_acc"]) for entry in history),
        "best_test_acc": max(float(entry["test_acc"]) for entry in history),
        "final_test_acc": float(test_metrics["acc"]),
        "distance_source": args.distance_source,
        "reference_run_dir": str(reference_dir),
        "val_difficulty_class_accuracy_variance": float(val_artifacts["class_accuracy_variance"]),
        "test_difficulty_class_accuracy_variance": float(test_artifacts["class_accuracy_variance"]),
        "test_difficulty_confusion_entropy_mean": float(test_artifacts["confusion_entropy_mean"]),
    }
    save_json(run_dir / "results.json", result)
    save_single_row_csv(run_dir / "summary.csv", result)
    return result


def main() -> None:
    args = resolve_defaults(parse_args())
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.run_id / f"{args.dataset}_{args.model}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir, f"training_log_{args.mode}.txt")
    save_json(run_dir / "config.json", vars(args))

    bundle = load_dataset(
        dataset_name=args.dataset,
        data_dir=Path(args.data_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=args.download,
        augmentation=args.augmentation,
        shapes_path=Path(args.shapes_path) if args.shapes_path else None,
        tiny_imagenet_path=Path(args.tiny_imagenet_path) if args.tiny_imagenet_path else None,
        shapes_test_ratio=args.shapes_test_ratio,
    )
    loaders = build_loaders(bundle, args.batch_size, args.num_workers, args.seed, device)

    logger.info(
        "Initialized run | mode=%s | dataset=%s | model=%s | device=%s | run_dir=%s",
        args.mode,
        bundle.name,
        args.model,
        device,
        run_dir,
    )
    logger.info(
        "num_classes=%d | batch_size=%d | epochs=%d | lr=%.6f | augmentation=%s",
        bundle.num_classes,
        args.batch_size,
        args.epochs,
        args.lr,
        args.augmentation,
    )

    if args.mode == "baseline":
        result = baseline_run(args, bundle, loaders, device, logger, run_dir)
    else:
        result = curriculum_run(args, bundle, loaders, device, logger, run_dir)

    logger.info("Run complete: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
