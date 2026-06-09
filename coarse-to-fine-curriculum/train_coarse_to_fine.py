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
    num_levels = len(hierarchy_levels)
    if num_levels > 0 and curriculum_epochs > 0:
        base_epochs = curriculum_epochs // num_levels
        remainder = curriculum_epochs % num_levels
        for level_idx, clusters in enumerate(hierarchy_levels):
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

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
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext
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

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device)
        test_metrics = evaluate(model, test_loader, device)
        train_loss = total_train_loss / max(total_train_samples, 1)
        record = {
            "epoch": epoch + 1,
            "stage": current_stage_name or "baseline",
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        logger.info(
            "Epoch [%d/%d] | stage=%s | train_loss=%.4f | val_acc=%.4f | test_acc=%.4f | lr=%.6f",
            epoch + 1,
            args.epochs,
            record["stage"],
            train_loss,
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
    result = {
        "mode": "baseline",
        "dataset": bundle.name,
        "model": args.model,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_val_acc": max(float(entry["val_acc"]) for entry in history),
        "best_test_acc": max(float(entry["test_acc"]) for entry in history),
        "final_test_acc": float(test_metrics["acc"]),
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
