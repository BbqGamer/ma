from __future__ import annotations

import argparse
from contextlib import nullcontext
import csv
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from ctf.data import DatasetBundle, load_dataset, seed_worker
from ctf.hierarchy import compute_hierarchy, singleton_clusters
from ctf.models import build_model
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PyTorch reproduction of Coarse-to-Fine Curriculum Learning"
    )
    parser.add_argument("--mode", choices=["baseline", "curriculum", "multiloss"], required=True)
    parser.add_argument(
        "--dataset",
        choices=[
            "cifar10",
            "cifar100",
            "mnist",
            "fashion-mnist",
            "kmnist",
            "svhn",
            "stl10",
            "shapes",
            "tiny-imagenet",
        ],
        default="cifar100",
    )
    parser.add_argument(
        "--model",
        choices=[
            "cnn",
            "cifar_resnet8",
            "cifar_resnet14",
            "cifar_resnet20",
            "cifar_resnet32",
            "cifar_resnet44",
            "cifar_resnet56",
            "resnet18",
            "resnet50",
        ],
        default="cnn",
    )
    parser.add_argument("--cnn-width-multiplier", type=float, default=1.0)
    parser.add_argument("--cifar-resnet-width-multiplier", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--curriculum_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default=None)
    parser.add_argument("--scheduler", choices=["none", "step", "cosine"], default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--shapes_test_ratio", type=float, default=0.2)
    parser.add_argument(
        "--distance_source",
        choices=["classifier_weights", "confusion", "random_permutation", "teacher_embeddings"],
        default="classifier_weights",
    )
    parser.add_argument(
        "--random-hierarchy-seed",
        type=int,
        default=None,
        help="Seed for random_permutation hierarchy; defaults to the training seed.",
    )
    parser.add_argument(
        "--curriculum_target_fraction",
        type=float,
        default=0.9,
        help="Auto curriculum length target: first baseline epoch reaching this fraction of best val acc.",
    )
    parser.add_argument(
        "--curriculum-policy",
        choices=["fixed", "adaptive_plateau"],
        default="fixed",
        help="Use fixed stage lengths or advance curriculum stages on validation plateau.",
    )
    parser.add_argument(
        "--curriculum-order",
        choices=["easy_to_hard", "hard_to_easy"],
        default="easy_to_hard",
        help="Order hierarchy stages from coarse-to-fine (default) or reverse them for an anti-curriculum control.",
    )
    parser.add_argument(
        "--curriculum-min-clusters",
        type=int,
        default=0,
        help="Skip hierarchy levels with fewer clusters; 0 keeps all non-singleton levels.",
    )
    parser.add_argument(
        "--curriculum-max-levels",
        type=int,
        default=0,
        help="Maximum hierarchy levels to use; 0 keeps all levels after filtering.",
    )
    parser.add_argument("--curriculum-stage-min-epochs", type=int, default=10)
    parser.add_argument("--curriculum-stage-max-epochs", type=int, default=50)
    parser.add_argument("--curriculum-stage-patience", type=int, default=5)
    parser.add_argument("--curriculum-stage-min-delta", type=float, default=0.002)
    parser.add_argument("--data_dir", type=str, default="/workspace/data")
    parser.add_argument("--output_dir", type=str, default="/workspace/runs")
    parser.add_argument("--run_id", type=str, default="run")
    parser.add_argument("--save-checkpoints", action="store_true")
    parser.add_argument("--no-save-checkpoints", action="store_false", dest="save_checkpoints")
    parser.add_argument("--wandb", action="store_true", help="Log metrics and artifacts to Weights & Biases.")
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument(
        "--multi-weighting",
        choices=["uncertainty", "gradnorm", "static"],
        default="uncertainty",
        help="Independent per-loss weighting strategy for --mode multiloss.",
    )
    parser.add_argument("--gradnorm-alpha", type=float, default=0.5)
    parser.add_argument(
        "--roughness-probes",
        action="store_true",
        help="Log optimization roughness probes: sharpness, Hessian estimates, gradient noise/skew.",
    )
    parser.add_argument(
        "--roughness-epochs",
        type=str,
        default="1,5,10,11,20,50,100",
        help="Comma-separated 1-indexed epochs for roughness probes.",
    )
    parser.add_argument("--roughness-batches", type=int, default=2)
    parser.add_argument("--sharpness-rho", type=float, default=0.05)
    parser.add_argument("--hessian-iters", type=int, default=10)
    parser.add_argument("--hessian-samples", type=int, default=2)
    parser.add_argument(
        "--multi-static-weights",
        type=str,
        default="1,1,1,1",
        help="Comma-separated fine,coarse1,coarse2,coarse3 weights for static multiloss.",
    )
    parser.add_argument(
        "--multi-initial-weights",
        type=str,
        default="1,1,1,1",
        help="Comma-separated initial fine,coarse1,coarse2,coarse3 weights for adaptive multiloss.",
    )
    parser.add_argument("--reference_run_dir", type=str, default=None)
    parser.add_argument(
        "--teacher_run_dir",
        type=str,
        default=None,
        help="Optional external teacher run directory with best_model.pt and config.json for teacher_embeddings.",
    )
    parser.add_argument(
        "--teacher_checkpoint_path",
        type=str,
        default=None,
        help="Optional external teacher checkpoint path for teacher_embeddings.",
    )
    parser.add_argument(
        "--teacher_model",
        choices=[
            "cnn",
            "cifar_resnet8",
            "cifar_resnet14",
            "cifar_resnet20",
            "cifar_resnet32",
            "cifar_resnet44",
            "cifar_resnet56",
            "resnet18",
            "resnet50",
        ],
        default=None,
        help="Teacher model architecture when loading a teacher checkpoint directly.",
    )
    parser.add_argument("--teacher_cnn_width_multiplier", type=float, default=None)
    parser.add_argument("--teacher_cifar_resnet_width_multiplier", type=float, default=None)
    parser.add_argument(
        "--teacher_embedding_split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split used to compute teacher class prototypes for teacher_embeddings.",
    )
    parser.add_argument(
        "--teacher_pretrained_source",
        choices=["none", "torchvision_imagenet"],
        default="none",
        help="Optional built-in teacher source when no external checkpoint is available.",
    )
    parser.add_argument("--shapes_path", type=str, default=None)
    parser.add_argument("--tiny_imagenet_path", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Enable deterministic PyTorch/CUDA settings for reproducible runs.",
    )
    parser.add_argument(
        "--no-deterministic",
        action="store_false",
        dest="deterministic",
        help="Allow nondeterministic/cuDNN benchmark kernels for speed.",
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Initialize ResNet backbones from torchvision ImageNet weights.",
    )
    parser.add_argument(
        "--export-teacher-hierarchy",
        action="store_true",
        help="After a baseline run, export a hierarchy built from the trained model's embeddings.",
    )
    parser.add_argument(
        "--export-teacher-hierarchy-split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split used for automatic teacher hierarchy export.",
    )
    parser.add_argument(
        "--export-teacher-hierarchy-dir",
        type=str,
        default=None,
        help="Optional output directory for automatic teacher hierarchy export.",
    )
    parser.add_argument("--augmentation", action="store_true", default=None)
    parser.add_argument("--no-augmentation", action="store_false", dest="augmentation")
    parser.set_defaults(save_checkpoints=False, wandb=False)
    return parser.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.epochs is None:
        args.epochs = 200 if args.model != "cnn" else 400
    if args.batch_size is None:
        args.batch_size = 128 if args.model != "cnn" else 512
    if args.optimizer is None:
        args.optimizer = "sgd" if args.model != "cnn" else "adam"
    if args.scheduler is None:
        if args.optimizer == "sgd" and args.model != "cnn":
            args.scheduler = "step"
        else:
            args.scheduler = "none"
    if args.lr is None:
        if args.optimizer == "sgd":
            args.lr = 0.1
        else:
            args.lr = 1e-3
    if args.weight_decay is None:
        if args.optimizer == "sgd":
            args.weight_decay = 5e-4 if args.model != "cnn" else 0.0
        else:
            args.weight_decay = 0.0
    if args.augmentation is None:
        no_default_aug = {"shapes", "mnist", "fashion-mnist", "kmnist", "svhn"}
        args.augmentation = args.model != "cnn" and args.dataset not in no_default_aug
    return args


def seed_everything(seed: int, deterministic: bool = True) -> dict[str, Any]:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    return {
        "seed": seed,
        "deterministic": deterministic,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def model_size_metrics(model: nn.Module) -> dict[str, int]:
    return {
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "num_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def model_spec_metrics(model: nn.Module) -> dict[str, Any]:
    spec = getattr(model, "spec", None)
    if spec is None:
        return {}
    return {
        "model_spec_name": getattr(spec, "name", ""),
        "model_feature_dim": getattr(spec, "feature_dim", None),
    }


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


def setup_wandb(args: argparse.Namespace, run_dir: Path) -> Any | None:
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested but wandb is not installed") from exc

    project = args.wandb_project or os.environ.get("WANDB_PROJECT") or "coarse-to-fine-curriculum"
    entity = args.wandb_entity or os.environ.get("WANDB_ENTITY") or None
    group = args.wandb_group or os.environ.get("WANDB_GROUP") or args.run_id
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run_name = f"{args.run_id}/{args.dataset}_{args.model}_{args.mode}"
    return wandb.init(
        project=project,
        entity=entity,
        group=group,
        name=run_name,
        tags=tags,
        config=vars(args),
        dir=str(run_dir),
    )


def hierarchy_to_markdown(
    hierarchy_levels: list[list[list[int]]],
    class_names: list[str] | None,
    title: str,
) -> str:
    lines = [f"# {title}", ""]
    for level_idx, clusters in enumerate(hierarchy_levels, start=1):
        lines.append(f"## Level {level_idx}: {len(clusters)} clusters")
        lines.append("")
        for cluster_idx, cluster in enumerate(clusters, start=1):
            names = [class_names[item] if class_names is not None else str(item) for item in cluster]
            lines.append(f"- Cluster {cluster_idx}: " + ", ".join(names))
        lines.append("")
    return "\n".join(lines)


def save_hierarchy_artifacts(
    run_dir: Path,
    hierarchy_levels: list[list[list[int]]],
    class_names: list[str] | None,
    distance_source: str,
    extra: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "distance_source": distance_source,
        "levels": hierarchy_levels,
        "class_names": class_names,
    }
    if extra:
        payload.update(extra)
    save_json(run_dir / "hierarchy.json", payload)
    title = f"Hierarchy ({distance_source})"
    (run_dir / "hierarchy.md").write_text(
        hierarchy_to_markdown(hierarchy_levels, class_names, title),
    )


def random_permutation_hierarchy(
    template_levels: list[list[list[int]]],
    num_classes: int,
    seed: int,
) -> tuple[list[list[list[int]]], list[int]]:
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(num_classes).astype(int).tolist()
    randomized_levels: list[list[list[int]]] = []
    for level in template_levels:
        randomized_level = []
        for cluster in level:
            randomized_level.append(
                sorted(int(permutation[int(class_id)]) for class_id in cluster)
            )
        randomized_levels.append(randomized_level)
    return randomized_levels, permutation


def log_wandb_artifacts(run_dir: Path, name: str) -> None:
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    artifact = wandb.Artifact(name=name, type="run_outputs")
    patterns = ["*.json", "*.csv", "*.md", "training_log_*.txt", "schedule.json", "hierarchy.json"]
    added = False
    for pattern in patterns:
        for path in run_dir.glob(pattern):
            if path.is_file():
                artifact.add_file(str(path), name=path.name)
                added = True
    if added:
        wandb.log_artifact(artifact)


def build_loaders(
    bundle: DatasetBundle,
    batch_size: int,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    def make_generator(offset: int) -> torch.Generator:
        generator = torch.Generator()
        generator.manual_seed(seed + offset)
        return generator

    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(
        bundle.train_dataset,
        shuffle=True,
        generator=make_generator(0),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        bundle.val_dataset,
        shuffle=False,
        generator=make_generator(1),
        **loader_kwargs,
    )
    test_loader = DataLoader(
        bundle.test_dataset,
        shuffle=False,
        generator=make_generator(2),
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def create_optimizer_and_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
    extra_parameters: list[nn.Parameter] | None = None,
) -> tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler | None]:
    parameters: list[Any] = [{"params": model.parameters()}]
    if extra_parameters:
        parameters.append({"params": extra_parameters, "weight_decay": 0.0})
    if args.optimizer == "adam":
        optimizer = optim.Adam(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            parameters,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

    if args.scheduler == "none":
        scheduler = None
    elif args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[round(args.epochs * 0.37), round(args.epochs * 0.75)],
            gamma=0.1,
        )
    return optimizer, scheduler


def hierarchy_distance_matrix_from_levels(
    hierarchy_levels: list[list[list[int]]],
    num_classes: int,
) -> np.ndarray:
    effective_levels = [clusters for clusters in hierarchy_levels if len(clusters) < num_classes]
    max_distance = len(effective_levels) + 1
    distances = np.full((num_classes, num_classes), max_distance, dtype=np.float32)
    np.fill_diagonal(distances, 0.0)
    for level_from_fine, clusters in enumerate(reversed(effective_levels), start=1):
        for cluster in clusters:
            idx = np.asarray(cluster, dtype=np.int64)
            distances[np.ix_(idx, idx)] = np.minimum(distances[np.ix_(idx, idx)], level_from_fine)
    return distances / max_distance


def classification_metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float]:
    confusion_f = confusion.to(dtype=torch.float64)
    tp = confusion_f.diag()
    pred_count = confusion_f.sum(dim=0)
    true_count = confusion_f.sum(dim=1)
    precision = tp / pred_count.clamp_min(1.0)
    recall = tp / true_count.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    support = true_count
    total = support.sum().clamp_min(1.0)
    present = support > 0
    return {
        "precision_macro": float(precision[present].mean().item()) if present.any() else 0.0,
        "recall_macro": float(recall[present].mean().item()) if present.any() else 0.0,
        "f1_macro": float(f1[present].mean().item()) if present.any() else 0.0,
        "f1_weighted": float(((f1 * support).sum() / total).item()),
        "balanced_acc": float(recall[present].mean().item()) if present.any() else 0.0,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_group_ids: list[int] | None = None,
    learned_hierdist_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top5_correct = 0
    total_samples = 0
    confidence_sum = 0.0
    ece_bins = 15
    ece_conf_sum = torch.zeros(ece_bins, dtype=torch.float64)
    ece_acc_sum = torch.zeros(ece_bins, dtype=torch.float64)
    ece_count = torch.zeros(ece_bins, dtype=torch.float64)
    confusion: torch.Tensor | None = None
    official_same_group_correct = 0
    official_hierdist_sum = 0.0
    group_ids_tensor = torch.tensor(class_group_ids, dtype=torch.long) if class_group_ids is not None else None
    learned_hierdist_sum = 0.0
    learned_hierdist_tensor = (
        torch.tensor(learned_hierdist_matrix, dtype=torch.float32)
        if learned_hierdist_matrix is not None
        else None
    )

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(inputs)
        if confusion is None:
            num_classes = logits.shape[1]
            confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        loss = F.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=1)
        confidence, preds = probs.max(dim=1)
        correct = preds == labels
        total_loss += loss.item() * labels.size(0)
        total_correct += correct.sum().item()
        if logits.shape[1] >= 5:
            top5 = logits.topk(5, dim=1).indices
            total_top5_correct += (top5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        total_samples += labels.size(0)
        confidence_sum += confidence.sum().item()

        labels_cpu = labels.cpu()
        preds_cpu = preds.cpu()
        if group_ids_tensor is not None:
            true_groups = group_ids_tensor[labels_cpu]
            pred_groups = group_ids_tensor[preds_cpu]
            same_group = true_groups == pred_groups
            official_same_group_correct += same_group.sum().item()
            official_dist = torch.where(correct.cpu(), 0.0, torch.where(same_group, 0.5, 1.0))
            official_hierdist_sum += official_dist.sum().item()
        if learned_hierdist_tensor is not None:
            learned_hierdist_sum += learned_hierdist_tensor[labels_cpu, preds_cpu].sum().item()
        indices = labels_cpu * confusion.shape[1] + preds_cpu
        confusion += torch.bincount(indices, minlength=confusion.numel()).reshape_as(confusion)

        bin_ids = torch.clamp((confidence.cpu() * ece_bins).long(), max=ece_bins - 1)
        correct_cpu = correct.cpu().to(dtype=torch.float64)
        confidence_cpu = confidence.cpu().to(dtype=torch.float64)
        ece_count += torch.bincount(bin_ids, minlength=ece_bins).to(dtype=torch.float64)
        ece_conf_sum += torch.bincount(bin_ids, weights=confidence_cpu, minlength=ece_bins)
        ece_acc_sum += torch.bincount(bin_ids, weights=correct_cpu, minlength=ece_bins)

    metrics = classification_metrics_from_confusion(confusion if confusion is not None else torch.zeros((1, 1), dtype=torch.int64))
    non_empty_bins = ece_count > 0
    ece = torch.tensor(0.0, dtype=torch.float64)
    if non_empty_bins.any():
        bin_conf = ece_conf_sum[non_empty_bins] / ece_count[non_empty_bins]
        bin_acc = ece_acc_sum[non_empty_bins] / ece_count[non_empty_bins]
        ece = ((ece_count[non_empty_bins] / max(total_samples, 1)) * (bin_acc - bin_conf).abs()).sum()

    metrics.update(
        {
            "loss": total_loss / max(total_samples, 1),
            "acc": total_correct / max(total_samples, 1),
            "top5_acc": total_top5_correct / max(total_samples, 1),
            "mean_confidence": confidence_sum / max(total_samples, 1),
            "ece": float(ece.item()),
        }
    )
    if group_ids_tensor is not None:
        official_hierdist = official_hierdist_sum / max(total_samples, 1)
        metrics.update(
            {
                "same_superclass_acc_official": official_same_group_correct / max(total_samples, 1),
                "hierdist_official": official_hierdist,
                "hier_score_official": 1.0 - official_hierdist,
            }
        )
    if learned_hierdist_tensor is not None:
        learned_hierdist = learned_hierdist_sum / max(total_samples, 1)
        metrics.update(
            {
                "hierdist_learned": learned_hierdist,
                "hier_score_learned": 1.0 - learned_hierdist,
            }
        )
    return metrics


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


@torch.inference_mode()
def teacher_embedding_distance(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> np.ndarray:
    feature_sums: torch.Tensor | None = None
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    was_training = model.training
    model.eval()
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        features = model.forward_features(inputs).detach().cpu().to(dtype=torch.float32)
        if feature_sums is None:
            feature_sums = torch.zeros((num_classes, features.shape[1]), dtype=torch.float32)
        for class_idx in range(num_classes):
            class_mask = labels == class_idx
            if not torch.any(class_mask):
                continue
            feature_sums[class_idx] += features[class_mask].sum(dim=0)
            class_counts[class_idx] += int(class_mask.sum().item())
    if feature_sums is None:
        raise RuntimeError("Teacher embedding distance could not be computed because the loader is empty")
    prototypes = feature_sums / class_counts.clamp_min(1).unsqueeze(1)
    prototypes = F.normalize(prototypes, p=2, dim=1)
    sim = prototypes @ prototypes.T
    dist = 1.0 - sim.numpy()
    missing = class_counts == 0
    if torch.any(missing):
        dist[missing.numpy(), :] = 1.0
        dist[:, missing.numpy()] = 1.0
    np.fill_diagonal(dist, 0.0)
    if was_training:
        model.train()
    return dist


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
        membership[cluster_tensor] = cluster_mask.unsqueeze(0).expand(len(cluster), -1)
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


def parse_weight_list(value: str, expected: int) -> list[float]:
    weights = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not weights:
        raise ValueError("At least one multiloss weight is required")
    if any(weight <= 0 for weight in weights):
        raise ValueError("All multiloss weights must be positive")
    if len(weights) < expected:
        weights.extend([weights[-1]] * (expected - len(weights)))
    return weights[:expected]


def log_weights_from_initial_weights(weights: list[float], device: torch.device) -> nn.Parameter:
    raw = torch.log(torch.tensor(weights, dtype=torch.float32, device=device))
    return nn.Parameter(raw)


def normalized_gradnorm_weights(log_weights: torch.Tensor) -> torch.Tensor:
    return torch.softmax(log_weights, dim=0) * log_weights.numel()


def parse_epoch_set(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def flatten_tensors(tensors: list[torch.Tensor | None], parameters: list[nn.Parameter]) -> torch.Tensor:
    chunks = []
    for tensor, parameter in zip(tensors, parameters, strict=True):
        if tensor is None:
            chunks.append(torch.zeros_like(parameter).reshape(-1))
        else:
            chunks.append(tensor.reshape(-1))
    return torch.cat(chunks)


def parameter_norm(parameters: list[nn.Parameter]) -> torch.Tensor:
    return torch.sqrt(sum(torch.sum(parameter.detach() ** 2) for parameter in parameters)).clamp_min(1e-12)


def assign_parameter_vector(parameters: list[nn.Parameter], vector: torch.Tensor, scale: float) -> None:
    cursor = 0
    with torch.no_grad():
        for parameter in parameters:
            numel = parameter.numel()
            parameter.add_(vector[cursor : cursor + numel].view_as(parameter), alpha=scale)
            cursor += numel


def fixed_probe_batches(loader: DataLoader, device: torch.device, max_batches: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches = []
    for batch_idx, (inputs, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batches.append((inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)))
    return batches


def build_probe_objective(
    model: nn.Module,
    probe_batches: list[tuple[torch.Tensor, torch.Tensor]],
    membership: torch.Tensor | None,
    multiloss_memberships: list[tuple[str, torch.Tensor]] | None,
    args: argparse.Namespace,
    adaptive_log_weights: nn.Parameter | None,
) -> torch.Tensor:
    losses = []
    for inputs, labels in probe_batches:
        logits = model(inputs)
        if multiloss_memberships is None:
            losses.append(
                F.cross_entropy(logits, labels)
                if membership is None
                else marginalized_loss(logits, labels, membership)
            )
        else:
            component_losses = [F.cross_entropy(logits, labels)]
            for _, coarse_membership in multiloss_memberships:
                component_losses.append(marginalized_loss(logits, labels, coarse_membership))
            if args.multi_weighting == "static":
                weights = parse_weight_list(args.multi_static_weights, len(component_losses))
                losses.append(
                    sum(weight * component for weight, component in zip(weights, component_losses, strict=True))
                )
            elif args.multi_weighting == "uncertainty":
                assert adaptive_log_weights is not None
                precisions = torch.exp(-adaptive_log_weights.detach())
                losses.append(
                    sum(
                        precision * component
                        for precision, component in zip(precisions, component_losses, strict=True)
                    )
                )
            else:
                assert adaptive_log_weights is not None
                weights_tensor = normalized_gradnorm_weights(adaptive_log_weights.detach())
                losses.append(
                    sum(
                        weight * component
                        for weight, component in zip(weights_tensor, component_losses, strict=True)
                    )
                )
    return torch.stack(losses).mean()


def estimate_roughness_metrics(
    model: nn.Module,
    probe_batches: list[tuple[torch.Tensor, torch.Tensor]],
    membership: torch.Tensor | None,
    multiloss_memberships: list[tuple[str, torch.Tensor]] | None,
    args: argparse.Namespace,
    adaptive_log_weights: nn.Parameter | None,
) -> dict[str, float]:
    if not probe_batches:
        return {}
    was_training = model.training
    model.eval()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    metrics: dict[str, float] = {}

    per_batch_grads = []
    per_batch_norms = []
    for inputs, labels in probe_batches:
        model.zero_grad(set_to_none=True)
        loss = build_probe_objective(
            model,
            [(inputs, labels)],
            membership,
            multiloss_memberships,
            args,
            adaptive_log_weights,
        )
        grads = torch.autograd.grad(loss, parameters, retain_graph=False, create_graph=False, allow_unused=True)
        flat = flatten_tensors(list(grads), parameters).detach()
        per_batch_grads.append(flat)
        per_batch_norms.append(torch.norm(flat, p=2))

    grad_stack = torch.stack(per_batch_grads)
    grad_norms = torch.stack(per_batch_norms)
    grad_norm_mean = grad_norms.mean()
    grad_norm_std = grad_norms.std(unbiased=False) if grad_norms.numel() > 1 else torch.tensor(0.0, device=grad_norms.device)
    centered = grad_norms - grad_norm_mean
    grad_norm_skew = torch.mean(centered**3) / grad_norm_std.clamp_min(1e-12) ** 3
    mean_grad = grad_stack.mean(dim=0)
    mean_grad_norm_sq = torch.sum(mean_grad**2).clamp_min(1e-12)
    mean_sq_grad_norm = torch.mean(torch.sum(grad_stack**2, dim=1))
    grad_noise_scale = (mean_sq_grad_norm - mean_grad_norm_sq).clamp_min(0.0) / mean_grad_norm_sq
    metrics.update(
        {
            "rough_grad_norm_mean": float(grad_norm_mean.item()),
            "rough_grad_norm_std": float(grad_norm_std.item()),
            "rough_grad_norm_cv": float((grad_norm_std / grad_norm_mean.clamp_min(1e-12)).item()),
            "rough_grad_norm_skew": float(grad_norm_skew.item()),
            "rough_gradient_noise_scale": float(grad_noise_scale.item()),
        }
    )

    model.zero_grad(set_to_none=True)
    base_loss = build_probe_objective(
        model,
        probe_batches,
        membership,
        multiloss_memberships,
        args,
        adaptive_log_weights,
    )
    grads = torch.autograd.grad(base_loss, parameters, create_graph=True, allow_unused=True)
    flat_grad = flatten_tensors(list(grads), parameters)
    grad_norm = torch.norm(flat_grad.detach(), p=2).clamp_min(1e-12)
    rho = args.sharpness_rho * float(parameter_norm(parameters).item())
    direction = flat_grad.detach() / grad_norm
    assign_parameter_vector(parameters, direction, rho)
    with torch.no_grad():
        perturbed_loss = build_probe_objective(
            model,
            probe_batches,
            membership,
            multiloss_memberships,
            args,
            adaptive_log_weights,
        )
    assign_parameter_vector(parameters, direction, -rho)
    metrics["rough_critical_sharpness"] = float((perturbed_loss - base_loss.detach()).item())
    metrics["rough_relative_critical_sharpness"] = float(
        ((perturbed_loss - base_loss.detach()) / base_loss.detach().abs().clamp_min(1e-12)).item()
    )

    def hessian_vector_product(vector: torch.Tensor) -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        loss = build_probe_objective(
            model,
            probe_batches,
            membership,
            multiloss_memberships,
            args,
            adaptive_log_weights,
        )
        first_grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
        flat_first_grads = flatten_tensors(list(first_grads), parameters)
        grad_dot_vector = torch.dot(flat_first_grads, vector)
        hvp = torch.autograd.grad(grad_dot_vector, parameters, retain_graph=False, allow_unused=True)
        return flatten_tensors(list(hvp), parameters).detach()

    vector = torch.randn_like(flat_grad.detach())
    vector = vector / torch.norm(vector, p=2).clamp_min(1e-12)
    top_eigenvalue = torch.tensor(0.0, device=vector.device)
    for _ in range(max(args.hessian_iters, 1)):
        hvp = hessian_vector_product(vector)
        hvp_norm = torch.norm(hvp, p=2).clamp_min(1e-12)
        vector = hvp / hvp_norm
        top_eigenvalue = torch.dot(vector, hessian_vector_product(vector))
    metrics["rough_hessian_top_eigenvalue"] = float(top_eigenvalue.item())

    frobenius_estimates = []
    trace_estimates = []
    for _ in range(max(args.hessian_samples, 1)):
        rademacher = torch.randint(0, 2, flat_grad.shape, device=flat_grad.device, dtype=flat_grad.dtype) * 2 - 1
        hvp = hessian_vector_product(rademacher)
        frobenius_estimates.append(torch.sum(hvp**2))
        trace_estimates.append(torch.dot(rademacher, hvp))
    metrics["rough_hessian_frobenius"] = float(torch.sqrt(torch.stack(frobenius_estimates).mean()).item())
    metrics["rough_hessian_trace"] = float(torch.stack(trace_estimates).mean().item())

    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return metrics


def build_multiloss_memberships(
    hierarchy_levels: list[list[list[int]]],
    num_classes: int,
    device: torch.device,
) -> list[tuple[str, torch.Tensor]]:
    effective_levels = [clusters for clusters in hierarchy_levels if len(clusters) < num_classes]
    memberships: list[tuple[str, torch.Tensor]] = []
    for idx, clusters in enumerate(effective_levels[:3], start=1):
        memberships.append((f"coarse_{idx}_{len(clusters)}", clusters_to_membership(clusters, num_classes, device)))
    return memberships


def filtered_curriculum_levels(
    hierarchy_levels: list[list[list[int]]],
    num_classes: int,
    min_clusters: int = 0,
    max_levels: int = 0,
) -> list[list[list[int]]]:
    effective_levels = [
        clusters
        for clusters in hierarchy_levels
        if len(clusters) < num_classes and len(clusters) >= max(min_clusters, 0)
    ]
    if max_levels > 0:
        effective_levels = effective_levels[:max_levels]
    return effective_levels


def build_curriculum_schedule(
    num_classes: int,
    hierarchy_levels: list[list[list[int]]],
    curriculum_epochs: int,
    total_epochs: int,
    min_clusters: int = 0,
    max_levels: int = 0,
    policy: str = "fixed",
    stage_max_epochs: int = 50,
    curriculum_order: str = "easy_to_hard",
) -> list[dict[str, Any]]:
    schedule: list[dict[str, Any]] = []
    effective_levels = filtered_curriculum_levels(
        hierarchy_levels,
        num_classes,
        min_clusters=min_clusters,
        max_levels=max_levels,
    )
    if curriculum_order == "hard_to_easy":
        effective_levels = list(reversed(effective_levels))
    num_levels = len(effective_levels)
    if policy == "adaptive_plateau":
        for level_idx, clusters in enumerate(effective_levels):
            schedule.append(
                {
                    "name": f"level_{level_idx + 1}_{len(clusters)}clusters",
                    "clusters": clusters,
                    "epochs": max(stage_max_epochs, 1),
                    "adaptive": True,
                }
            )
        schedule.append(
            {
                "name": "fine_tune",
                "clusters": singleton_clusters(num_classes),
                "epochs": total_epochs,
                "adaptive": False,
            }
        )
        return schedule

    if num_levels > 0 and curriculum_epochs > 0:
        base_epochs = curriculum_epochs // num_levels
        remainder = curriculum_epochs % num_levels
        for level_idx, clusters in enumerate(effective_levels):
            epochs_this_level = base_epochs + (1 if level_idx < remainder else 0)
            if epochs_this_level <= 0:
                continue
            schedule.append(
                {
                    "name": f"level_{level_idx + 1}_{len(clusters)}clusters",
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
        cnn_width_multiplier=args.cnn_width_multiplier,
        cifar_resnet_width_multiplier=args.cifar_resnet_width_multiplier,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    history = json.loads(history_path.read_text())
    return model, history


def load_checkpoint_state_dict(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint format at {checkpoint_path}")


def teacher_model_spec_from_run_dir(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Teacher run directory is missing config.json: {run_dir}")
    config = json.loads(config_path.read_text())
    return {
        "model": config.get("model"),
        "cnn_width_multiplier": config.get("cnn_width_multiplier", 1.0),
        "cifar_resnet_width_multiplier": config.get("cifar_resnet_width_multiplier", 1.0),
    }


def load_teacher_model(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    device: torch.device,
) -> tuple[nn.Module, dict[str, Any]]:
    teacher_run_dir = Path(args.teacher_run_dir) if args.teacher_run_dir else None
    if teacher_run_dir is not None:
        checkpoint_path = teacher_run_dir / "best_model.pt"
        spec = teacher_model_spec_from_run_dir(teacher_run_dir)
        source = {"teacher_run_dir": str(teacher_run_dir)}
    elif args.teacher_checkpoint_path:
        checkpoint_path = Path(args.teacher_checkpoint_path)
        spec = {
            "model": args.teacher_model,
            "cnn_width_multiplier": args.teacher_cnn_width_multiplier,
            "cifar_resnet_width_multiplier": args.teacher_cifar_resnet_width_multiplier,
        }
        source = {"teacher_checkpoint_path": str(checkpoint_path)}
    elif args.teacher_pretrained_source == "torchvision_imagenet":
        checkpoint_path = None
        spec = {
            "model": args.teacher_model,
            "cnn_width_multiplier": args.teacher_cnn_width_multiplier,
            "cifar_resnet_width_multiplier": args.teacher_cifar_resnet_width_multiplier,
        }
        source = {"teacher_pretrained_source": args.teacher_pretrained_source}
    else:
        raise ValueError(
            "teacher_embeddings requires --teacher_run_dir, --teacher_checkpoint_path, or --teacher_pretrained_source torchvision_imagenet"
        )

    model_name = spec.get("model") or args.teacher_model
    if model_name is None:
        raise ValueError("Teacher model architecture is required to load teacher_embeddings")
    teacher_cnn_width = spec.get("cnn_width_multiplier")
    if teacher_cnn_width is None:
        teacher_cnn_width = args.teacher_cnn_width_multiplier if args.teacher_cnn_width_multiplier is not None else 1.0
    teacher_cifar_width = spec.get("cifar_resnet_width_multiplier")
    if teacher_cifar_width is None:
        teacher_cifar_width = (
            args.teacher_cifar_resnet_width_multiplier
            if args.teacher_cifar_resnet_width_multiplier is not None
            else 1.0
        )

    if checkpoint_path is not None and not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")

    pretrained_backbone = args.teacher_pretrained_source == "torchvision_imagenet" and checkpoint_path is None
    if pretrained_backbone and model_name not in {"resnet18", "resnet50"}:
        raise ValueError("torchvision_imagenet teacher source currently supports only resnet18 or resnet50")

    model = build_model(
        model_name=model_name,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        dropout=0.0,
        cnn_width_multiplier=float(teacher_cnn_width),
        cifar_resnet_width_multiplier=float(teacher_cifar_width),
        pretrained_backbone=pretrained_backbone,
    ).to(device)
    if checkpoint_path is not None:
        model.load_state_dict(load_checkpoint_state_dict(checkpoint_path, device))
    return model, {
        **source,
        "teacher_model": model_name,
        "teacher_cnn_width_multiplier": float(teacher_cnn_width),
        "teacher_cifar_resnet_width_multiplier": float(teacher_cifar_width),
    }


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
    multiloss_memberships: list[tuple[str, torch.Tensor]] | None = None,
    class_group_ids: list[int] | None = None,
    learned_hierdist_matrix: np.ndarray | None = None,
) -> tuple[nn.Module, list[dict[str, Any]], dict[str, float]]:
    multiloss_names = ["fine"] + [name for name, _ in (multiloss_memberships or [])]
    adaptive_log_weights: nn.Parameter | None = None
    gradnorm_optimizer: optim.Optimizer | None = None
    initial_gradnorm_losses: torch.Tensor | None = None
    if multiloss_memberships is not None:
        expected_weights = len(multiloss_names)
        if args.multi_weighting in {"uncertainty", "gradnorm"}:
            adaptive_log_weights = log_weights_from_initial_weights(
                parse_weight_list(args.multi_initial_weights, expected_weights),
                device,
            )
        if args.multi_weighting == "gradnorm" and adaptive_log_weights is not None:
            gradnorm_optimizer = optim.Adam([adaptive_log_weights], lr=args.lr)

    extra_params = [adaptive_log_weights] if adaptive_log_weights is not None and args.multi_weighting == "uncertainty" else None
    optimizer, scheduler = create_optimizer_and_scheduler(model, args, extra_params)
    use_amp = bool(args.amp and device.type == "cuda" and args.multi_weighting != "gradnorm")
    if args.amp and device.type == "cuda" and args.multi_weighting == "gradnorm":
        logger.info("Disabling AMP for GradNorm because it needs higher-order gradients")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history: list[dict[str, Any]] = []
    best_val_acc = -float("inf")
    best_epoch = -1
    best_path = run_dir / "best_model.pt"
    last_path = run_dir / "last_model.pt"
    current_stage_name = None
    adaptive_stage_idx = 0
    adaptive_stage_epoch = 0
    adaptive_stage_best_val = -float("inf")
    adaptive_stage_bad_epochs = 0
    use_adaptive_curriculum = bool(schedule) and args.curriculum_policy == "adaptive_plateau"
    roughness_epochs = parse_epoch_set(args.roughness_epochs) if args.roughness_probes else set()
    probe_batches = (
        fixed_probe_batches(val_loader, device, max(args.roughness_batches, 1))
        if args.roughness_probes
        else []
    )

    for epoch in range(args.epochs):
        if use_adaptive_curriculum:
            assert schedule is not None
            stage = schedule[min(adaptive_stage_idx, len(schedule) - 1)]
            adaptive_stage_epoch += 1
        else:
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
        loss_component_totals = {name: 0.0 for name in multiloss_names}
        weight_component_totals = {name: 0.0 for name in multiloss_names}
        gradnorm_component_totals = {name: 0.0 for name in multiloss_names}
        num_weight_observations = 0
        static_weights: list[float] | None = None
        if multiloss_memberships is not None and args.multi_weighting == "static":
            static_weights = parse_weight_list(args.multi_static_weights, len(multiloss_names))

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if gradnorm_optimizer is not None:
                gradnorm_optimizer.zero_grad(set_to_none=True)
            autocast_ctx = (lambda: torch.amp.autocast("cuda")) if use_amp else nullcontext
            with autocast_ctx():
                logits = model(inputs)
                if multiloss_memberships is None:
                    if membership is None:
                        loss = F.cross_entropy(logits, labels)
                    else:
                        loss = marginalized_loss(logits, labels, membership)
                    component_losses = [loss]
                    component_weights = [1.0]
                    gradnorm_values = [0.0]
                else:
                    component_losses = [F.cross_entropy(logits, labels)]
                    for _, coarse_membership in multiloss_memberships:
                        component_losses.append(marginalized_loss(logits, labels, coarse_membership))
                    if args.multi_weighting == "static":
                        component_weights = static_weights or [1.0] * len(component_losses)
                        loss = sum(weight * component for weight, component in zip(component_weights, component_losses, strict=True))
                        gradnorm_values = [0.0] * len(component_losses)
                    elif args.multi_weighting == "uncertainty":
                        assert adaptive_log_weights is not None
                        precisions = torch.exp(-adaptive_log_weights)
                        loss = sum(
                            precision * component + log_weight
                            for precision, component, log_weight in zip(
                                precisions,
                                component_losses,
                                adaptive_log_weights,
                                strict=True,
                            )
                        )
                        component_weights = precisions.detach().cpu().tolist()
                        gradnorm_values = [0.0] * len(component_losses)
                    else:
                        assert adaptive_log_weights is not None
                        assert gradnorm_optimizer is not None
                        weights_tensor = normalized_gradnorm_weights(adaptive_log_weights)
                        component_weights = weights_tensor.detach().cpu().tolist()
                        if initial_gradnorm_losses is None:
                            initial_gradnorm_losses = torch.stack([item.detach() for item in component_losses]).clamp_min(1e-8)
                        shared_parameter = model.classifier_weight
                        grad_norms = []
                        for weight, component in zip(weights_tensor, component_losses, strict=True):
                            grad = torch.autograd.grad(
                                weight * component,
                                shared_parameter,
                                retain_graph=True,
                                create_graph=True,
                            )[0]
                            grad_norms.append(torch.norm(grad, p=2))
                        grad_norm_tensor = torch.stack(grad_norms)
                        loss_ratios = torch.stack([item.detach() for item in component_losses]).clamp_min(1e-8) / initial_gradnorm_losses
                        inverse_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-8)
                        target_grad_norms = grad_norm_tensor.detach().mean() * (inverse_rates ** args.gradnorm_alpha)
                        gradnorm_loss = F.l1_loss(grad_norm_tensor, target_grad_norms, reduction="sum")
                        gradnorm_loss.backward(retain_graph=True)
                        gradnorm_optimizer.step()
                        loss = sum(
                            weight.detach() * component
                            for weight, component in zip(weights_tensor, component_losses, strict=True)
                        )
                        gradnorm_values = grad_norm_tensor.detach().cpu().tolist()
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
            for name, component in zip(multiloss_names, component_losses, strict=True):
                loss_component_totals[name] += float(component.detach().item()) * labels.size(0)
            for name, weight in zip(multiloss_names, component_weights, strict=True):
                weight_component_totals[name] += float(weight)
            for name, grad_norm in zip(multiloss_names, gradnorm_values, strict=True):
                gradnorm_component_totals[name] += float(grad_norm)
            num_weight_observations += 1

        if scheduler is not None:
            scheduler.step()

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            class_group_ids=class_group_ids,
            learned_hierdist_matrix=learned_hierdist_matrix,
        )
        adaptive_advance_reason = ""
        if use_adaptive_curriculum and stage is not None and stage["name"] != "fine_tune":
            val_acc = float(val_metrics["acc"])
            if val_acc > adaptive_stage_best_val + args.curriculum_stage_min_delta:
                adaptive_stage_best_val = val_acc
                adaptive_stage_bad_epochs = 0
            else:
                adaptive_stage_bad_epochs += 1
            min_reached = adaptive_stage_epoch >= max(args.curriculum_stage_min_epochs, 1)
            patience_reached = adaptive_stage_bad_epochs >= max(args.curriculum_stage_patience, 1)
            max_reached = adaptive_stage_epoch >= max(args.curriculum_stage_max_epochs, 1)
            if min_reached and (patience_reached or max_reached):
                adaptive_advance_reason = "max_epochs" if max_reached else "plateau"
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            class_group_ids=class_group_ids,
            learned_hierdist_matrix=learned_hierdist_matrix,
        )
        roughness_metrics = {}
        if args.roughness_probes and (epoch + 1) in roughness_epochs:
            logger.info("Computing roughness probes for epoch %d", epoch + 1)
            roughness_metrics = estimate_roughness_metrics(
                model=model,
                probe_batches=probe_batches,
                membership=membership,
                multiloss_memberships=multiloss_memberships,
                args=args,
                adaptive_log_weights=adaptive_log_weights,
            )
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
        if use_adaptive_curriculum:
            record["adaptive_stage_epoch"] = adaptive_stage_epoch
            record["adaptive_stage_best_val_acc"] = adaptive_stage_best_val
            record["adaptive_stage_bad_epochs"] = adaptive_stage_bad_epochs
            record["adaptive_advance_reason"] = adaptive_advance_reason
        for metric_name, metric_value in val_metrics.items():
            if metric_name not in {"loss", "acc"}:
                record[f"val_{metric_name}"] = metric_value
        for metric_name, metric_value in test_metrics.items():
            if metric_name not in {"loss", "acc"}:
                record[f"test_{metric_name}"] = metric_value
        record.update(roughness_metrics)
        if multiloss_memberships is not None:
            for name in multiloss_names:
                record[f"loss_{name}"] = loss_component_totals[name] / max(total_train_samples, 1)
                record[f"weight_{name}"] = weight_component_totals[name] / max(num_weight_observations, 1)
                if args.multi_weighting == "gradnorm":
                    record[f"gradnorm_{name}"] = gradnorm_component_totals[name] / max(num_weight_observations, 1)
        history.append(record)
        if args.wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(record, step=epoch + 1)
            except ImportError:
                pass
        if adaptive_advance_reason:
            assert schedule is not None
            old_stage_name = str(stage["name"] if stage is not None else "")
            adaptive_stage_idx = min(adaptive_stage_idx + 1, len(schedule) - 1)
            new_stage_name = str(schedule[adaptive_stage_idx]["name"])
            logger.info(
                "Adaptive curriculum advancing from %s to %s after %d epochs (%s)",
                old_stage_name,
                new_stage_name,
                adaptive_stage_epoch,
                adaptive_advance_reason,
            )
            adaptive_stage_epoch = 0
            adaptive_stage_best_val = -float("inf")
            adaptive_stage_bad_epochs = 0

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
            if args.save_checkpoints:
                save_checkpoint(best_path, model, optimizer, scheduler, epoch + 1, best_val_acc, args)

        if args.save_checkpoints:
            save_checkpoint(last_path, model, optimizer, scheduler, epoch + 1, best_val_acc, args)
        save_json(run_dir / "history.json", history)
        save_rows_csv(run_dir / "history.csv", history)

        if args.patience > 0 and epoch - best_epoch >= args.patience:
            logger.info("Early stopping triggered after %d epochs without val improvement", args.patience)
            break

    if args.save_checkpoints and best_path.exists():
        best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(best_checkpoint["model_state_dict"])
    final_test_metrics = evaluate(
        model,
        test_loader,
        device,
        class_group_ids=class_group_ids,
        learned_hierdist_matrix=learned_hierdist_matrix,
    )
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
        cnn_width_multiplier=args.cnn_width_multiplier,
        cifar_resnet_width_multiplier=args.cifar_resnet_width_multiplier,
        pretrained_backbone=args.pretrained_backbone,
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
        class_group_ids=bundle.class_group_ids,
    )
    baseline_dist_matrix = classifier_weight_distance(model)
    np.save(run_dir / "distance_matrix_classifier_weights.npy", baseline_dist_matrix)
    baseline_hierarchy = compute_hierarchy(baseline_dist_matrix, seed=args.seed)
    save_hierarchy_artifacts(
        run_dir,
        baseline_hierarchy,
        bundle.class_names,
        "classifier_weights",
    )
    baseline_learned_hierdist_matrix = hierarchy_distance_matrix_from_levels(
        baseline_hierarchy,
        bundle.num_classes,
    )

    if args.export_teacher_hierarchy:
        teacher_loader = {
            "train": loaders[0],
            "val": loaders[1],
            "test": loaders[2],
        }[args.export_teacher_hierarchy_split]
        teacher_export_dir = (
            Path(args.export_teacher_hierarchy_dir)
            if args.export_teacher_hierarchy_dir
            else run_dir.parent / f"{bundle.name}_{args.model}_teacher_hierarchy_{args.export_teacher_hierarchy_split}"
        )
        teacher_export_dir.mkdir(parents=True, exist_ok=True)
        teacher_dist_matrix = teacher_embedding_distance(model, teacher_loader, bundle.num_classes, device)
        np.save(teacher_export_dir / "distance_matrix_teacher_embeddings.npy", teacher_dist_matrix)
        teacher_hierarchy = compute_hierarchy(teacher_dist_matrix, seed=args.seed)
        save_hierarchy_artifacts(
            teacher_export_dir,
            teacher_hierarchy,
            bundle.class_names,
            "teacher_embeddings",
            extra={
                "teacher_source": "trained_baseline",
                "teacher_run_dir": str(run_dir),
                "teacher_embedding_split": args.export_teacher_hierarchy_split,
                "pretrained_backbone": bool(args.pretrained_backbone),
            },
        )
        save_json(
            run_dir / "teacher_hierarchy_export.json",
            {
                "teacher_export_dir": str(teacher_export_dir),
                "teacher_embedding_split": args.export_teacher_hierarchy_split,
                "pretrained_backbone": bool(args.pretrained_backbone),
            },
        )
        logger.info("Exported teacher embedding hierarchy to %s", teacher_export_dir)
        if args.wandb:
            try:
                import wandb
                if wandb.run is not None:
                    artifact = wandb.Artifact(
                        name=f"{args.run_id}-{bundle.name}-{args.model}-teacher-hierarchy",
                        type="teacher_hierarchy",
                    )
                    for filename in [
                        "hierarchy.json",
                        "hierarchy.md",
                        "distance_matrix_teacher_embeddings.npy",
                    ]:
                        path = teacher_export_dir / filename
                        if path.exists():
                            artifact.add_file(str(path), name=filename)
                    wandb.log_artifact(artifact)
            except ImportError:
                pass
    final_learned_hier_metrics = evaluate(
        model,
        loaders[2],
        device,
        class_group_ids=bundle.class_group_ids,
        learned_hierdist_matrix=baseline_learned_hierdist_matrix,
    )
    val_artifacts = save_evaluation_artifacts(model, loaders[1], device, bundle, run_dir, "val")
    test_artifacts = save_evaluation_artifacts(model, loaders[2], device, bundle, run_dir, "test")
    result = {
        "mode": "baseline",
        "dataset": bundle.name,
        "model": args.model,
        "cnn_width_multiplier": args.cnn_width_multiplier,
        "cifar_resnet_width_multiplier": args.cifar_resnet_width_multiplier,
        **model_size_metrics(model),
        **model_spec_metrics(model),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "best_val_acc": max(float(entry["val_acc"]) for entry in history),
        "best_test_acc": max(float(entry["test_acc"]) for entry in history),
        "best_test_f1_macro": max(float(entry.get("test_f1_macro", 0.0)) for entry in history),
        "final_test_acc": float(test_metrics["acc"]),
        "final_test_f1_macro": float(test_metrics["f1_macro"]),
        "final_test_precision_macro": float(test_metrics["precision_macro"]),
        "final_test_recall_macro": float(test_metrics["recall_macro"]),
        "final_test_top5_acc": float(test_metrics["top5_acc"]),
        "final_test_ece": float(test_metrics["ece"]),
        "final_test_hierdist_official": float(test_metrics.get("hierdist_official", 0.0)),
        "final_test_hier_score_official": float(test_metrics.get("hier_score_official", 0.0)),
        "final_test_same_superclass_acc_official": float(
            test_metrics.get("same_superclass_acc_official", 0.0)
        ),
        "final_test_hierdist_learned": float(final_learned_hier_metrics["hierdist_learned"]),
        "final_test_hier_score_learned": float(final_learned_hier_metrics["hier_score_learned"]),
        "val_difficulty_class_accuracy_variance": float(val_artifacts["class_accuracy_variance"]),
        "test_difficulty_class_accuracy_variance": float(test_artifacts["class_accuracy_variance"]),
        "test_difficulty_confusion_entropy_mean": float(test_artifacts["confusion_entropy_mean"]),
    }
    save_json(run_dir / "results.json", result)
    save_single_row_csv(run_dir / "summary.csv", result)
    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.summary.update(result)
                log_wandb_artifacts(run_dir, f"{args.run_id}-{bundle.name}-{args.model}-baseline")
        except ImportError:
            pass
    return result


def load_or_train_reference(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    run_dir: Path,
) -> tuple[Path, nn.Module | None, list[dict[str, Any]]]:
    train_loader, val_loader, test_loader = loaders
    if args.reference_run_dir is not None:
        reference_dir = Path(args.reference_run_dir)
        logger.info("Loading reference baseline from %s", reference_dir)
        if (reference_dir / "best_model.pt").exists():
            reference_model, reference_history = load_reference_model(reference_dir, args, bundle, device)
        else:
            reference_model = None
            reference_history = json.loads((reference_dir / "history.json").read_text())
        return reference_dir, reference_model, reference_history

    reference_dir = run_dir / "reference_baseline"
    reference_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Training reference baseline for hierarchy construction")
    reference_logger = setup_logger(reference_dir, "training_log_reference.txt")
    reference_model = build_model(
        model_name=args.model,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        dropout=args.dropout,
        cnn_width_multiplier=args.cnn_width_multiplier,
        cifar_resnet_width_multiplier=args.cifar_resnet_width_multiplier,
        pretrained_backbone=args.pretrained_backbone,
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
        class_group_ids=bundle.class_group_ids,
    )
    np.save(reference_dir / "distance_matrix_classifier_weights.npy", classifier_weight_distance(reference_model))
    save_evaluation_artifacts(reference_model, val_loader, device, bundle, reference_dir, "val")
    save_evaluation_artifacts(reference_model, test_loader, device, bundle, reference_dir, "test")
    return reference_dir, reference_model, reference_history


def reference_classifier_weight_distance(
    reference_dir: Path,
    reference_model: nn.Module | None,
) -> np.ndarray:
    if reference_model is not None:
        return classifier_weight_distance(reference_model)
    dist_path = reference_dir / "distance_matrix_classifier_weights.npy"
    if not dist_path.exists():
        raise FileNotFoundError(f"Missing reference distance matrix: {dist_path}")
    return np.load(dist_path)


def hierarchy_from_reference(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    run_dir: Path,
    reference_dir: Path,
    reference_model: nn.Module | None,
) -> list[list[list[int]]]:
    hierarchy_extra: dict[str, Any] = {}
    if args.distance_source in {"classifier_weights", "random_permutation"}:
        dist_matrix = reference_classifier_weight_distance(reference_dir, reference_model)
        template_levels = compute_hierarchy(dist_matrix, seed=args.seed)
        if args.distance_source == "random_permutation":
            random_seed = (
                args.random_hierarchy_seed if args.random_hierarchy_seed is not None else args.seed
            )
            hierarchy_levels, permutation = random_permutation_hierarchy(
                template_levels,
                bundle.num_classes,
                random_seed,
            )
            save_json(
                run_dir / "hierarchy_template_classifier_weights.json",
                {"distance_source": "classifier_weights", "levels": template_levels},
            )
            hierarchy_extra = {
                "template_distance_source": "classifier_weights",
                "random_hierarchy_seed": random_seed,
                "random_permutation": permutation,
            }
        else:
            hierarchy_levels = template_levels
    elif args.distance_source == "confusion":
        if reference_model is None:
            raise RuntimeError("confusion distance requires a reference checkpoint/model")
        dist_matrix = confusion_distance(reference_model, val_loader, bundle.num_classes, device)
        hierarchy_levels = compute_hierarchy(dist_matrix, seed=args.seed)
    else:
        teacher_model, teacher_meta = load_teacher_model(args, bundle, device)
        teacher_loader = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        }[args.teacher_embedding_split]
        dist_matrix = teacher_embedding_distance(teacher_model, teacher_loader, bundle.num_classes, device)
        hierarchy_levels = compute_hierarchy(dist_matrix, seed=args.seed)
        hierarchy_extra = {
            **teacher_meta,
            "teacher_embedding_split": args.teacher_embedding_split,
        }
    np.save(run_dir / "distance_matrix.npy", dist_matrix)

    save_hierarchy_artifacts(
        run_dir,
        hierarchy_levels,
        bundle.class_names,
        args.distance_source,
        extra=hierarchy_extra,
    )
    logger.info("Built hierarchy with %d levels", len(hierarchy_levels))
    return hierarchy_levels


def curriculum_run(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    run_dir: Path,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = loaders

    reference_dir, reference_model, reference_history = load_or_train_reference(
        args,
        bundle,
        loaders,
        device,
        logger,
        run_dir,
    )
    hierarchy_levels = hierarchy_from_reference(
        args,
        bundle,
        train_loader,
        val_loader,
        test_loader,
        device,
        logger,
        run_dir,
        reference_dir,
        reference_model,
    )

    if args.curriculum_policy == "adaptive_plateau":
        curriculum_epochs = args.epochs
        logger.info(
            "Adaptive plateau curriculum | min_clusters=%d | max_levels=%d | min_epochs=%d | max_epochs=%d | patience=%d | min_delta=%.4f",
            args.curriculum_min_clusters,
            args.curriculum_max_levels,
            args.curriculum_stage_min_epochs,
            args.curriculum_stage_max_epochs,
            args.curriculum_stage_patience,
            args.curriculum_stage_min_delta,
        )
    elif args.curriculum_epochs is None:
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
        min_clusters=args.curriculum_min_clusters,
        max_levels=args.curriculum_max_levels,
        policy=args.curriculum_policy,
        stage_max_epochs=args.curriculum_stage_max_epochs,
        curriculum_order=args.curriculum_order,
    )
    save_json(run_dir / "schedule.json", schedule)
    learned_hierdist_matrix = hierarchy_distance_matrix_from_levels(hierarchy_levels, bundle.num_classes)

    model = build_model(
        model_name=args.model,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        dropout=args.dropout,
        cnn_width_multiplier=args.cnn_width_multiplier,
        cifar_resnet_width_multiplier=args.cifar_resnet_width_multiplier,
        pretrained_backbone=args.pretrained_backbone,
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
        class_group_ids=bundle.class_group_ids,
        learned_hierdist_matrix=learned_hierdist_matrix,
    )
    val_artifacts = save_evaluation_artifacts(model, val_loader, device, bundle, run_dir, "val")
    test_artifacts = save_evaluation_artifacts(model, test_loader, device, bundle, run_dir, "test")
    result = {
        "mode": "curriculum",
        "dataset": bundle.name,
        "model": args.model,
        "cnn_width_multiplier": args.cnn_width_multiplier,
        "cifar_resnet_width_multiplier": args.cifar_resnet_width_multiplier,
        **model_size_metrics(model),
        **model_spec_metrics(model),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "curriculum_epochs": curriculum_epochs,
        "curriculum_policy": args.curriculum_policy,
        "curriculum_order": args.curriculum_order,
        "curriculum_min_clusters": args.curriculum_min_clusters,
        "curriculum_max_levels": args.curriculum_max_levels,
        "curriculum_stage_min_epochs": args.curriculum_stage_min_epochs,
        "curriculum_stage_max_epochs": args.curriculum_stage_max_epochs,
        "curriculum_stage_patience": args.curriculum_stage_patience,
        "curriculum_stage_min_delta": args.curriculum_stage_min_delta,
        "num_hierarchy_levels": len(hierarchy_levels),
        "best_val_acc": max(float(entry["val_acc"]) for entry in history),
        "best_test_acc": max(float(entry["test_acc"]) for entry in history),
        "best_test_f1_macro": max(float(entry.get("test_f1_macro", 0.0)) for entry in history),
        "final_test_acc": float(test_metrics["acc"]),
        "final_test_f1_macro": float(test_metrics["f1_macro"]),
        "final_test_precision_macro": float(test_metrics["precision_macro"]),
        "final_test_recall_macro": float(test_metrics["recall_macro"]),
        "final_test_top5_acc": float(test_metrics["top5_acc"]),
        "final_test_ece": float(test_metrics["ece"]),
        "final_test_hierdist_official": float(test_metrics.get("hierdist_official", 0.0)),
        "final_test_hier_score_official": float(test_metrics.get("hier_score_official", 0.0)),
        "final_test_same_superclass_acc_official": float(
            test_metrics.get("same_superclass_acc_official", 0.0)
        ),
        "final_test_hierdist_learned": float(test_metrics["hierdist_learned"]),
        "final_test_hier_score_learned": float(test_metrics["hier_score_learned"]),
        "distance_source": args.distance_source,
        "random_hierarchy_seed": (
            (args.random_hierarchy_seed if args.random_hierarchy_seed is not None else args.seed)
            if args.distance_source == "random_permutation"
            else None
        ),
        "reference_run_dir": str(reference_dir),
        "val_difficulty_class_accuracy_variance": float(val_artifacts["class_accuracy_variance"]),
        "test_difficulty_class_accuracy_variance": float(test_artifacts["class_accuracy_variance"]),
        "test_difficulty_confusion_entropy_mean": float(test_artifacts["confusion_entropy_mean"]),
    }
    save_json(run_dir / "results.json", result)
    save_single_row_csv(run_dir / "summary.csv", result)
    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.summary.update(result)
                log_wandb_artifacts(run_dir, f"{args.run_id}-{bundle.name}-{args.model}-curriculum")
        except ImportError:
            pass
    return result


def multiloss_run(
    args: argparse.Namespace,
    bundle: DatasetBundle,
    loaders: tuple[DataLoader, DataLoader, DataLoader],
    device: torch.device,
    logger: logging.Logger,
    run_dir: Path,
) -> dict[str, Any]:
    train_loader, val_loader, test_loader = loaders
    reference_dir, reference_model, reference_history = load_or_train_reference(
        args,
        bundle,
        loaders,
        device,
        logger,
        run_dir,
    )
    hierarchy_levels = hierarchy_from_reference(
        args,
        bundle,
        train_loader,
        val_loader,
        test_loader,
        device,
        logger,
        run_dir,
        reference_dir,
        reference_model,
    )
    memberships = build_multiloss_memberships(hierarchy_levels, bundle.num_classes, device)
    learned_hierdist_matrix = hierarchy_distance_matrix_from_levels(hierarchy_levels, bundle.num_classes)
    save_json(
        run_dir / "multiloss_config.json",
        {
            "weighting": args.multi_weighting,
            "losses": ["fine"] + [name for name, _ in memberships],
            "static_weights": args.multi_static_weights,
            "initial_weights": args.multi_initial_weights,
            "gradnorm_alpha": args.gradnorm_alpha,
            "reference_run_dir": str(reference_dir),
        },
    )

    model = build_model(
        model_name=args.model,
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
        dropout=args.dropout,
        cnn_width_multiplier=args.cnn_width_multiplier,
        cifar_resnet_width_multiplier=args.cifar_resnet_width_multiplier,
        pretrained_backbone=args.pretrained_backbone,
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
        schedule=None,
        multiloss_memberships=memberships,
        class_group_ids=bundle.class_group_ids,
        learned_hierdist_matrix=learned_hierdist_matrix,
    )
    val_artifacts = save_evaluation_artifacts(model, val_loader, device, bundle, run_dir, "val")
    test_artifacts = save_evaluation_artifacts(model, test_loader, device, bundle, run_dir, "test")
    result = {
        "mode": "multiloss",
        "dataset": bundle.name,
        "model": args.model,
        "cnn_width_multiplier": args.cnn_width_multiplier,
        "cifar_resnet_width_multiplier": args.cifar_resnet_width_multiplier,
        **model_size_metrics(model),
        **model_spec_metrics(model),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "multi_weighting": args.multi_weighting,
        "num_multiloss_levels": len(memberships),
        "best_val_acc": max(float(entry["val_acc"]) for entry in history),
        "best_test_acc": max(float(entry["test_acc"]) for entry in history),
        "best_test_f1_macro": max(float(entry.get("test_f1_macro", 0.0)) for entry in history),
        "final_test_acc": float(test_metrics["acc"]),
        "final_test_f1_macro": float(test_metrics["f1_macro"]),
        "final_test_precision_macro": float(test_metrics["precision_macro"]),
        "final_test_recall_macro": float(test_metrics["recall_macro"]),
        "final_test_top5_acc": float(test_metrics["top5_acc"]),
        "final_test_ece": float(test_metrics["ece"]),
        "final_test_hierdist_official": float(test_metrics.get("hierdist_official", 0.0)),
        "final_test_hier_score_official": float(test_metrics.get("hier_score_official", 0.0)),
        "final_test_same_superclass_acc_official": float(
            test_metrics.get("same_superclass_acc_official", 0.0)
        ),
        "final_test_hierdist_learned": float(test_metrics["hierdist_learned"]),
        "final_test_hier_score_learned": float(test_metrics["hier_score_learned"]),
        "distance_source": args.distance_source,
        "reference_run_dir": str(reference_dir),
        "reference_best_val_acc": max(float(entry["val_acc"]) for entry in reference_history),
        "val_difficulty_class_accuracy_variance": float(val_artifacts["class_accuracy_variance"]),
        "test_difficulty_class_accuracy_variance": float(test_artifacts["class_accuracy_variance"]),
        "test_difficulty_confusion_entropy_mean": float(test_artifacts["confusion_entropy_mean"]),
    }
    save_json(run_dir / "results.json", result)
    save_single_row_csv(run_dir / "summary.csv", result)
    if args.wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.run.summary.update(result)
                log_wandb_artifacts(run_dir, f"{args.run_id}-{bundle.name}-{args.model}-multiloss")
        except ImportError:
            pass
    return result


def main() -> None:
    args = resolve_defaults(parse_args())
    reproducibility = seed_everything(args.seed, deterministic=args.deterministic)
    args.reproducibility = reproducibility
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    run_dir = output_dir / args.run_id / f"{args.dataset}_{args.model}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(run_dir, f"training_log_{args.mode}.txt")
    save_json(run_dir / "config.json", vars(args))
    wandb_run = setup_wandb(args, run_dir)

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
        "num_classes=%d | batch_size=%d | epochs=%d | optimizer=%s | scheduler=%s | lr=%.6f | augmentation=%s",
        bundle.num_classes,
        args.batch_size,
        args.epochs,
        args.optimizer,
        args.scheduler,
        args.lr,
        args.augmentation,
    )
    logger.info("Reproducibility: %s", json.dumps(reproducibility, sort_keys=True))

    if args.mode == "baseline":
        result = baseline_run(args, bundle, loaders, device, logger, run_dir)
    elif args.mode == "curriculum":
        result = curriculum_run(args, bundle, loaders, device, logger, run_dir)
    else:
        result = multiloss_run(args, bundle, loaders, device, logger, run_dir)

    logger.info("Run complete: %s", json.dumps(result, indent=2))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
