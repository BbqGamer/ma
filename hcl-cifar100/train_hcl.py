"""
Hierarchical Class-Based Curriculum Loss (HCL) for CIFAR-100.

This script trains the same ResNet18 backbone in two modes:
- baseline: standard multi-label BCE over all hierarchy nodes
- hcl: hierarchical constrained loss + class-based curriculum selection

The implementation follows the paper more closely by modeling CIFAR-100 as a
small hierarchy of 20 coarse nodes + 100 fine nodes (120 total outputs).
"""

from __future__ import annotations

import argparse
import logging
import random
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


# CIFAR-100 fine label -> coarse superclass mapping.
FINE_TO_COARSE = np.array(
    [
        4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
        6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
        5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
        10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
        16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
    ],
    dtype=np.int64,
)


@dataclass(frozen=True)
class HierarchySpec:
    num_coarse: int = 20
    num_fine: int = 100

    @property
    def num_nodes(self) -> int:
        return self.num_coarse + self.num_fine

    @property
    def fine_offset(self) -> int:
        return self.num_coarse


HIERARCHY = HierarchySpec()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(mode: str, output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"HCL_{mode}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(output_dir / f"training_log_{mode}.txt")
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


class HierarchicalCIFAR100(Dataset):
    """Wrap CIFAR-100 and emit multi-hot targets for coarse + fine labels."""

    def __init__(self, base_dataset: torchvision.datasets.CIFAR100) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image, fine_label = self.base_dataset[idx]
        coarse_label = int(FINE_TO_COARSE[fine_label])

        target = torch.zeros(HIERARCHY.num_nodes, dtype=torch.float32)
        target[coarse_label] = 1.0
        target[HIERARCHY.fine_offset + fine_label] = 1.0

        return (
            image,
            target,
            torch.tensor(fine_label, dtype=torch.long),
            torch.tensor(coarse_label, dtype=torch.long),
        )


class HierarchicalResNet18(nn.Module):
    """ResNet18 with a CIFAR stem and one logit per hierarchy node."""

    def __init__(self, num_outputs: int) -> None:
        super().__init__()
        base_model = resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        base_model.maxpool = nn.Identity()
        base_model.fc = nn.Linear(base_model.fc.in_features, num_outputs)
        self.model = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class HierarchicalCurriculumLoss(nn.Module):
    """Exact paper-style hierarchical constrained loss + curriculum selection."""

    def __init__(self, hierarchy: HierarchySpec, thresh: float = 50.0) -> None:
        super().__init__()
        self.hierarchy = hierarchy
        self.thresh = thresh
        self.base_loss = nn.BCEWithLogitsLoss(reduction="none")

        parent = torch.full((hierarchy.num_nodes,), -1, dtype=torch.long)
        # Fine nodes are children of the corresponding coarse node.
        for fine_idx in range(hierarchy.num_fine):
            node_idx = hierarchy.fine_offset + fine_idx
            parent[node_idx] = int(FINE_TO_COARSE[fine_idx])
        self.register_buffer("parent", parent)

    def base_node_losses(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.base_loss(logits, targets)

    def hierarchical_node_losses(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        base = self.base_node_losses(logits, targets)
        h_loss = base.clone()
        fine_start = self.hierarchy.fine_offset

        # For fine nodes, enforce the hierarchical constraint: loss(child) >= loss(parent)
        for node_idx in range(fine_start, self.hierarchy.num_nodes):
            parent_idx = int(self.parent[node_idx].item())
            h_loss[:, node_idx] = torch.maximum(base[:, node_idx], base[:, parent_idx])

        return h_loss

    @torch.no_grad()
    def select_classes(
        self,
        model: nn.Module,
        trainloader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        """Select curriculum classes using the current model, as in Algorithm 1.

        We aggregate per-class loss over the whole curriculum set and then normalize
        by the number of examples. On CIFAR-100 this keeps the paper threshold on a
        sensible scale and avoids collapsing to a single selected class.
        """
        model.eval()
        class_loss_sums = torch.zeros(self.hierarchy.num_nodes, device=device)
        total_examples = 0

        for inputs, targets, _, _ in trainloader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(inputs)
            h_loss = self.hierarchical_node_losses(logits, targets)
            class_loss_sums += h_loss.sum(dim=0)
            total_examples += targets.size(0)

        class_losses = class_loss_sums / max(total_examples, 1)
        sorted_losses, sorted_indices = torch.sort(class_losses, descending=False)
        cumulative_loss = torch.cumsum(sorted_losses, dim=0)
        thresholds = self.thresh + 1 - torch.arange(
            1,
            self.hierarchy.num_nodes + 1,
            device=device,
            dtype=sorted_losses.dtype,
        )
        cutoff = torch.nonzero(cumulative_loss > thresholds, as_tuple=False)
        k = int(cutoff[0].item()) + 1 if cutoff.numel() else self.hierarchy.num_nodes

        class_mask = torch.zeros(self.hierarchy.num_nodes, device=device)
        class_mask[sorted_indices[:k]] = 1.0
        return class_mask

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the baseline or HCL objective for a mini-batch."""
        if class_mask is None:
            return self.base_node_losses(logits, targets).mean()

        h_loss = self.hierarchical_node_losses(logits, targets)
        selected = class_mask.bool()
        if not torch.any(selected):
            return h_loss.mean()
        return h_loss[:, selected].mean()


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5071, 0.4867, 0.4408),
                (0.2675, 0.2565, 0.2761),
            ),
        ]
    )
    return transform_train, transform_test


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    testloader: DataLoader,
    criterion: HierarchicalCurriculumLoss,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    val_loss = 0.0
    total_correct = 0
    total_hier_dist = 0
    total_samples = 0

    for inputs, targets, fine_targets, coarse_targets in testloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        fine_targets = fine_targets.to(device, non_blocking=True)
        coarse_targets = coarse_targets.to(device, non_blocking=True)

        logits = model(inputs)
        batch_loss = criterion(logits, targets)
        val_loss += batch_loss.item()

        fine_logits = logits[:, HIERARCHY.fine_offset :]
        fine_preds = fine_logits.argmax(dim=1)
        coarse_preds = torch.from_numpy(FINE_TO_COARSE[fine_preds.cpu().numpy()]).to(device)

        total_correct += (fine_preds == fine_targets).sum().item()
        hier_dist = torch.zeros_like(fine_targets)
        hier_dist[fine_preds != fine_targets] = 1
        hier_dist[coarse_preds != coarse_targets] = 2
        total_hier_dist += hier_dist.sum().item()
        total_samples += fine_targets.size(0)

    return (
        val_loss / len(testloader),
        total_correct / total_samples,
        total_hier_dist / total_samples,
    )


def train_one_epoch(
    model: nn.Module,
    trainloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: HierarchicalCurriculumLoss,
    device: torch.device,
    mode: str,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    class_mask: torch.Tensor | None,
) -> float:
    model.train()
    running_loss = 0.0
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    for inputs, targets, _, _ in trainloader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            logits = model(inputs)
            if mode == "baseline":
                loss = criterion(logits, targets, class_mask=None)
            else:
                loss = criterion(logits, targets, class_mask=class_mask)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)


def save_checkpoint(
    path: Path,
    epoch: int,
    mode: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    best_acc: float,
    args: dict,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "mode": mode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc": best_acc,
            "args": args,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["baseline", "hcl"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--thresh",
        type=float,
        default=50.0,
        help="Threshold for class selection in HCL.",
    )
    parser.add_argument("--data_dir", type=str, default="/workspace/data")
    parser.add_argument("--output_dir", type=str, default="/workspace/runs")
    parser.add_argument(
        "--run_id",
        type=str,
        default="run",
        help="Run-level directory name used to avoid overwriting previous runs.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-100 if it is missing in data_dir.",
    )
    parser.add_argument(
        "--no-download",
        action="store_false",
        dest="download",
        help="Disable CIFAR-100 download.",
    )
    parser.set_defaults(download=True)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable mixed precision training on CUDA.",
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir) / args.run_id / f"cifar100_{args.mode}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.mode, output_dir)
    logger.info(
        f"Initialized on {device}. Mode: {args.mode.upper()} | Run ID: {args.run_id}"
    )

    transform_train, transform_test = build_transforms()
    trainset = HierarchicalCIFAR100(
        torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            download=args.download,
            transform=transform_train,
        )
    )
    curriculumset = HierarchicalCIFAR100(
        torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            download=args.download,
            transform=transform_test,
        )
    )
    testset = HierarchicalCIFAR100(
        torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=False,
            download=args.download,
            transform=transform_test,
        )
    )

    pin_memory = device.type == "cuda"
    persistent_workers = args.num_workers > 0
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4

    trainloader = DataLoader(trainset, shuffle=True, **loader_kwargs)
    curriculumloader = DataLoader(curriculumset, shuffle=False, **loader_kwargs)
    testloader = DataLoader(testset, shuffle=False, **loader_kwargs)

    model = HierarchicalResNet18(num_outputs=HIERARCHY.num_nodes).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    criterion = HierarchicalCurriculumLoss(HIERARCHY, thresh=args.thresh).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp := args.amp and device.type == "cuda"))

    best_acc = 0.0
    best_path = output_dir / f"best_{args.mode}.pt"
    last_path = output_dir / f"last_{args.mode}.pt"

    for epoch in range(args.epochs):
        class_mask = None
        if args.mode == "hcl":
            class_mask = criterion.select_classes(model, curriculumloader, device)
            logger.info(f"Selected {int(class_mask.sum().item())}/{HIERARCHY.num_nodes} classes")

        train_loss = train_one_epoch(
            model=model,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            mode=args.mode,
            scaler=scaler,
            use_amp=use_amp,
            class_mask=class_mask,
        )
        val_loss, val_acc, val_hier_dist = evaluate(
            model=model,
            testloader=testloader,
            criterion=criterion,
            device=device,
        )
        scheduler.step()

        logger.info(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Hit@1: {val_acc * 100:.2f}% | "
            f"HierDist: {val_hier_dist:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint(
            last_path,
            epoch=epoch + 1,
            mode=args.mode,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_acc=best_acc,
            args=vars(args),
        )
        if is_best:
            save_checkpoint(
                best_path,
                epoch=epoch + 1,
                mode=args.mode,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                best_acc=best_acc,
                args=vars(args),
            )

    logger.info(f"Training complete. Best Hit@1: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
