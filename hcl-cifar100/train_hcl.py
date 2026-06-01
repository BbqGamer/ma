"""
Paper-inspired HCL adaptation for CIFAR-100.

Important: the original paper evaluates on hierarchical multi-label datasets with
pre-extracted features and an MLP. CIFAR-100 is a different setting, so this file
implements the closest clean adaptation for controlled ablations:

- baseline: fine-label cross entropy
- hier: hierarchical max loss max(CE_fine, CE_coarse)
- hcl: class-curriculum over the hierarchical max loss

To avoid confounding variables:
- same shared ResNet18 backbone in all modes
- same optimizer, scheduler, data, augmentations, seed, and evaluation
- same dual-head architecture in all modes
- only the loss changes across modes
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
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18


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


class CIFAR100WithHierarchy(Dataset):
    def __init__(self, base_dataset: torchvision.datasets.CIFAR100) -> None:
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, fine_label = self.base_dataset[idx]
        coarse_label = int(FINE_TO_COARSE[fine_label])
        return (
            image,
            torch.tensor(fine_label, dtype=torch.long),
            torch.tensor(coarse_label, dtype=torch.long),
        )


class DualHeadResNet18(nn.Module):
    """Shared CIFAR-style ResNet18 backbone with coarse and fine heads."""

    def __init__(self, num_coarse: int, num_fine: int) -> None:
        super().__init__()
        base_model = resnet18(weights=None)
        base_model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        base_model.maxpool = nn.Identity()

        self.features = nn.Sequential(*list(base_model.children())[:-1])
        num_ftrs = base_model.fc.in_features
        self.fc_coarse = nn.Linear(num_ftrs, num_coarse)
        self.fc_fine = nn.Linear(num_ftrs, num_fine)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc_coarse(x), self.fc_fine(x)


class HierarchicalCurriculumHelper:
    """Helper for the CIFAR adaptation of HCL.

    For each sample i:
      l_fine(i)   = CE(fine_logits_i, fine_target_i)
      l_coarse(i) = CE(coarse_logits_i, coarse_target_i)
      l_h(i)      = max(l_fine(i), l_coarse(i))

    Curriculum is defined over fine classes by aggregating l_h over the training set.
    """

    def __init__(self, num_fine: int, select_frac: float) -> None:
        self.num_fine = num_fine
        self.select_frac = float(max(0.05, min(1.0, select_frac)))

    def per_sample_losses(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        coarse_targets: torch.Tensor,
        fine_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_coarse = F.cross_entropy(coarse_logits, coarse_targets, reduction="none")
        loss_fine = F.cross_entropy(fine_logits, fine_targets, reduction="none")
        loss_hier = torch.maximum(loss_fine, loss_coarse)
        return loss_coarse, loss_fine, loss_hier

    @torch.no_grad()
    def select_classes(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
    ) -> torch.Tensor:
        model.eval()
        class_loss_sums = torch.zeros(self.num_fine, device=device)
        class_counts = torch.zeros(self.num_fine, device=device)

        for inputs, fine_targets, coarse_targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            fine_targets = fine_targets.to(device, non_blocking=True)
            coarse_targets = coarse_targets.to(device, non_blocking=True)

            coarse_logits, fine_logits = model(inputs)
            _, _, loss_hier = self.per_sample_losses(
                coarse_logits, fine_logits, coarse_targets, fine_targets
            )
            class_loss_sums.scatter_add_(0, fine_targets, loss_hier)
            class_counts.scatter_add_(0, fine_targets, torch.ones_like(loss_hier))

        class_loss_means = class_loss_sums / class_counts.clamp_min(1.0)
        k = max(1, int(round(self.select_frac * self.num_fine)))
        easiest = torch.argsort(class_loss_means, descending=False)[:k]

        class_mask = torch.zeros(self.num_fine, device=device)
        class_mask[easiest] = 1.0
        return class_mask


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
    loader: DataLoader,
    helper: HierarchicalCurriculumHelper,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    val_loss = 0.0
    total_correct_fine = 0
    total_correct_coarse = 0
    total_hier_dist = 0
    total_samples = 0

    for inputs, fine_targets, coarse_targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        fine_targets = fine_targets.to(device, non_blocking=True)
        coarse_targets = coarse_targets.to(device, non_blocking=True)

        coarse_logits, fine_logits = model(inputs)
        batch_loss = F.cross_entropy(fine_logits, fine_targets)
        val_loss += batch_loss.item()

        fine_preds = fine_logits.argmax(dim=1)
        coarse_preds = coarse_logits.argmax(dim=1)

        total_correct_fine += (fine_preds == fine_targets).sum().item()
        total_correct_coarse += (coarse_preds == coarse_targets).sum().item()

        pred_coarse_from_fine = torch.from_numpy(FINE_TO_COARSE[fine_preds.cpu().numpy()]).to(device)
        hier_dist = torch.zeros_like(fine_targets)
        hier_dist[fine_preds != fine_targets] = 1
        hier_dist[pred_coarse_from_fine != coarse_targets] = 2
        total_hier_dist += hier_dist.sum().item()
        total_samples += fine_targets.size(0)

    return (
        val_loss / len(loader),
        total_correct_fine / total_samples,
        total_hier_dist / total_samples,
        total_correct_coarse / total_samples,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    helper: HierarchicalCurriculumHelper,
    device: torch.device,
    mode: str,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool,
    class_mask: torch.Tensor | None,
) -> float:
    model.train()
    running_loss = 0.0
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    for inputs, fine_targets, coarse_targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        fine_targets = fine_targets.to(device, non_blocking=True)
        coarse_targets = coarse_targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx():
            coarse_logits, fine_logits = model(inputs)
            loss_coarse, loss_fine, loss_hier = helper.per_sample_losses(
                coarse_logits, fine_logits, coarse_targets, fine_targets
            )

            if mode == "baseline":
                loss = loss_fine.mean()
            elif mode == "hier":
                loss = loss_hier.mean()
            else:
                if class_mask is None:
                    loss = loss_hier.mean()
                else:
                    sample_mask = class_mask[fine_targets]
                    selected = sample_mask > 0
                    if torch.any(selected):
                        loss = loss_hier[selected].mean()
                    else:
                        loss = loss_hier.mean()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


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
    parser.add_argument("--mode", type=str, choices=["baseline", "hier", "hcl"], required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--select_frac",
        type=float,
        default=0.8,
        help="Fraction of easiest fine classes kept active for HCL.",
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
    trainset = CIFAR100WithHierarchy(
        torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            download=args.download,
            transform=transform_train,
        )
    )
    curriculumset = CIFAR100WithHierarchy(
        torchvision.datasets.CIFAR100(
            root=args.data_dir,
            train=True,
            download=args.download,
            transform=transform_test,
        )
    )
    testset = CIFAR100WithHierarchy(
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

    model = DualHeadResNet18(
        num_coarse=HIERARCHY.num_coarse,
        num_fine=HIERARCHY.num_fine,
    ).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp := args.amp and device.type == "cuda"))

    helper = HierarchicalCurriculumHelper(
        num_fine=HIERARCHY.num_fine,
        select_frac=args.select_frac,
    )
    logger.info(f"select_frac={args.select_frac:.2f}")

    best_acc = 0.0
    best_path = output_dir / f"best_{args.mode}.pt"
    last_path = output_dir / f"last_{args.mode}.pt"

    for epoch in range(args.epochs):
        class_mask = None
        if args.mode == "hcl":
            class_mask = helper.select_classes(model, curriculumloader, device)
            logger.info(
                f"Selected {int(class_mask.sum().item())}/{HIERARCHY.num_fine} fine classes"
            )

        train_loss = train_one_epoch(
            model=model,
            loader=trainloader,
            optimizer=optimizer,
            helper=helper,
            device=device,
            mode=args.mode,
            scaler=scaler,
            use_amp=use_amp,
            class_mask=class_mask,
        )
        val_loss, val_acc, val_hier_dist, val_coarse_acc = evaluate(
            model=model,
            loader=testloader,
            helper=helper,
            device=device,
        )
        scheduler.step()

        logger.info(
            f"Epoch [{epoch + 1}/{args.epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Fine-CE: {val_loss:.4f} | "
            f"Fine Hit@1: {val_acc * 100:.2f}% | "
            f"Coarse Hit@1: {val_coarse_acc * 100:.2f}% | "
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

    logger.info(f"Training complete. Best Fine Hit@1: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
