"""Simple PyTorch Lightning pipeline for CIFAR-100 easy-vs-hard weighting."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

from lightning.pytorch import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import CIFAR100

from ma_thesis.cifar100_schedule import (
    TaskScheduleContext,
    TaskScheduleInitContext,
    TaskSchedulePolicy,
    normalize_weights,
)
from ma_thesis.config import DATA_DIR, REPORTS_DIR

CIFAR100_ROOT = DATA_DIR / "raw" / "cifar100"
CIFAR100_REPORTS = REPORTS_DIR / "cifar100_easy_hard"

COARSE_NAMES = [
    "aquatic mammals",
    "fish",
    "flowers",
    "food containers",
    "fruit and vegetables",
    "household electrical devices",
    "household furniture",
    "insects",
    "large carnivores",
    "large man-made outdoor things",
    "large natural outdoor scenes",
    "large omnivores and herbivores",
    "medium-sized mammals",
    "non-insect invertebrates",
    "people",
    "reptiles",
    "small mammals",
    "trees",
    "vehicles 1",
    "vehicles 2",
]

FINE_TO_COARSE_NAME = {
    "apple": "fruit and vegetables",
    "aquarium_fish": "fish",
    "baby": "people",
    "bear": "large carnivores",
    "beaver": "aquatic mammals",
    "bed": "household furniture",
    "bee": "insects",
    "beetle": "insects",
    "bicycle": "vehicles 1",
    "bottle": "food containers",
    "bowl": "food containers",
    "boy": "people",
    "bridge": "large man-made outdoor things",
    "bus": "vehicles 1",
    "butterfly": "insects",
    "camel": "large omnivores and herbivores",
    "can": "food containers",
    "castle": "large man-made outdoor things",
    "caterpillar": "insects",
    "cattle": "large omnivores and herbivores",
    "chair": "household furniture",
    "chimpanzee": "large omnivores and herbivores",
    "clock": "household electrical devices",
    "cloud": "large natural outdoor scenes",
    "cockroach": "insects",
    "couch": "household furniture",
    "crab": "non-insect invertebrates",
    "crocodile": "reptiles",
    "cup": "food containers",
    "dinosaur": "reptiles",
    "dolphin": "aquatic mammals",
    "elephant": "large omnivores and herbivores",
    "flatfish": "fish",
    "forest": "large natural outdoor scenes",
    "fox": "medium-sized mammals",
    "girl": "people",
    "hamster": "small mammals",
    "house": "large man-made outdoor things",
    "kangaroo": "large omnivores and herbivores",
    "keyboard": "household electrical devices",
    "lamp": "household electrical devices",
    "lawn_mower": "vehicles 2",
    "leopard": "large carnivores",
    "lion": "large carnivores",
    "lizard": "reptiles",
    "lobster": "non-insect invertebrates",
    "man": "people",
    "maple_tree": "trees",
    "motorcycle": "vehicles 1",
    "mountain": "large natural outdoor scenes",
    "mouse": "small mammals",
    "mushroom": "fruit and vegetables",
    "oak_tree": "trees",
    "orange": "fruit and vegetables",
    "orchid": "flowers",
    "otter": "aquatic mammals",
    "palm_tree": "trees",
    "pear": "fruit and vegetables",
    "pickup_truck": "vehicles 1",
    "pine_tree": "trees",
    "plain": "large natural outdoor scenes",
    "plate": "food containers",
    "poppy": "flowers",
    "porcupine": "medium-sized mammals",
    "possum": "medium-sized mammals",
    "rabbit": "small mammals",
    "raccoon": "medium-sized mammals",
    "ray": "fish",
    "road": "large man-made outdoor things",
    "rocket": "vehicles 2",
    "rose": "flowers",
    "sea": "large natural outdoor scenes",
    "seal": "aquatic mammals",
    "shark": "fish",
    "shrew": "small mammals",
    "skunk": "medium-sized mammals",
    "skyscraper": "large man-made outdoor things",
    "snail": "non-insect invertebrates",
    "snake": "reptiles",
    "spider": "non-insect invertebrates",
    "squirrel": "small mammals",
    "streetcar": "vehicles 2",
    "sunflower": "flowers",
    "sweet_pepper": "fruit and vegetables",
    "table": "household furniture",
    "tank": "vehicles 2",
    "telephone": "household electrical devices",
    "television": "household electrical devices",
    "tiger": "large carnivores",
    "tractor": "vehicles 2",
    "train": "vehicles 1",
    "trout": "fish",
    "tulip": "flowers",
    "turtle": "reptiles",
    "wardrobe": "household furniture",
    "whale": "aquatic mammals",
    "willow_tree": "trees",
    "wolf": "large carnivores",
    "woman": "people",
    "worm": "non-insect invertebrates",
}
COARSE_TO_INDEX = {name: idx for idx, name in enumerate(COARSE_NAMES)}


@dataclass
class TaskTrainState:
    current_train_losses: tuple[float, float] | None = None
    current_val_losses: tuple[float, float] | None = None
    prev_train_losses: tuple[float, float] | None = None
    prev_val_losses: tuple[float, float] | None = None
    ema_train_losses: tuple[float, float] | None = None
    ema_val_losses: tuple[float, float] | None = None
    best_val_losses: tuple[float, float] | None = None
    prev_weights: tuple[float, float] | None = None
    best_hard_val_loss: float = float("inf")


class CIFAR100MultiTaskSubset(Dataset[tuple[torch.Tensor, int, int]]):
    def __init__(self, base: CIFAR100, indices: list[int], fine_to_coarse: list[int]) -> None:
        self.base = base
        self.indices = indices
        self.fine_to_coarse = fine_to_coarse

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, int]:
        image, fine = self.base[self.indices[idx]]
        coarse = self.fine_to_coarse[int(fine)]
        return image, int(fine), int(coarse)


class CIFAR100EasyHardDataModule(LightningDataModule):
    def __init__(
        self,
        *,
        data_dir: Path = CIFAR100_ROOT,
        batch_size: int = 256,
        num_workers: int = 8,
        val_fraction: float = 0.1,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_fraction = val_fraction
        self.seed = seed
        self.mean = (0.5071, 0.4867, 0.4408)
        self.std = (0.2675, 0.2565, 0.2761)
        self.train_set: Dataset[tuple[torch.Tensor, int, int]] | None = None
        self.val_set: Dataset[tuple[torch.Tensor, int, int]] | None = None
        self.test_set: Dataset[tuple[torch.Tensor, int, int]] | None = None
        self.fine_to_coarse: list[int] | None = None

    def prepare_data(self) -> None:
        CIFAR100(root=self.data_dir, train=True, download=True)
        CIFAR100(root=self.data_dir, train=False, download=True)

    def setup(self, stage: str | None = None) -> None:
        train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        eval_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        full_train = CIFAR100(root=self.data_dir, train=True, download=False, transform=train_transform)
        full_train_eval = CIFAR100(
            root=self.data_dir,
            train=True,
            download=False,
            transform=eval_transform,
        )
        test_base = CIFAR100(root=self.data_dir, train=False, download=False, transform=eval_transform)

        fine_to_coarse = [COARSE_TO_INDEX[FINE_TO_COARSE_NAME[name]] for name in full_train.classes]
        self.fine_to_coarse = fine_to_coarse

        n_total = len(full_train)
        indices = list(range(n_total))
        rng = random.Random(self.seed)
        rng.shuffle(indices)
        n_val = int(n_total * self.val_fraction)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.train_set = CIFAR100MultiTaskSubset(full_train, train_indices, fine_to_coarse)
        self.val_set = CIFAR100MultiTaskSubset(full_train_eval, val_indices, fine_to_coarse)
        self.test_set = CIFAR100MultiTaskSubset(test_base, list(range(len(test_base))), fine_to_coarse)

    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, int, int]]:
        assert self.train_set is not None
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, int, int]]:
        assert self.val_set is not None
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, int, int]]:
        assert self.test_set is not None
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


def _ema_update(prev_ema: tuple[float, float] | None, current: tuple[float, float], *, alpha: float) -> tuple[float, float]:
    if prev_ema is None:
        return current
    return tuple(alpha * cur + (1.0 - alpha) * prev for prev, cur in zip(prev_ema, current))


def _min_update(prev_best: tuple[float, float] | None, current: tuple[float, float]) -> tuple[float, float]:
    if prev_best is None:
        return current
    return tuple(min(prev, cur) for prev, cur in zip(prev_best, current))


class LinearHardeningPolicy:
    def reset(self, ctx: TaskScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: TaskScheduleContext) -> tuple[float, float]:
        t = ctx.epoch / max(1, ctx.total_epochs - 1)
        hard = min(0.95, 0.15 + 0.8 * t)
        return (1.0 - hard, hard)


class ValGapPolicy:
    def reset(self, ctx: TaskScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: TaskScheduleContext) -> tuple[float, float]:
        t = ctx.epoch / max(1, ctx.total_epochs - 1)
        hard = 0.3 + 0.5 * t
        if ctx.current_val_losses and ctx.best_val_losses:
            hard_cur = ctx.current_val_losses[ctx.hard_index]
            hard_best = ctx.best_val_losses[ctx.hard_index]
            gap = max(0.0, (hard_cur - hard_best) / (abs(hard_best) + 1e-8))
            hard += min(0.25, 2.0 * gap)
        hard = min(0.98, max(0.05, hard))
        return (1.0 - hard, hard)


class ParametricEasyHardPolicy:
    def __init__(self, *, tau: float, a: float, b: float, c: float, d: float) -> None:
        self.tau = tau
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def reset(self, ctx: TaskScheduleInitContext) -> None:
        return None

    def get_weights(self, ctx: TaskScheduleContext) -> tuple[float, float]:
        t = ctx.epoch / max(1, ctx.total_epochs - 1)
        hard_gap = 0.0
        hard_trend = 0.0
        if ctx.current_val_losses and ctx.best_val_losses:
            hard_cur = ctx.current_val_losses[ctx.hard_index]
            hard_best = ctx.best_val_losses[ctx.hard_index]
            hard_gap = max(0.0, (hard_cur - hard_best) / (abs(hard_best) + 1e-8))
        if ctx.current_val_losses and ctx.ema_val_losses:
            hard_cur = ctx.current_val_losses[ctx.hard_index]
            hard_ema = ctx.ema_val_losses[ctx.hard_index]
            hard_trend = (hard_cur - hard_ema) / (abs(hard_ema) + 1e-8)
        logit = (self.a + self.b * t + self.c * hard_gap + self.d * hard_trend) / max(1e-4, self.tau)
        hard = 1.0 / (1.0 + np.exp(-logit))
        hard = float(min(0.995, max(0.005, hard)))
        return (1.0 - hard, hard)


class CIFAR100EasyHardModule(LightningModule):
    def __init__(
        self,
        *,
        schedule_policy: TaskSchedulePolicy,
        lr: float,
        weight_decay: float,
        max_epochs: int,
        ema_alpha: float,
        history_window: int,
        run_name: str,
        seed: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["schedule_policy"])
        self.schedule_policy = schedule_policy
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.ema_alpha = ema_alpha
        self.history_window = history_window
        self.run_name = run_name
        self.seed = seed

        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.easy_head = nn.Linear(features, 20)
        self.hard_head = nn.Linear(features, 100)
        self.loss_fn = nn.CrossEntropyLoss()

        self.task_names = ("easy_coarse", "hard_fine")
        self.state = TaskTrainState()
        self.recent_train_losses: deque[tuple[float, float]] = deque(maxlen=history_window)
        self.recent_val_losses: deque[tuple[float, float]] = deque(maxlen=history_window)
        self.history: list[dict[str, float]] = []
        self.current_weights = (0.5, 0.5)
        self.best_hard_val_loss = float("inf")
        self._train_loss_sum = torch.zeros(2)
        self._train_batches = 0
        self._train_epoch_losses: tuple[float, float] | None = None
        self._val_rows: list[dict[str, float]] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.easy_head(feat), self.hard_head(feat)

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_fit_start(self) -> None:
        init_ctx = TaskScheduleInitContext(
            task_names=self.task_names,
            easy_index=0,
            hard_index=1,
            total_epochs=self.max_epochs,
            history_window=self.history_window,
            seed=self.seed,
            run_name=self.run_name,
        )
        self.schedule_policy.reset(init_ctx)

    def _build_context(self) -> TaskScheduleContext:
        return TaskScheduleContext(
            epoch=int(self.current_epoch),
            total_epochs=self.max_epochs,
            task_names=self.task_names,
            easy_index=0,
            hard_index=1,
            current_train_losses=self.state.current_train_losses,
            current_val_losses=self.state.current_val_losses,
            prev_train_losses=self.state.prev_train_losses,
            prev_val_losses=self.state.prev_val_losses,
            ema_train_losses=self.state.ema_train_losses,
            ema_val_losses=self.state.ema_val_losses,
            best_val_losses=self.state.best_val_losses,
            prev_weights=self.state.prev_weights,
            best_hard_val_loss=self.state.best_hard_val_loss,
            recent_train_losses=tuple(self.recent_train_losses),
            recent_val_losses=tuple(self.recent_val_losses),
        )

    def on_train_epoch_start(self) -> None:
        ctx = self._build_context()
        self.current_weights = normalize_weights(self.schedule_policy.get_weights(ctx), n_tasks=2)
        self._train_loss_sum = torch.zeros(2, device=self.device)
        self._train_batches = 0

    def training_step(self, batch: tuple[torch.Tensor, int, int], batch_idx: int) -> torch.Tensor:
        x, fine, coarse = batch
        easy_logits, hard_logits = self(x)
        easy_loss = self.loss_fn(easy_logits, coarse)
        hard_loss = self.loss_fn(hard_logits, fine)
        weights = torch.tensor(self.current_weights, dtype=torch.float32, device=self.device)
        loss = weights[0] * easy_loss + weights[1] * hard_loss
        self._train_loss_sum += torch.stack([easy_loss.detach(), hard_loss.detach()])
        self._train_batches += 1
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        return loss

    def on_train_epoch_end(self) -> None:
        denom = max(1, self._train_batches)
        vals = (self._train_loss_sum / denom).detach().cpu().tolist()
        self._train_epoch_losses = (float(vals[0]), float(vals[1]))

    def on_validation_epoch_start(self) -> None:
        self._val_rows = []

    def validation_step(self, batch: tuple[torch.Tensor, int, int], batch_idx: int) -> None:
        x, fine, coarse = batch
        easy_logits, hard_logits = self(x)
        easy_loss = self.loss_fn(easy_logits, coarse)
        hard_loss = self.loss_fn(hard_logits, fine)
        easy_acc = (easy_logits.argmax(dim=1) == coarse).float().mean()
        hard_acc = (hard_logits.argmax(dim=1) == fine).float().mean()
        self._val_rows.append(
            {
                "easy_loss": float(easy_loss.detach().cpu()),
                "hard_loss": float(hard_loss.detach().cpu()),
                "easy_acc": float(easy_acc.detach().cpu()),
                "hard_acc": float(hard_acc.detach().cpu()),
            }
        )

    def on_validation_epoch_end(self) -> None:
        if not self._val_rows:
            return
        easy_val = float(np.mean([row["easy_loss"] for row in self._val_rows]))
        hard_val = float(np.mean([row["hard_loss"] for row in self._val_rows]))
        easy_acc = float(np.mean([row["easy_acc"] for row in self._val_rows]))
        hard_acc = float(np.mean([row["hard_acc"] for row in self._val_rows]))
        train_losses = self._train_epoch_losses or (float("nan"), float("nan"))
        val_losses = (easy_val, hard_val)

        self.state.prev_train_losses = self.state.current_train_losses
        self.state.prev_val_losses = self.state.current_val_losses
        self.state.current_train_losses = train_losses
        self.state.current_val_losses = val_losses
        self.state.ema_train_losses = _ema_update(self.state.ema_train_losses, train_losses, alpha=self.ema_alpha)
        self.state.ema_val_losses = _ema_update(self.state.ema_val_losses, val_losses, alpha=self.ema_alpha)
        self.state.best_val_losses = _min_update(self.state.best_val_losses, val_losses)
        self.state.prev_weights = self.current_weights
        self.state.best_hard_val_loss = min(self.state.best_hard_val_loss, hard_val)
        self.best_hard_val_loss = self.state.best_hard_val_loss
        self.recent_train_losses.append(train_losses)
        self.recent_val_losses.append(val_losses)

        row = {
            "epoch": float(self.current_epoch),
            "weight_easy": float(self.current_weights[0]),
            "weight_hard": float(self.current_weights[1]),
            "train_easy_loss": float(train_losses[0]),
            "train_hard_loss": float(train_losses[1]),
            "val_easy_loss": float(easy_val),
            "val_hard_loss": float(hard_val),
            "best_val_hard_loss": float(self.state.best_hard_val_loss),
            "val_easy_acc": float(easy_acc),
            "val_hard_acc": float(hard_acc),
        }
        self.history.append(row)

        self.log("val_easy_loss", easy_val, prog_bar=False, on_epoch=True)
        self.log("val_hard_loss", hard_val, prog_bar=True, on_epoch=True)
        self.log("best_val_hard_loss", self.state.best_hard_val_loss, prog_bar=True, on_epoch=True)
        self.log("val_easy_acc", easy_acc, prog_bar=False, on_epoch=True)
        self.log("val_hard_acc", hard_acc, prog_bar=True, on_epoch=True)
        self.log("weight_easy", float(self.current_weights[0]), prog_bar=False, on_epoch=True)
        self.log("weight_hard", float(self.current_weights[1]), prog_bar=False, on_epoch=True)

    def test_step(self, batch: tuple[torch.Tensor, int, int], batch_idx: int) -> dict[str, torch.Tensor]:
        x, fine, coarse = batch
        easy_logits, hard_logits = self(x)
        easy_loss = self.loss_fn(easy_logits, coarse)
        hard_loss = self.loss_fn(hard_logits, fine)
        easy_acc = (easy_logits.argmax(dim=1) == coarse).float().mean()
        hard_acc = (hard_logits.argmax(dim=1) == fine).float().mean()
        self.log("test_easy_loss", easy_loss, on_epoch=True)
        self.log("test_hard_loss", hard_loss, on_epoch=True)
        self.log("test_easy_acc", easy_acc, on_epoch=True)
        self.log("test_hard_acc", hard_acc, on_epoch=True)
        return {
            "test_easy_loss": easy_loss.detach(),
            "test_hard_loss": hard_loss.detach(),
            "test_easy_acc": easy_acc.detach(),
            "test_hard_acc": hard_acc.detach(),
        }


def _plot_weight_history(history_df: pd.DataFrame, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.plot(history_df["epoch"], history_df["weight_easy"], label="easy / coarse", linewidth=2.0)
    ax.plot(history_df["epoch"], history_df["weight_hard"], label="hard / fine", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Task weight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)

    loss_ax = ax.twinx()
    smoothed = history_df["val_hard_loss"].rolling(window=5, min_periods=1, center=True).mean()
    loss_ax.plot(history_df["epoch"], smoothed, color="black", linestyle="--", linewidth=2.0, alpha=0.55)
    loss_ax.set_ylabel("Hard validation loss")
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_weighted_cifar100_training(
    *,
    output_dir: Path,
    run_name: str,
    schedule_policy: TaskSchedulePolicy,
    seed: int = 42,
    batch_size: int = 256,
    num_workers: int = 8,
    max_epochs: int = 30,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    ema_alpha: float = 0.3,
    history_window: int = 5,
    val_fraction: float = 0.1,
    patience: int = 8,
    min_delta: float = 1e-4,
    use_early_stopping: bool = False,
    accelerator: str = "auto",
    devices: int | str = 1,
    precision: str = "16-mixed",
) -> dict[str, Any]:
    seed_everything(seed, workers=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = CIFAR100EasyHardDataModule(
        batch_size=batch_size,
        num_workers=num_workers,
        val_fraction=val_fraction,
        seed=seed,
    )
    model = CIFAR100EasyHardModule(
        schedule_policy=schedule_policy,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        ema_alpha=ema_alpha,
        history_window=history_window,
        run_name=run_name,
        seed=seed,
    )

    logger_csv = CSVLogger(save_dir=str(output_dir), name="lightning_logs")
    checkpoint = ModelCheckpoint(
        dirpath=output_dir / "checkpoints",
        filename="best",
        monitor="val_hard_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks: list[Any] = [checkpoint]
    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_hard_loss",
                mode="min",
                patience=patience,
                min_delta=min_delta,
            )
        )
    trainer = Trainer(
        default_root_dir=str(output_dir),
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger_csv,
        callbacks=callbacks,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        deterministic=True,
        log_every_n_steps=25,
    )
    trainer.fit(model, datamodule=data)
    test_rows = trainer.test(model, datamodule=data, ckpt_path="best", verbose=False)

    history_df = pd.DataFrame(model.history)
    history_path = output_dir / "trajectory.csv"
    history_df.to_csv(history_path, index=False)
    weight_plot_path = output_dir / "weights_and_hard_val.png"
    _plot_weight_history(history_df, weight_plot_path, title=run_name)

    test_row = test_rows[0] if test_rows else {}
    summary = {
        "run_name": run_name,
        "seed": seed,
        "best_hard_val_loss": float(model.best_hard_val_loss),
        "final_hard_val_loss": float(history_df["val_hard_loss"].iloc[-1]),
        "best_hard_val_acc": float(history_df["val_hard_acc"].max()),
        "final_hard_val_acc": float(history_df["val_hard_acc"].iloc[-1]),
        "epochs_trained": int(len(history_df)),
        "test_hard_loss": float(test_row.get("test_hard_loss", float("nan"))),
        "test_hard_acc": float(test_row.get("test_hard_acc", float("nan"))),
        "trajectory_path": str(history_path),
        "weight_plot_path": str(weight_plot_path),
        "best_checkpoint_path": str(checkpoint.best_model_path),
        "use_early_stopping": bool(use_early_stopping),
        "max_epochs": int(max_epochs),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    logger.info(
        f"{run_name} | best_hard_val_loss={summary['best_hard_val_loss']:.4f} "
        f"| epochs={summary['epochs_trained']}"
    )
    return summary
