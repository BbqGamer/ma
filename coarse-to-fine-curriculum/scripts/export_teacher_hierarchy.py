from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ctf.data import load_dataset
from ctf.hierarchy import compute_hierarchy
from train_coarse_to_fine import load_teacher_model, save_hierarchy_artifacts, teacher_embedding_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a hierarchy from teacher embeddings.")
    parser.add_argument("--dataset", default="cifar100", choices=["cifar10", "cifar100", "fashion-mnist", "tiny-imagenet", "stl10"])
    parser.add_argument("--data_dir", default="/workspace/data")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teacher_run_dir", default=None)
    parser.add_argument("--teacher_checkpoint_path", default=None)
    parser.add_argument("--teacher_model", choices=["cnn", "cifar_resnet8", "cifar_resnet14", "cifar_resnet20", "cifar_resnet32", "cifar_resnet44", "cifar_resnet56", "resnet18", "resnet50"], default=None)
    parser.add_argument("--teacher_cnn_width_multiplier", type=float, default=None)
    parser.add_argument("--teacher_cifar_resnet_width_multiplier", type=float, default=None)
    parser.add_argument("--teacher_embedding_split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--teacher_pretrained_source", choices=["none", "torchvision_imagenet"], default="none")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.set_defaults(download=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_dataset(
        dataset_name=args.dataset,
        data_dir=Path(args.data_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        download=args.download,
        augmentation=False,
        shapes_path=None,
        tiny_imagenet_path=None,
        shapes_test_ratio=0.2,
    )
    from train_coarse_to_fine import build_loaders

    train_loader, val_loader, test_loader = build_loaders(bundle, batch_size=256, num_workers=args.num_workers, seed=args.seed, device=device)
    teacher_model, teacher_meta = load_teacher_model(args, bundle, device)
    loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.teacher_embedding_split]
    dist_matrix = teacher_embedding_distance(teacher_model, loader, bundle.num_classes, device)
    hierarchy_levels = compute_hierarchy(dist_matrix, seed=args.seed)

    np.save(out_dir / "distance_matrix_teacher_embeddings.npy", dist_matrix)
    save_hierarchy_artifacts(
        out_dir,
        hierarchy_levels,
        bundle.class_names,
        "teacher_embeddings",
        extra={**teacher_meta, "teacher_embedding_split": args.teacher_embedding_split},
    )
    (out_dir / "teacher_hierarchy_meta.json").write_text(
        json.dumps({**teacher_meta, "dataset": args.dataset, "teacher_embedding_split": args.teacher_embedding_split}, indent=2)
    )
    print(f"Wrote hierarchy to {out_dir}")


if __name__ == "__main__":
    main()
