#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR100


CIFAR100_CLASS_NAMES = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse",
    "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear",
    "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman",
    "worm",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show random CIFAR-100 examples for one class"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("./data"))
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--class-name", type=str, default=None)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--num-images", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--list-classes", action="store_true")
    return parser.parse_args()


def resolve_class_index(args: argparse.Namespace) -> int:
    if args.class_name is not None:
        normalized = args.class_name.strip().lower()
        name_to_idx = {name.lower(): idx for idx, name in enumerate(CIFAR100_CLASS_NAMES)}
        if normalized not in name_to_idx:
            raise ValueError(f"Unknown class name: {args.class_name}")
        return name_to_idx[normalized]
    if args.class_idx is not None:
        if not (0 <= args.class_idx < len(CIFAR100_CLASS_NAMES)):
            raise ValueError(f"class_idx must be in [0, {len(CIFAR100_CLASS_NAMES) - 1}]")
        return args.class_idx
    raise ValueError("Pass either --class-name or --class-idx")


def main() -> None:
    args = parse_args()

    if args.list_classes:
        for idx, name in enumerate(CIFAR100_CLASS_NAMES):
            print(f"{idx:2d}: {name}")
        return

    class_idx = resolve_class_index(args)
    class_name = CIFAR100_CLASS_NAMES[class_idx]

    dataset = CIFAR100(
        root=args.data_dir,
        train=(args.split == "train"),
        download=args.download,
    )

    matching_indices = [idx for idx, label in enumerate(dataset.targets) if label == class_idx]
    if not matching_indices:
        raise RuntimeError(f"No images found for class {class_name} in split {args.split}")

    rng = random.Random(args.seed)
    selected = rng.sample(matching_indices, k=min(args.num_images, len(matching_indices)))

    cols = min(4, len(selected))
    rows = math.ceil(len(selected) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    flat_axes = np.atleast_1d(axes).ravel()

    for ax, sample_idx in zip(flat_axes, selected, strict=False):
        image, _ = dataset[sample_idx]
        ax.imshow(image)
        ax.set_title(f"idx={sample_idx}")
        ax.axis("off")

    for ax in flat_axes[len(selected):]:
        ax.axis("off")

    fig.suptitle(f"CIFAR-100 {args.split}: {class_name} ({len(selected)} random examples)")
    fig.tight_layout()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=180, bbox_inches="tight")
        print(f"Saved {args.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
