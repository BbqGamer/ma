from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from urllib.request import urlretrieve
import zipfile

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_FOLDER = "tiny-imagenet-200"


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    num_classes: int
    class_names: list[str] | None
    input_shape: tuple[int, int, int]
    class_group_ids: list[int] | None = None
    class_group_names: list[str] | None = None


class ImageArrayDataset(Dataset):
    def __init__(
        self,
        images: Sequence[np.ndarray],
        labels: Sequence[int],
        transform: Callable | None = None,
    ) -> None:
        self.images = list(images)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = self.images[index]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image))
        image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


class ImagePathDataset(Dataset):
    def __init__(
        self,
        image_paths: Sequence[Path],
        labels: Sequence[int],
        transform: Callable | None = None,
    ) -> None:
        self.image_paths = list(image_paths)
        self.labels = list(labels)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[index])


CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

MNIST_CLASS_NAMES = [str(idx) for idx in range(10)]

FASHION_MNIST_CLASS_NAMES = [
    "t-shirt_top",
    "trouser",
    "pullover",
    "dress",
    "coat",
    "sandal",
    "shirt",
    "sneaker",
    "bag",
    "ankle_boot",
]

KMNIST_CLASS_NAMES = ["o", "ki", "su", "tsu", "na", "ha", "ma", "ya", "re", "wo"]

SVHN_CLASS_NAMES = [str(idx) for idx in range(10)]

STL10_CLASS_NAMES = [
    "airplane",
    "bird",
    "car",
    "cat",
    "deer",
    "dog",
    "horse",
    "monkey",
    "ship",
    "truck",
]

CIFAR100_CLASS_NAMES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]


CIFAR100_COARSE_CLASS_NAMES = [
    "aquatic_mammals",
    "fish",
    "flowers",
    "food_containers",
    "fruit_and_vegetables",
    "household_electrical_devices",
    "household_furniture",
    "insects",
    "large_carnivores",
    "large_man-made_outdoor_things",
    "large_natural_outdoor_scenes",
    "large_omnivores_and_herbivores",
    "medium_mammals",
    "non-insect_invertebrates",
    "people",
    "reptiles",
    "small_mammals",
    "trees",
    "vehicles_1",
    "vehicles_2",
]

CIFAR100_FINE_TO_COARSE = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def split_indices(num_samples: int, ratio: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_samples)
    num_val = int(num_samples * ratio)
    val_indices = perm[:num_val]
    train_indices = perm[num_val:]
    return train_indices, val_indices


def build_transforms(dataset_name: str, augmentation: bool) -> tuple[Callable, Callable]:
    train_transforms: list[Callable] = []
    eval_transforms: list[Callable] = []

    if dataset_name in {"mnist", "fashion-mnist", "kmnist"}:
        train_transforms.extend([transforms.Resize(32), transforms.Grayscale(num_output_channels=3)])
        eval_transforms.extend([transforms.Resize(32), transforms.Grayscale(num_output_channels=3)])

    if augmentation:
        if dataset_name in {"cifar10", "cifar100", "svhn", "mnist", "fashion-mnist", "kmnist"}:
            train_transforms.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        elif dataset_name == "stl10":
            train_transforms.extend(
                [
                    transforms.RandomCrop(96, padding=12),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        elif dataset_name == "tiny-imagenet":
            train_transforms.extend(
                [
                    transforms.RandomCrop(64, padding=8),
                    transforms.RandomHorizontalFlip(),
                ]
            )

    train_transforms.append(transforms.ToTensor())
    eval_transforms.append(transforms.ToTensor())
    return transforms.Compose(train_transforms), transforms.Compose(eval_transforms)


def load_dataset(
    dataset_name: str,
    data_dir: Path,
    val_ratio: float,
    seed: int,
    download: bool,
    augmentation: bool,
    shapes_path: Path | None = None,
    tiny_imagenet_path: Path | None = None,
    shapes_test_ratio: float = 0.2,
) -> DatasetBundle:
    train_transform, eval_transform = build_transforms(dataset_name, augmentation)
    if dataset_name == "cifar10":
        return load_cifar10(data_dir, val_ratio, seed, download, train_transform, eval_transform)
    if dataset_name == "cifar100":
        return load_cifar100(data_dir, val_ratio, seed, download, train_transform, eval_transform)
    if dataset_name == "mnist":
        return load_mnist_like(
            datasets.MNIST,
            "mnist",
            MNIST_CLASS_NAMES,
            data_dir,
            val_ratio,
            seed,
            download,
            train_transform,
            eval_transform,
        )
    if dataset_name == "fashion-mnist":
        return load_mnist_like(
            datasets.FashionMNIST,
            "fashion-mnist",
            FASHION_MNIST_CLASS_NAMES,
            data_dir,
            val_ratio,
            seed,
            download,
            train_transform,
            eval_transform,
        )
    if dataset_name == "kmnist":
        return load_mnist_like(
            datasets.KMNIST,
            "kmnist",
            KMNIST_CLASS_NAMES,
            data_dir,
            val_ratio,
            seed,
            download,
            train_transform,
            eval_transform,
        )
    if dataset_name == "svhn":
        return load_svhn(data_dir, val_ratio, seed, download, train_transform, eval_transform)
    if dataset_name == "stl10":
        return load_stl10(data_dir, val_ratio, seed, download, train_transform, eval_transform)
    if dataset_name == "shapes":
        root = shapes_path if shapes_path is not None else data_dir / "shapes"
        return load_shapes(root, val_ratio, shapes_test_ratio, seed, train_transform, eval_transform)
    if dataset_name == "tiny-imagenet":
        root = tiny_imagenet_path if tiny_imagenet_path is not None else data_dir / TINY_IMAGENET_FOLDER
        return load_tiny_imagenet(root, val_ratio, seed, download, train_transform, eval_transform)
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def load_cifar10(
    data_dir: Path,
    val_ratio: float,
    seed: int,
    download: bool,
    train_transform: Callable,
    eval_transform: Callable,
) -> DatasetBundle:
    train_base = datasets.CIFAR10(root=data_dir, train=True, download=download, transform=train_transform)
    train_eval_base = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=download,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=download,
        transform=eval_transform,
    )
    train_indices, val_indices = split_indices(len(train_base), val_ratio, seed)
    return DatasetBundle(
        name="cifar10",
        train_dataset=Subset(train_base, train_indices.tolist()),
        val_dataset=Subset(train_eval_base, val_indices.tolist()),
        test_dataset=test_dataset,
        num_classes=10,
        class_names=CIFAR10_CLASS_NAMES,
        input_shape=(3, 32, 32),
    )


def load_cifar100(
    data_dir: Path,
    val_ratio: float,
    seed: int,
    download: bool,
    train_transform: Callable,
    eval_transform: Callable,
) -> DatasetBundle:
    train_base = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=train_transform,
    )
    train_eval_base = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=download,
        transform=eval_transform,
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=download,
        transform=eval_transform,
    )
    train_indices, val_indices = split_indices(len(train_base), val_ratio, seed)
    return DatasetBundle(
        name="cifar100",
        train_dataset=Subset(train_base, train_indices.tolist()),
        val_dataset=Subset(train_eval_base, val_indices.tolist()),
        test_dataset=test_dataset,
        num_classes=100,
        class_names=CIFAR100_CLASS_NAMES,
        input_shape=(3, 32, 32),
        class_group_ids=CIFAR100_FINE_TO_COARSE,
        class_group_names=CIFAR100_COARSE_CLASS_NAMES,
    )


def load_mnist_like(
    dataset_cls: type,
    dataset_name: str,
    class_names: list[str],
    data_dir: Path,
    val_ratio: float,
    seed: int,
    download: bool,
    train_transform: Callable,
    eval_transform: Callable,
) -> DatasetBundle:
    train_base = dataset_cls(root=data_dir, train=True, download=download, transform=train_transform)
    train_eval_base = dataset_cls(root=data_dir, train=True, download=download, transform=eval_transform)
    test_dataset = dataset_cls(root=data_dir, train=False, download=download, transform=eval_transform)
    train_indices, val_indices = split_indices(len(train_base), val_ratio, seed)
    return DatasetBundle(
        name=dataset_name,
        train_dataset=Subset(train_base, train_indices.tolist()),
        val_dataset=Subset(train_eval_base, val_indices.tolist()),
        test_dataset=test_dataset,
        num_classes=10,
        class_names=class_names,
        input_shape=(3, 32, 32),
    )


def load_svhn(
    data_dir: Path,
    val_ratio: float,
    seed: int,
    download: bool,
    train_transform: Callable,
    eval_transform: Callable,
) -> DatasetBundle:
    train_base = datasets.SVHN(root=data_dir, split="train", download=download, transform=train_transform)
    train_eval_base = datasets.SVHN(root=data_dir, split="train", download=download, transform=eval_transform)
    test_dataset = datasets.SVHN(root=data_dir, split="test", download=download, transform=eval_transform)
    train_indices, val_indices = split_indices(len(train_base), val_ratio, seed)
    return DatasetBundle(
        name="svhn",
        train_dataset=Subset(train_base, train_indices.tolist()),
        val_dataset=Subset(train_eval_base, val_indices.tolist()),
        test_dataset=test_dataset,
        num_classes=10,
        class_names=SVHN_CLASS_NAMES,
        input_shape=(3, 32, 32),
    )


def load_stl10(
    data_dir: Path,
    val_ratio: float,
    seed: int,
    download: bool,
    train_transform: Callable,
    eval_transform: Callable,
) -> DatasetBundle:
    train_base = datasets.STL10(root=data_dir, split="train", download=download, transform=train_transform)
    train_eval_base = datasets.STL10(root=data_dir, split="train", download=download, transform=eval_transform)
    test_dataset = datasets.STL10(root=data_dir, split="test", download=download, transform=eval_transform)
    train_indices, val_indices = split_indices(len(train_base), val_ratio, seed)
    return DatasetBundle(
        name="stl10",
        train_dataset=Subset(train_base, train_indices.tolist()),
        val_dataset=Subset(train_eval_base, val_indices.tolist()),
        test_dataset=test_dataset,
        num_classes=10,
        class_names=STL10_CLASS_NAMES,
        input_shape=(3, 96, 96),
    )


def load_shapes(
    root: Path,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    train_transform: Callable,
    eval_transform: Callable,
) -> DatasetBundle:
    png_dir = root / "png"
    captions_path = root / "full_captions.txt"
    if not png_dir.exists() or not captions_path.exists():
        raise FileNotFoundError(
            f"Shapes dataset not found at {root}. Expected {png_dir} and {captions_path}."
        )

    labels_text = captions_path.read_text().splitlines()
    unique_labels = sorted(set(labels_text))
    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
    labels = [label_to_id[label] for label in labels_text]

    image_paths = [png_dir / f"{idx}.png" for idx in range(len(labels))]
    missing = [str(path) for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing shapes images, e.g. {missing[0]}")

    train_pool, test_indices = split_indices(len(image_paths), test_ratio, seed)
    train_indices, val_indices = split_indices(len(train_pool), val_ratio, seed + 1)
    train_indices = train_pool[train_indices]
    val_indices = train_pool[val_indices]

    train_dataset = Subset(
        ImagePathDataset(image_paths, labels, transform=train_transform),
        train_indices.tolist(),
    )
    val_dataset = Subset(
        ImagePathDataset(image_paths, labels, transform=eval_transform),
        val_indices.tolist(),
    )
    test_dataset = Subset(
        ImagePathDataset(image_paths, labels, transform=eval_transform),
        test_indices.tolist(),
    )
    return DatasetBundle(
        name="shapes",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_classes=len(unique_labels),
        class_names=unique_labels,
        input_shape=(3, 64, 64),
    )


def maybe_download_tiny_imagenet(target_root: Path) -> Path:
    if target_root.name == TINY_IMAGENET_FOLDER and (target_root / "wnids.txt").exists():
        print(f"Tiny ImageNet already available at {target_root}", flush=True)
        return target_root
    if (target_root / TINY_IMAGENET_FOLDER / "wnids.txt").exists():
        existing_root = target_root / TINY_IMAGENET_FOLDER
        print(f"Tiny ImageNet already available at {existing_root}", flush=True)
        return existing_root

    target_root.mkdir(parents=True, exist_ok=True)
    archive_path = target_root / f"{TINY_IMAGENET_FOLDER}.zip"
    print(f"Downloading Tiny ImageNet from {TINY_IMAGENET_URL} to {archive_path}", flush=True)
    urlretrieve(TINY_IMAGENET_URL, archive_path)
    print(f"Extracting Tiny ImageNet archive to {target_root}", flush=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(target_root)
    return target_root / TINY_IMAGENET_FOLDER


def load_tiny_imagenet(
    root: Path,
    val_ratio: float,
    seed: int,
    download: bool,
    train_transform: Callable,
    eval_transform: Callable,
) -> DatasetBundle:
    if root.name != TINY_IMAGENET_FOLDER:
        canonical_root = root / TINY_IMAGENET_FOLDER
    else:
        canonical_root = root

    if download:
        canonical_root = maybe_download_tiny_imagenet(root if root.name != TINY_IMAGENET_FOLDER else root.parent)
    if not canonical_root.exists():
        raise FileNotFoundError(f"Tiny ImageNet not found at {canonical_root}")

    wnids = canonical_root.joinpath("wnids.txt").read_text().splitlines()
    wnid_to_idx = {wnid: idx for idx, wnid in enumerate(wnids)}
    class_names = []
    words = {}
    for line in canonical_root.joinpath("words.txt").read_text().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            words[parts[0]] = parts[1].replace(", ", "->").replace(" ", "_")
    for wnid in wnids:
        class_names.append(words.get(wnid, wnid))

    train_paths: list[Path] = []
    train_labels: list[int] = []
    for wnid in wnids:
        image_dir = canonical_root / "train" / wnid / "images"
        for image_path in sorted(image_dir.glob("*.JPEG")):
            train_paths.append(image_path)
            train_labels.append(wnid_to_idx[wnid])

    val_annotations = {}
    for line in (canonical_root / "val" / "val_annotations.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            val_annotations[parts[0]] = parts[1]

    test_paths: list[Path] = []
    test_labels: list[int] = []
    for image_path in sorted((canonical_root / "val" / "images").glob("*.JPEG")):
        wnid = val_annotations[image_path.name]
        test_paths.append(image_path)
        test_labels.append(wnid_to_idx[wnid])

    train_indices, val_indices = split_indices(len(train_paths), val_ratio, seed)
    train_dataset = Subset(
        ImagePathDataset(train_paths, train_labels, transform=train_transform),
        train_indices.tolist(),
    )
    val_dataset = Subset(
        ImagePathDataset(train_paths, train_labels, transform=eval_transform),
        val_indices.tolist(),
    )
    test_dataset = ImagePathDataset(test_paths, test_labels, transform=eval_transform)
    return DatasetBundle(
        name="tiny-imagenet",
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        num_classes=200,
        class_names=class_names,
        input_shape=(3, 64, 64),
    )


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
