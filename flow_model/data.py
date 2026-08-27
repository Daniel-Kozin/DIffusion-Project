import os
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import rotate

CASE_ID_RE = re.compile(r"^(.*)_(\d+)$")

# raw value {0,127,200,255} -> class index {0,1,2,3} (background/insert/pillar/lump)
CLASS_LUT = np.zeros(256, dtype=np.int64)
CLASS_LUT[0] = 0
CLASS_LUT[127] = 1
CLASS_LUT[200] = 2
CLASS_LUT[255] = 3


def remap_labels(raw: np.ndarray) -> np.ndarray:
    """uint8 (K,H,W) values {0,127,200,255} -> int64 (K,H,W) values {0,1,2,3}."""
    return CLASS_LUT[raw]


def parse_case_id(stem: str) -> str:
    m = CASE_ID_RE.match(stem)
    return m.group(1) if m else stem


def load_all_volumes(data_dir: Path) -> Dict[str, List[np.ndarray]]:
    """Walks data_dir, groups remapped int-label volumes by case id."""
    case_volumes: Dict[str, List[np.ndarray]] = {}
    for dirpath, _, filenames in os.walk(data_dir):
        for f in sorted(filenames):
            if not f.endswith(".npy"):
                continue
            stem = Path(f).stem
            case_id = parse_case_id(stem)
            raw = np.load(Path(dirpath) / f)
            label = remap_labels(raw)
            case_volumes.setdefault(case_id, []).append(label)
    return case_volumes


def split_case_ids(case_ids: List[str], val_frac: float = 0.2, seed: int = 0) -> Tuple[List[str], List[str]]:
    ids = sorted(case_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = max(1, round(len(ids) * val_frac))
    val_ids = sorted(ids[:n_val])
    train_ids = sorted(ids[n_val:])
    return train_ids, val_ids


def rotate_label_volume(label: np.ndarray, orientation: int) -> np.ndarray:
    """
    label: int64 (K,H,W), values 0-3
    orientation: 0-7 -> angle = -orientation * 45 degrees, applied in-plane (H,W)
    to every depth slice at once via nearest-neighbor interpolation (preserves
    exact class labels, no blending).
    """
    if orientation == 0:
        return label
    angle = -orientation * 45.0
    t = torch.from_numpy(label)  # [K, H, W], leading dim treated as batch by TF.rotate
    rotated = rotate(t, angle, interpolation=InterpolationMode.NEAREST, fill=0)
    return rotated.numpy()


class PhantomDataset(Dataset):
    def __init__(self, case_volumes: Dict[str, List[np.ndarray]], case_ids: List[str],
                 num_orientations: int = 8, augment: bool = True, num_classes: int = 4):
        self.case_volumes = case_volumes
        self.num_classes = num_classes
        orientations = list(range(num_orientations)) if augment else [0]

        self.index: List[Tuple[str, int, int]] = []
        for case_id in case_ids:
            for variant_idx in range(len(case_volumes[case_id])):
                for orientation in orientations:
                    self.index.append((case_id, variant_idx, orientation))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        case_id, variant_idx, orientation = self.index[idx]
        label = self.case_volumes[case_id][variant_idx]
        label = rotate_label_volume(label, orientation)

        label_t = torch.from_numpy(label.copy()).long()  # [K, H, W]
        one_hot = F.one_hot(label_t, num_classes=self.num_classes)  # [K, H, W, C]
        one_hot = one_hot.permute(3, 0, 1, 2).float()  # [C, K, H, W]
        return one_hot


def build_datasets(data_dir: Path, val_frac: float = 0.2, num_orientations: int = 8,
                    seed: int = 0, num_classes: int = 4) -> Tuple[PhantomDataset, PhantomDataset]:
    case_volumes = load_all_volumes(data_dir)
    train_ids, val_ids = split_case_ids(list(case_volumes.keys()), val_frac=val_frac, seed=seed)

    train_ds = PhantomDataset(case_volumes, train_ids, num_orientations=num_orientations,
                               augment=True, num_classes=num_classes)
    val_ds = PhantomDataset(case_volumes, val_ids, num_orientations=num_orientations,
                             augment=False, num_classes=num_classes)
    return train_ds, val_ds


def build_dataloaders(train_ds: PhantomDataset, val_ds: PhantomDataset,
                       batch_size: int = 2, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
