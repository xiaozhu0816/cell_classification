from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tifffile
from PIL import Image
from torch.utils.data import Dataset

_LABEL_PATTERN = re.compile(r"_p(?P<position>\d+)_")


@dataclass
class FrameSelectionConfig:
    """Configuration describing how to sample a frame from a time-course TIFF."""

    strategy: str = "window_random"  # options: target, window_random, window_center, mean_project, max_project
    target_frame: int = 64
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    frames_per_hour: float = 2.0
    percentile: float = 90.0

    @classmethod
    def from_dict(cls, data: Optional[Dict]) -> "FrameSelectionConfig":
        data = data or {}
        cfg = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        if "window_hours" in (data or {}):
            start_h, end_h = data["window_hours"]
            cfg.window_start = math.floor(start_h * cfg.frames_per_hour)
            cfg.window_end = math.ceil(end_h * cfg.frames_per_hour)
        return cfg


@dataclass
class DataItem:
    path: Path
    label: int
    condition: str
    position: str
    total_frames: Optional[int] = None


@dataclass
class DataSplit:
    train: List[DataItem]
    val: List[DataItem]
    test: List[DataItem]


class TimeCourseTiffDataset(Dataset):
    """Torch dataset for multi-frame TIFF stacks coming from live-cell imaging."""

    def __init__(
        self,
        items: Sequence[DataItem],
        transform: Optional[Callable] = None,
        frame_cfg: Optional[FrameSelectionConfig] = None,
    ) -> None:
        self.items = list(items)
        self.transform = transform
        self.frame_cfg = frame_cfg or FrameSelectionConfig()

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        frame = self._load_frame(item)
        image = self._to_image(frame)
        if self.transform:
            image = self.transform(image)
        return image, item.label, {"path": str(item.path), "condition": item.condition, "position": item.position}

    def _load_frame(self, item: DataItem) -> np.ndarray:
        cfg = self.frame_cfg
        with tifffile.TiffFile(item.path) as tif:
            total_frames = len(tif.pages)
            index = self._select_frame_index(cfg, total_frames)
            if cfg.strategy in {"mean_project", "max_project"}:
                stack = tif.asarray()
                if cfg.strategy == "mean_project":
                    frame = stack.mean(axis=0)
                else:
                    frame = stack.max(axis=0)
            else:
                index = max(0, min(total_frames - 1, index))
                frame = tif.asarray(key=index)
        return frame.astype(np.float32)

    def _select_frame_index(self, cfg: FrameSelectionConfig, total_frames: int) -> int:
        strategy = cfg.strategy.lower()
        if strategy == "target":
            return cfg.target_frame
        if strategy == "window_center":
            start = cfg.window_start or 0
            end = cfg.window_end or total_frames
            return (start + end) // 2
        if strategy == "window_random":
            start = cfg.window_start or 0
            end = cfg.window_end or total_frames
            if end <= start:
                end = total_frames
            return random.randint(start, max(start, end - 1))
        if strategy == "percentile":
            percentile = np.clip(cfg.percentile, 0, 100)
            return int(round((percentile / 100.0) * (total_frames - 1)))
        return cfg.target_frame

    def _to_image(self, frame: np.ndarray) -> Image.Image:
        frame = frame - frame.min()
        max_val = frame.max()
        if max_val > 0:
            frame = frame / max_val
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        elif frame.shape[-1] == 1:
            frame = np.repeat(frame, 3, axis=-1)
        return Image.fromarray(frame)


def _scan_condition(condition_dir: Path, label: int, condition_name: str) -> List[DataItem]:
    condition_dir = Path(condition_dir)
    if not condition_dir.exists():
        raise FileNotFoundError(f"Condition directory not found: {condition_dir}")
    items: List[DataItem] = []
    for path in sorted(condition_dir.glob("*.tif*")):
        match = _LABEL_PATTERN.search(path.name)
        position = match.group("position") if match else "unknown"
        items.append(DataItem(path=path, label=label, condition=condition_name, position=position))
    if not items:
        raise RuntimeError(f"No TIFF files discovered in {condition_dir}")
    return items


def _stratified_split(items: Sequence[DataItem], ratios: Sequence[float], seed: int = 42) -> DataSplit:
    if len(ratios) != 3:
        raise ValueError("Split ratios must contain three values (train/val/test)")
    if not math.isclose(sum(ratios), 1.0, abs_tol=1e-2):
        raise ValueError("Split ratios must sum to 1.0")
    label_buckets: Dict[int, List[DataItem]] = {}
    for item in items:
        label_buckets.setdefault(item.label, []).append(item)
    rng = random.Random(seed)
    for bucket in label_buckets.values():
        rng.shuffle(bucket)
    train, val, test = [], [], []
    for bucket in label_buckets.values():
        n = len(bucket)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        train.extend(bucket[:n_train])
        val.extend(bucket[n_train : n_train + n_val])
        test.extend(bucket[n_train + n_val :])
    return DataSplit(train=train, val=val, test=test)


def build_datasets(
    data_cfg: Dict,
    transforms: Dict[str, Optional[Callable]],
) -> Tuple[TimeCourseTiffDataset, TimeCourseTiffDataset, TimeCourseTiffDataset]:
    infected_dir = Path(data_cfg["infected_dir"])
    uninfected_dir = Path(data_cfg["uninfected_dir"])
    infected_label = data_cfg.get("infected_label", 1)
    uninfected_label = data_cfg.get("uninfected_label", 0)
    infected_items = _scan_condition(infected_dir, infected_label, "infected")
    uninfected_items = _scan_condition(uninfected_dir, uninfected_label, "uninfected")
    all_items = infected_items + uninfected_items

    ratios = data_cfg.get("split_ratios", [0.7, 0.15, 0.15])
    split = _stratified_split(all_items, ratios, seed=data_cfg.get("split_seed", 42))

    frame_cfg = FrameSelectionConfig.from_dict(data_cfg.get("frames"))

    train_ds = TimeCourseTiffDataset(split.train, transform=transforms.get("train"), frame_cfg=frame_cfg)
    val_ds = TimeCourseTiffDataset(split.val, transform=transforms.get("val"), frame_cfg=frame_cfg)
    test_ds = TimeCourseTiffDataset(split.test, transform=transforms.get("test"), frame_cfg=frame_cfg)
    return train_ds, val_ds, test_ds
