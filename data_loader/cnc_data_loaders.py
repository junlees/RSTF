"""
CNC Machining Dataset Loader
────────────────────────────
Supports the Bosch CNC Machining Dataset:
  https://github.com/boschresearch/CNC_Machining

Expected directory layout
─────────────────────────
data/
  M01/
    OP02/
      good/
        <name>.h5   ← OK (normal) samples
      bad/
        <name>.h5   ← NOK (anomaly) samples
    OP05/ ...
  M02/ ...
  M03/ ...

Each HDF5 file contains 'vibration_data' key: (N, 3) float array (x, y, z axes).
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset
from base import BaseDataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class CNCVibrationDataset(Dataset):
    """
    Sliding-window dataset built from the Bosch CNC Machining HDF5 files.

    Parameters
    ──────────
    data_dir      : root directory (contains M01/, M02/, M03/)
    machines      : list of machine IDs to load, e.g. ['M01', 'M02']
    operations    : list of operation IDs to load, e.g. ['OP02', 'OP05']
                    None → load all found operations
    window_size   : number of time-steps per sample  (default 50)
    overlap       : overlap between consecutive windows (default 25)
    training      : if True, augment NOK samples with Gaussian noise
    normalize     : z-score normalise each window independently
    """

    LABEL_OK  = 1   # normal  (good/)
    LABEL_NOK = 0   # anomaly (bad/)

    def __init__(
        self,
        data_dir: str,
        machines=None,
        operations=None,
        window_size: int = 50,
        overlap: int = 25,
        training: bool = True,
        normalize: bool = True,
    ):
        self.data_dir    = Path(data_dir)
        self.window_size = window_size
        self.stride      = window_size - overlap
        self.training    = training
        self.normalize   = normalize

        self.windows: list[np.ndarray] = []
        self.labels:  list[int]        = []

        # Discover machines / ops
        if machines is None:
            machines = sorted(p.name for p in self.data_dir.iterdir() if p.is_dir())

        self._load(machines, operations)

        self.windows = np.stack(self.windows, axis=0).astype(np.float32)
        self.labels  = np.array(self.labels, dtype=np.int64)

        print(
            f"[CNCVibrationDataset] {len(self.windows)} windows  |  "
            f"OK={np.sum(self.labels == 1)}  NOK={np.sum(self.labels == 0)}"
        )

    # ── internal helpers ──────────────────────────────────────────────────

    def _load(self, machines, operations):
        for machine in machines:
            machine_dir = self.data_dir / machine
            if not machine_dir.exists():
                print(f"  [WARN] machine dir not found: {machine_dir}")
                continue

            op_dirs = sorted(machine_dir.iterdir()) if operations is None else [
                machine_dir / op for op in operations
            ]

            for op_dir in op_dirs:
                if not op_dir.is_dir():
                    continue
                # good/ → LABEL_OK, bad/ → LABEL_NOK
                for subdir, label in [("good", self.LABEL_OK), ("bad", self.LABEL_NOK)]:
                    sub_path = op_dir / subdir
                    if not sub_path.is_dir():
                        continue
                    for h5_path in sorted(sub_path.glob("*.h5")):
                        self._extract_windows(h5_path, label)

    def _extract_windows(self, h5_path: Path, label: int):
        try:
            with h5py.File(h5_path, "r") as f:
                data = f["vibration_data"][:]   # (N, 3)
        except Exception as e:
            print(f"  [WARN] could not read {h5_path}: {e}")
            return

        if data.ndim == 1 or data.shape[0] < self.window_size:
            return

        data = data[:, :3].astype(np.float32)

        n_windows = (len(data) - self.window_size) // self.stride + 1
        for i in range(n_windows):
            start = i * self.stride
            end   = start + self.window_size
            window = data[start:end].copy()          # (W, 3)

            # Data augmentation for NOK samples during training
            if self.training and label == self.LABEL_NOK:
                noise = np.random.normal(0, 0.01 * np.std(window), window.shape)
                window = window + noise

            self.windows.append(window)
            self.labels.append(label)

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        window = self.windows[idx].copy()            # (W, 3)
        label  = self.labels[idx]

        if self.normalize:
            mean = window.mean(axis=0, keepdims=True)
            std  = window.std(axis=0, keepdims=True) + 1e-8
            window = (window - mean) / std

        return torch.from_numpy(window), torch.tensor(label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader
# ─────────────────────────────────────────────────────────────────────────────

class CNCDataLoader(BaseDataLoader):
    """
    DataLoader that wraps CNCVibrationDataset and inherits train/val splitting
    from BaseDataLoader.

    config.json example
    ───────────────────
    "data_loader": {
        "type": "CNCDataLoader",
        "args": {
            "data_dir"        : "data/",
            "machines"        : ["M01"],
            "operations"      : ["OP02","OP05","OP08","OP11","OP14"],
            "window_size"     : 50,
            "overlap"         : 25,
            "batch_size"      : 128,
            "validation_split": 0.15,
            "num_workers"     : 2,
            "training"        : true
        }
    }
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        shuffle: bool = True,
        validation_split: float = 0.15,
        num_workers: int = 2,
        machines=None,
        operations=None,
        window_size: int = 50,
        overlap: int = 25,
        training: bool = True,
        normalize: bool = True,
    ):
        self.dataset = CNCVibrationDataset(
            data_dir=data_dir,
            machines=machines,
            operations=operations,
            window_size=window_size,
            overlap=overlap,
            training=training,
            normalize=normalize,
        )

        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_split=validation_split,
            num_workers=num_workers,
        )
