import os
import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List


class GhanaTileDataset(Dataset):
    def __init__(self, root_dir: str, transforms=None) -> None:
        self.root_dir: str = root_dir
        self.files: List[str] = [f for f in os.listdir(root_dir) if f.endswith(".ncf")]
        self.transforms = transforms

        self.coordinates = []
        self.bboxes = []
        for file in self.files:
            ds = xr.open_dataset(os.path.join(root_dir, file))
            self.bboxes.append((ds.x.min(), ds.x.max(), ds.y.min(), ds.y.max()))

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_path: str = os.path.join(self.root_dir, self.files[idx])
        ds = xr.open_dataset(file_path)

        # Extract RGB bands
        rgb = ds[["red", "green", "blue"]].to_array().values
        rgb = np.squeeze(rgb)

        ## normalize images to [0, 1]
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

        # Transpose to (H, W, 3) then to (3, H, W) for PyTorch
        rgb = rgb.transpose(1, 2, 0).astype(np.float32)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1)

        if self.transforms is not None:
            rgb_tensor = self.transforms(rgb_tensor)

        return rgb_tensor
