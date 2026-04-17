import os
import xarray as xr
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List
from glob import glob


class GhanaTileDataset(Dataset):
    target: List[int]
    def __init__(self, root_dir: str, transform=None) -> None:
        self.root_dir: str = root_dir
        # self.files: List[str] = [f for f in os.listdir(root_dir) if f.endswith(".ncf")]
        self.files: List[str] = glob(f"{root_dir}/**/*.nc") + glob(f"{root_dir}/**/*.ncf")
        self.transforms = transform
        self.class_to_idx = {}
        self.target = []
        

        self.coordinates = []
        self.bboxes = []
        for file in self.files:
            ds = xr.open_dataset(file, engine="netcdf4")
            self.bboxes.append(
                (
                    ds.x.values.min(),
                    ds.x.values.max(),
                    ds.y.values.min(),
                    ds.y.values.max(),
                )
            )

            if "category" in ds.attrs:
                if ds.attrs["category"] in self.class_to_idx:
                    self.target.append(self.class_to_idx[ds.attrs["category"]])
                else:
                    new_index = len(self.class_to_idx)
                    self.class_to_idx[ds.attrs["category"]] = new_index
                    self.target.append(new_index)
            else:
                self.target.append(-1)

            self.coordinates.append(
                (
                    ds.x.values.mean(),
                    ds.y.values.mean(),
                )
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        file_path: str = self.files[idx]
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

        return rgb_tensor, self.target[idx]
