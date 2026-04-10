## extract features
## use pca or tsne to reduce dimensionality
## use k-means or dbscan to cluster the data
## visualize the clusters using matplotlib or seaborn
import torch
import numpy as np
from GeoSSL.geossl.datasets import get_dataset_spec, EuroSATRGB, Resisc45, EuroSAT
from scripts.gh_tile_dataset import GhanaTileDataset
import pandas as pd
import xarray as xr
from argparse import ArgumentParser
import os
import torchvision.transforms as T
from tqdm import tqdm

from GeoSSL.geossl.backbones import ResNetBackbone


def extract_features_to_dataframe(
    dataset: torch.utils.data.Dataset,
    fraction: float = 1.0,
    backbone_type: str = "resnet18",
    dataset_id: str = "eurosat_rgb",
    method: str = "simclr",
    device: str = "cuda:0",
) -> pd.DataFrame:
    """
    Extract features from images using ResNetBackbone and create a DataFrame.

    Args:
        root_dir: Path to the directory containing .ncf files
        fraction: Fraction of the dataset to use (0.0 to 1.0)
        export_csv: Path to export the DataFrame as CSV (optional)

    Returns:
        DataFrame with image names, coordinates, and feature columns
    """

    # Sample indices if fraction < 1.0
    total = len(dataset)
    if fraction < 1.0:
        sample_size = int(total * fraction)
        indices = np.random.choice(total, sample_size, replace=False)
    else:
        indices = range(total)

    # Load pretrained backbone for eurosat (remote sensing dataset)
    backbone = ResNetBackbone.from_pretrained(f"{backbone_type}/{dataset_id}/{method}")

    # Move backbone to device
    backbone.to(device)

    # Set backbone to evaluation mode
    backbone.eval()

    features_list = []

    with torch.no_grad():
        for idx in tqdm(indices, total=len(indices), unit=" tile"):

            img = dataset[idx]
            feature = backbone(img.unsqueeze(0).to(device))  # Add batch dimension
            feature_np = feature.cpu().squeeze().numpy()

            # Create row dict
            row = {}

            for i, f in enumerate(feature_np):
                row[f"feature_{i}"] = f

            features_list.append(row)

    # Create DataFrame
    df = pd.DataFrame(features_list, index=indices)

    return df


def select_dataset(dataset_id: str) -> torch.utils.data.Dataset:
    """
    Selects the appropriate dataset given the dataset_id (eurosat, eurosat_rgb, resisc45, ghana)

    Args:
        dataset_id (str): _description_

    Raises:
        NotImplementedError: _description_

    Returns:
        torch.utils.data.Dataset: _description_
    """

    # Define transform for resizing to match eurosat input size (64x64)
    dataset_spec = get_dataset_spec(args.dataset_id)
    normalize = T.Normalize(mean=dataset_spec.mean, std=dataset_spec.std)
    transform = T.Compose(
        [
            T.Resize(dataset_spec.size),
            T.CenterCrop(dataset_spec.crop_size),
            *([T.ToTensor()] if args.dataset_id == "eurosat_rgb" else []),
            normalize,
        ]
    )

    if args.dataset_id == "eurosat":
        dataset = EuroSAT(
            root=os.path.abspath(__file__) + "/../data/eurosat",
            download=True,
            split="val",
            transform=transform,
        )
    elif args.dataset_id == "eurosat_rgb":
        dataset = EuroSATRGB(
            root=os.path.abspath(__file__) + "/../data/eurosat",
            download=True,
            split="val",
            transform=transform,
        )
    elif args.dataset_id == "resisc45":
        dataset = Resisc45(
            root=os.path.abspath(__file__) + "/../data/resisc45",
            download=True,
            split="val",
            transform=transform,
        )
    elif args.dataset_id == "ghana":
        dataset = GhanaTileDataset(root_dir=args.root_dir, transform=transform)
    else:
        raise NotImplementedError("Dataset not supported")

    return dataset



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="data/ghana-grid-tiles")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--backbone_type", type=str, default="resnet18")
    parser.add_argument("--dataset_id", type=str, default="resisc45")
    parser.add_argument("--method", type=str, default="simclr")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()

    dataset = select_dataset(args.dataset_id)

    df = extract_features_to_dataframe(
        dataset,
        fraction=args.fraction,
        backbone_type=args.backbone_type,
        dataset_id=args.dataset_id,
        method=args.method,
        device=args.device,
    )

    ## add extra features depending on dataset_id
    if args.dataset_id == "ghana":
        df["image_name"] = dataset.files
        df["x_min"] = dataset.bboxes[:, 0]
        df["x_max"] = dataset.bboxes[:, 1]
        df["y_min"] = dataset.bboxes[:, 2]
        df["y_max"] = dataset.bboxes[:, 3]
    else:
        df["image_name"] = [os.path.basename(f) for f in dataset.imgs]
        df["label"] = [f[1] for f in dataset.imgs]

    df.to_csv(args.export_csv, index=False)
