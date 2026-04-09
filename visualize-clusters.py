## extract features
## use pca or tsne to reduce dimensionality
## use k-means or dbscan to cluster the data
## visualize the clusters using matplotlib or seaborn
import torch
import numpy as np
from GeoSSL.geossl.datasets import get_dataset_spec
from scripts.gh_tile_dataset import GhanaTileDataset
import pandas as pd
import xarray as xr
from argparse import ArgumentParser
import os
import torchvision.transforms as T
from tqdm import tqdm


from GeoSSL.geossl.backbones import ResNetBackbone


def extract_features_to_dataframe(
    root_dir: str,
    fraction: float = 1.0,
    backbone_type: str = "resnet18",
    dataset_id: str = "eurosat_rgb",
    method: str = "simclr",
    export_csv: str = None,
    device: str = "cuda:0",
    tile_data=True,
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

    # Define transform for resizing to match eurosat input size (64x64)
    dataset_spec = get_dataset_spec(dataset_id)
    normalize = T.Normalize(mean=dataset_spec.mean, std=dataset_spec.std)
    transform = T.Compose(
        [
            T.Resize(dataset_spec.size),
            T.CenterCrop(dataset_spec.crop_size),
            *([T.ToTensor()] if dataset_id == "eurosat_rgb" else []),
            normalize,
        ]
    )
    dataset = GhanaTileDataset(root_dir, transforms=transform)

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

            if tile_data: ## if reading xarrays
                row["image_name"] = dataset.files[idx]
                row["x_min"] = dataset.bboxes[idx][0]
                row["x_max"] = dataset.bboxes[idx][1]
                row["y_min"] = dataset.bboxes[idx][2]
                row["y_max"] = dataset.bboxes[idx][3]

            else:
                row["image_name"] = os.path.basename(dataset.imgs[idx])
                row["label"] = dataset.imgs[idx][1]

            for i, f in enumerate(feature_np):
                row[f"feature_{i}"] = f

            features_list.append(row)

    # Create DataFrame
    df = pd.DataFrame(features_list)

    # Export to CSV if specified
    if export_csv:
        df.to_csv(export_csv, index=False)

    return df


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="data/ghana-grid-tiles")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--backbone_type", type=str, default="resnet18")
    parser.add_argument("--dataset_id", type=str, default="resisc45")
    parser.add_argument("--method", type=str, default="simclr")
    parser.add_argument("--export_csv", type=str, default="data/features.csv")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tile_data", type=bool, default=True)

    args = parser.parse_args()

    df = extract_features_to_dataframe(
        root_dir=args.root_dir,
        fraction=args.fraction,
        backbone_type=args.backbone_type,
        dataset_id=args.dataset_id,
        method=args.method,
        export_csv=args.export_csv,
        device=args.device,
        tile_data=args.tile_data,
    )
