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
import torchvision.transforms.v2 as v2
from tqdm import tqdm

from GeoSSL.geossl.backbones import ResNetBackbone


def extract_features_to_dataframe(
    dataset: torch.utils.data.Dataset,
    backbone: torch.nn.Module,
    fraction: float = 1.0,
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

    # Move backbone to device
    backbone.to(device)

    # Set backbone to evaluation mode
    backbone.eval()

    features_list = []

    with torch.no_grad():
        for idx in tqdm(indices, total=len(indices), unit=" tile"):

            img = dataset[idx][0]
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


def select_dataset(
    dataset_id: str, download_dataset: bool = False
) -> torch.utils.data.Dataset:
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
    dataset_spec = get_dataset_spec(
        "eurosat_rgb" if dataset_id == "ghana" else dataset_id
    )
    normalize = T.Normalize(mean=dataset_spec.mean, std=dataset_spec.std)
    transform = T.Compose(
        [
            T.Resize(dataset_spec.size),
            T.CenterCrop(dataset_spec.crop_size),
            *(
                [v2.ToDtype(torch.float32, scale=True)]
                if dataset_id == "eurosat_rgb"
                else []
            ),
            normalize,
        ]
    )

    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/data"
    if dataset_id == "eurosat":
        dataset = EuroSAT(
            root=root_dir + "/eurosat",
            download=download_dataset,
            split="val",
            transform=transform,
        )
    elif dataset_id == "eurosat_rgb":
        dataset = EuroSATRGB(
            root=root_dir + "/eurosat",
            download=download_dataset,
            split="val",
            transform=transform,
        )
        dataset.imgs = dataset.eurosat.imgs
    elif dataset_id == "resisc45":
        dataset = Resisc45(
            root=root_dir + "/resisc45",
            download=download_dataset,
            split="val",
            transform=transform,
        )
    elif dataset_id == "ghana":
        dataset = GhanaTileDataset(
            root_dir=root_dir + "/ghana-grid-tiles", transform=transform
        )
    else:
        raise NotImplementedError(f"Dataset [{dataset_id}] not supported")

    return dataset


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="data/ghana-grid-tiles")
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--backbone_type", type=str, default="resnet18")
    parser.add_argument("--backbone_data", type=str, default="eurosat")
    parser.add_argument("--dataset_id", type=str, default="eurosat_rgb")
    parser.add_argument("--method", type=str, default="simclr")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--export_csv", type=str, default="features.csv")
    parser.add_argument("--download_dataset", action="store_true")

    args = parser.parse_args()

    dataset = select_dataset(args.dataset_id, download_dataset=args.download_dataset)

    # Load pretrained backbone for eurosat (remote sensing dataset)
    backbone = ResNetBackbone.from_pretrained(
        f"{args.backbone_type}/{args.backbone_data}/{args.method}"
    )

    df = extract_features_to_dataframe(
        dataset,
        backbone=backbone,
        fraction=args.fraction,
        device=args.device,
    )

    ## add extra features depending on dataset_id
    if args.dataset_id == "ghana":
        names= []
        xmin = []
        xmax = []
        ymin = []
        ymax = []
        
        for idx in df.index:
            xmin.append(dataset.bboxes[idx][0])
            xmax.append(dataset.bboxes[idx][1])
            ymin.append(dataset.bboxes[idx][2])
            ymax.append(dataset.bboxes[idx][3])
            names.append(dataset.files[idx])
        
        df.insert(0, "image_name", names)
        df.insert(1, "x_min", xmin)
        df.insert(2, "x_max", xmax)
        df.insert(3, "y_min", ymin)
        df.insert(4, "y_max", ymax)
    else:
        df.insert(
            0,
            "image_name",
            [os.path.basename(dataset.imgs[idx][0]) for idx in df.index],
        )
        df.insert(1, "label", [dataset.imgs[idx][1] for idx in df.index])

    root_dir = os.path.dirname(os.path.abspath(__file__)) + "/data/features/"
    df.to_csv(root_dir + args.export_csv, index=False)
