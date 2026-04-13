import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.manifold import TSNE
from typing import Iterable
from argparse import ArgumentParser
from glob import glob
import math
import os


def reduce_dimension(df: pd.DataFrame, n_components=2) -> pd.DataFrame:
    tsne = TSNE(n_components=n_components, perplexity=30)
    reduced_data = tsne.fit_transform(df)
    reduced_df = pd.DataFrame(reduced_data, columns=[f"component_{i}" for i in range(n_components)])
    return reduced_df

def plot_scatter(component_1:Iterable, component_2:Iterable, axis:Axes, title:str, c:Iterable = None):
    assert len(component_1) == len(component_2), "Both components must have the same length" # make sure both comopnents have the same length
    axis.scatter(component_1, component_2, s=4, c=c, cmap="tab10")
    axis.set_xlabel("Component 1")
    axis.set_ylabel("Component 2")
    axis.set_title(title)
    axis.axis("off")

def generate_plot(root_dir: str, dataset_id:str, save_path:str):
    files = glob(f"{root_dir}/{dataset_id}*.csv")

    assert len(files) > 0, f"No csv files found like {root_dir}/{dataset_id}*.csv"

    n_cols = 2
    n_rows = int(math.ceil(len(files) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,n_rows*6))
    axes=axes.flatten()
    # plt.figure(figsize=(12,n_rows*6))

    for i, file in enumerate(files):
        title=os.path.basename(file).split(".")[0]
        print("Processing", file)
        df = pd.read_csv(file)
        df = df[~df.isna().any(axis=1)]
        features = df.filter(regex="^feature")
        features_decomposed = reduce_dimension(features, n_components=2)
        plot_scatter(component_1=features_decomposed["component_0"], component_2=features_decomposed["component_1"], title=title, axis=axes[i], c=df["label"] if "label" in df.columns else None)

    # plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path)
    


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--dataset_id", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    
    args = parser.parse_args()

    generate_plot(args.root_dir, args.dataset_id, args.output)

