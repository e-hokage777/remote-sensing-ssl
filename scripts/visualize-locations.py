from glob import glob
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def normalize(channel):
    p2, p98 = np.percentile(channel, (2, 98))
    return np.clip((channel - p2) / (p98 - p2), 0, 1)


## plot one image per category
if __name__ == "__main__":
    ## configurations
    np.random.seed(103)
    folders = glob("data/ghana-satellite-imgs/*")

    plt.figure(figsize=(8, 20))

    for index, folder in tqdm(enumerate(folders)):
        category = os.path.basename(folder)
        files = glob(f"{folder}/*.ncf")
        if len(files) == 0:
            continue

        file = files[np.random.randint(0, len(files))]

        ds = xr.open_dataset(file, engine="netcdf4")
        red = np.squeeze(ds["red"])
        green = np.squeeze(ds["green"])
        blue = np.squeeze(ds["blue"])
        image = np.dstack([normalize(red), normalize(green), normalize(blue)])
        plt.subplot(len(folders)//2, 2,  index + 1)
        plt.title(category)
        plt.imshow(image)

    plt.tight_layout
    plt.savefig("plots/location-satellite-imgs.png")
