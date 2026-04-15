import geopandas as gpd
import pystac_client
import planetary_computer as pc
import xarray as xr
from odc.stac import stac_load
from shapely.geometry import box
import os
from tqdm import tqdm
from glob import glob
import json
from datetime import datetime
from argparse import ArgumentParser
from typing import Dict, List, Any


def download_sentinel_image(
    lat: float, lon: float, output_dir: str, category_name:str, save_name:str, tile_size: int = 640
):
    """Download Sentinel-2 images for the given boundary and save them as NetCDF files."""
    meters_per_degree = 111320.0
    tile_size_degrees = tile_size / meters_per_degree

    boundary = box(
        lon - tile_size_degrees / 2,
        lat - tile_size_degrees / 2,
        lon + tile_size_degrees / 2,
        lat + tile_size_degrees / 2,
    )

    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=boundary,
        datetime="2025-12-01/2026-04-08",
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items = list(search.items())

    if not items:
        print("No items found for the given boundary")
        return

    ## sorting items
    items = sorted(items, key=lambda item: item.datetime or datetime(1900, 1, 1), reverse=True)
    item = items[0]
    item = pc.sign(item)

    bands = ["red", "green", "blue"]
    data = stac_load([item], bands=bands, intersects=boundary)
    data.attrs["category"] = category_name
    data.attrs["lat"] = lat
    data.attrs["lon"] = lon

    save_name = os.path.join(output_dir, f"{save_name}.ncf")
    data.to_netcdf(save_name)


def get_locations_from_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["elements"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    args = parser.parse_args()

    files = glob(f"{args.input_dir}/*.geojson")

    for file in files:
        category_name = os.path.basename(file).split(".")[0]
        print("Processing: ", category_name)
        locations = get_locations_from_json(file)
        for location in tqdm(locations, unit="location", desc="Downloading images"):
            save_dir = os.path.join(args.output_dir, category_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)

            download_sentinel_image(
                location["center"]["lat"],
                location["center"]["lon"],
                category_name=category_name,
                save_name = location["id"],
                output_dir=save_dir
            )
