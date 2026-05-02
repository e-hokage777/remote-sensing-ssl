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


import time
import random
from requests.exceptions import RequestException
import socket


def retry(func, retries=10, delay=2, backoff=2, exceptions=(Exception,)):
    """Retry a function with exponential backoff."""
    for attempt in range(retries):
        try:
            return func()
        except exceptions as e:
            if attempt == retries - 1:
                raise  # re-raise last error

            sleep_time = delay * (backoff**attempt) + random.uniform(0, 1)
            print(
                f"Retry {attempt+1}/{retries} failed: {e}. Sleeping {sleep_time:.2f}s..."
            )
            time.sleep(sleep_time)


from requests.exceptions import RequestException
import socket

RETRY_EXCEPTIONS = (RequestException, TimeoutError, socket.timeout, pystac_client.exceptions.APIError)


def download_sentinel_image(
    lat: float,
    lon: float,
    output_dir: str,
    category_name: str,
    save_name: str,
    tile_size: int = 640,
):
    save_path = os.path.join(output_dir, f"{save_name}.ncf")

    if os.path.exists(save_path):
        return

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

    def search_items():
        search = client.search(
            collections=["sentinel-2-l2a"],
            intersects=boundary,
            datetime="2025-12-01/2026-04-30",
            query={"eo:cloud_cover": {"lt": 10}},
        )
        return list(search.items())

    items = retry(search_items, exceptions=RETRY_EXCEPTIONS)

    if not items:
        print("No items found for the given boundary")
        return

    items = sorted(
        items, key=lambda item: item.datetime or datetime(1900, 1, 1), reverse=True
    )
    items = [pc.sign(item) for item in items]

    bands = ["red", "green", "blue"]

    def load_data():
        return stac_load(items[:10], bands=bands, intersects=boundary)

    data = retry(load_data, exceptions=RETRY_EXCEPTIONS)

    data.attrs["category"] = category_name
    data.attrs["lat"] = lat
    data.attrs["lon"] = lon

    median_data = data.median(dim="time")

    def save_file():
        median_data.to_netcdf(save_path)

    retry(save_file, exceptions=RETRY_EXCEPTIONS)


# def download_sentinel_image(
#     lat: float,
#     lon: float,
#     output_dir: str,
#     category_name: str,
#     save_name: str,
#     tile_size: int = 640,
# ):
#     """Download Sentinel-2 images for the given boundary and save them as NetCDF files."""
#     save_path = os.path.join(output_dir, f"{save_name}.ncf")

#     if os.path.exists(save_path):
#         return

#     meters_per_degree = 111320.0
#     tile_size_degrees = tile_size / meters_per_degree

#     boundary = box(
#         lon - tile_size_degrees / 2,
#         lat - tile_size_degrees / 2,
#         lon + tile_size_degrees / 2,
#         lat + tile_size_degrees / 2,
#     )

#     client = pystac_client.Client.open(
#         "https://planetarycomputer.microsoft.com/api/stac/v1"
#     )
#     search = client.search(
#         collections=["sentinel-2-l2a"],
#         intersects=boundary,
#         datetime="2025-12-01/2026-04-30",
#         query={"eo:cloud_cover": {"lt": 10}},
#     )
#     items = list(search.items())

#     if not items:
#         print("No items found for the given boundary")
#         return

#     # sorting items
#     items = sorted(
#         items, key=lambda item: item.datetime or datetime(1900, 1, 1), reverse=True
#     )
#     # item = items[0]
#     # item = pc.sign(item)
#     items = [pc.sign(item) for item in items]

#     bands = ["red", "green", "blue"]
#     data = stac_load(items[:10], bands=bands, intersects=boundary)
#     data.attrs["category"] = category_name
#     data.attrs["lat"] = lat
#     data.attrs["lon"] = lon

#     median_data = data.median(dim="time")

#     median_data.to_netcdf(save_path)


def get_locations_from_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data["elements"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to the input directory (directory containing geojson metadata files downloaded)",
        default="data/ghana-locations",
    )
    parser.add_argument("--output-dir", type=str, default="data/ghana-satellite-imgs")

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

            if os.path.exists(os.path.join(save_dir, f"{location['id']}.ncf")):
                continue

            download_sentinel_image(
                location["center"]["lat"],
                location["center"]["lon"],
                category_name=category_name,
                save_name=location["id"],
                output_dir=save_dir,
            )
