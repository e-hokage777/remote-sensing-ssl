import xarray as xr
import planetary_computer as pc
import numpy as np
import geopandas as gpd
import argparse
from odc.stac import stac_load
from shapely.geometry import box
import pystac_client
import os
from tqdm import tqdm


def prepare_grid(boundary_path: str):
    """Prepare a grid of tiles that intersect with the given boundary."""
    boundary = gpd.read_file(boundary_path)

    meters_per_degree = 111320.0
    tile_size_meters = 640
    tile_size_degrees = tile_size_meters / meters_per_degree

    minx, miny, maxx, maxy = boundary.total_bounds

    tiles = []
    for x in np.arange(minx, maxx, tile_size_degrees):
        for y in np.arange(miny, maxy, tile_size_degrees):
            tile = box(x, y, x + tile_size_degrees, y + tile_size_degrees)
            if tile.intersects(boundary.union_all("unary")):
                tiles.append(tile)

    return gpd.GeoDataFrame({"geometry": tiles}, crs=boundary.crs)


def get_tile_image(geometry: gpd.GeoSeries):
    """get the image for a given tile geometry."""
    client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1"
    )
    search = client.search(
        collections=["sentinel-2-l2a"],
        intersects=geometry,
        datetime="2025-12-01/2026-04-08",
        query={"eo:cloud_cover": {"lt": 10}},
    )
    items = list(search.items())

    if items:
        ## sorting items
        items = sorted(items, key=lambda item: item.datetime)
        item = items[-1]
        item = pc.sign(item)

        bands = ["red", "green", "blue"]
        data = stac_load([item], bands=bands, intersects=geometry)

        return data
    else:
        print("No items found for the given geometry")
        return None


def save_tiles(grid: gpd.GeoDataFrame, root: str):
    """Save the tiles in the grid to the specified root directory."""
    for index, tile in tqdm(
        grid.sample(frac=1).iterrows(),  ## download randomly
        unit=" tiles",
        desc="Download tiles for given boundary",
        total=len(grid),
    ):
        save_name = root + f"/tile_{index}.ncf"
        if os.path.exists(save_name):
            continue
        tile_raster = get_tile_image(tile.geometry)
        if tile_raster is not None:
            tile_raster.to_netcdf(save_name)


if __name__ == "__main__":
    """Script to download Sentinel-2 tiles for Ghana using the STAC API and save them as NetCDF files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="data/ghana-grid-tiles")
    parser.add_argument("--boundary", type=str, required=True)

    args = parser.parse_args()

    ## create grid
    grid = prepare_grid(args.boundary)

    ## save tiles
    save_tiles(grid, args.output)
