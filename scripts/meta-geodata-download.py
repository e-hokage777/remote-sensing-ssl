import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm
from argparse import ArgumentParser

OVERPASS_URL = "https://overpass-api.de/api/interpreter"


def get_locations_by_category(category, country="Ghana", limit: int | None = None):
    category_map = {
        "annual_crop": [
            'way["landuse"="farmland"]',
            'relation["landuse"="farmland"]',
        ],
        "permanent_crop": [
            'way["landuse"~"orchard|vineyard|plantation"]',
            'relation["landuse"~"orchard|vineyard|plantation"]',
        ],
        "forest": [
            'way["landuse"="forest"]',
            'relation["landuse"="forest"]',
            'way["natural"="wood"]',
            'relation["natural"="wood"]',
        ],
        "herbaceous_vegetation": [
            'way["natural"~"grassland|scrub"]',
            'relation["natural"~"grassland|scrub"]',
        ],
        "pasture": ['way["landuse"="meadow"]', 'relation["landuse"="meadow"]'],
        "residential": [
            'way["landuse"="residential"]',
            'relation["landuse"="residential"]',
        ],
        "industrial": [
            'way["landuse"="industrial"]',
            'relation["landuse"="industrial"]',
        ],
        "highway": ['way["highway"]'],
        "river": [
            'way["waterway"~"river|stream"]',
            'relation["waterway"~"river|stream"]',
        ],
        "sea_lake": ['way["natural"="water"]', 'relation["natural"="water"]'],
    }

    if category not in category_map:
        raise ValueError(f"Invalid category: {category}")

    # Build query parts
    query_parts = "\n  ".join(
        [f"{query}(area.{country.lower()});" for query in category_map[category]]
    )

    query = f"""
    [out:json][timeout:120];
    area["name"="{country}"]->.{country.lower()};
    (
      {query_parts}
    );
    out ids center {limit};
    """

    session = requests.Session()

    retry_strategy = Retry(
        total=5,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=2,
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount(OVERPASS_URL, adapter)

    response = session.get(OVERPASS_URL, params={"data": query}, stream=True)

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--country", type=str, default="Ghana")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default="data/ghana-locations")
    args = parser.parse_args()

    categories = [
        "annual_crop",
        "permanent_crop",
        "forest",
        "herbaceous_vegetation",
        "pasture",
        "residential",
        "industrial",
        "highway",
        "river",
        "sea_lake",
    ]

    print("Categories are ", ", ".join(categories))

    for category in tqdm(
        categories,
        desc=f"Extracting metadata for {args.limit} locations per category",
        unit="category",
        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}",
    ):
        # print(f"Downloading data for category: {category}")
        data = get_locations_by_category(
            category, country=args.country, limit=args.limit
        )
        with open(f"{args.output_dir}/{category}_ghana.geojson", "w") as f:
            f.write(str(data))
