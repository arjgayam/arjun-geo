"""
Functions to download DC boundary data, generate (or load) temperature/NDVI
rasters, and compute zonal statistics per census tract.
"""

import os
import zipfile
import io
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from rasterio.mask import mask as rasterio_mask
from rasterio.features import geometry_mask
from shapely.geometry import mapping
import requests


# ---------------------------------------------------------------------------
# Download DC census tracts
# ---------------------------------------------------------------------------

TIGER_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2020/TRACT/tl_2020_11_tract.zip"
)


def download_dc_tracts(data_dir):
    """Download DC census tracts from Census Bureau TIGER/Line files.

    Returns a GeoDataFrame in EPSG:4326. Caches the result as a GeoJSON
    so subsequent runs skip the download.
    """
    geojson_path = os.path.join(data_dir, "dc_tracts.geojson")

    if os.path.exists(geojson_path):
        gdf = gpd.read_file(geojson_path)
        print(f"Loaded cached census tracts: {len(gdf)} tracts")
        return gdf

    print(f"Downloading DC census tracts from Census Bureau...")
    try:
        resp = requests.get(TIGER_URL, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"Download failed: {e}")
        print("Check your internet connection and try again.")
        raise

    # Extract the shapefile from the zip
    shp_dir = os.path.join(data_dir, "tracts_shp")
    os.makedirs(shp_dir, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(shp_dir)

    # Find the .shp file
    shp_file = [f for f in os.listdir(shp_dir) if f.endswith(".shp")][0]
    gdf = gpd.read_file(os.path.join(shp_dir, shp_file))
    gdf = gdf.to_crs(epsg=4326)

    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"Downloaded {len(gdf)} census tracts for Washington, DC")
    return gdf


# ---------------------------------------------------------------------------
# Generate synthetic rasters
# ---------------------------------------------------------------------------

# Approximate location of the US Capitol — serves as "city center"
CENTER_LON, CENTER_LAT = -77.009, 38.890


def generate_synthetic_rasters(dc_gdf, data_dir):
    """Create synthetic LST and NDVI GeoTIFFs that mimic realistic urban
    heat island patterns for Washington, DC.

    The temperature raster is driven by a vegetation proxy that is correlated
    with — but not identical to — the NDVI raster, so an ML model trained on
    NDVI won't achieve a perfect score.

    To use real satellite data instead, replace the generated .tif files in
    data/ with actual Landsat-derived LST and NDVI rasters covering DC
    (EPSG:4326). The rest of the pipeline will work unchanged.
    """
    temp_path = os.path.join(data_dir, "dc_temperature.tif")
    ndvi_path = os.path.join(data_dir, "dc_ndvi.tif")

    if os.path.exists(temp_path) and os.path.exists(ndvi_path):
        print("Using cached rasters.")
        return temp_path, ndvi_path

    lon_min, lat_min, lon_max, lat_max = dc_gdf.total_bounds
    res = 0.001  # roughly 100 m

    n_cols = int(np.ceil((lon_max - lon_min) / res))
    n_rows = int(np.ceil((lat_max - lat_min) / res))
    transform = from_bounds(lon_min, lat_min, lon_max, lat_max, n_cols, n_rows)

    # Coordinate grids (pixel centers)
    cols, rows = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    lons = lon_min + (cols + 0.5) * res
    lats = lat_max - (rows + 0.5) * res

    # Distance from city center, normalized to [0, 1]
    dist = np.sqrt((lons - CENTER_LON) ** 2 + (lats - CENTER_LAT) ** 2)
    norm_dist = dist / dist.max()

    # --- NDVI raster ---
    # Higher vegetation farther from city center, with boosts at major parks
    rng1 = np.random.default_rng(seed=42)
    ndvi = 0.15 + 0.35 * norm_dist + rng1.normal(0, 0.08, (n_rows, n_cols))
    # Boost near Rock Creek Park
    rc_dist = np.sqrt((lons - (-77.05)) ** 2 + (lats - 38.93) ** 2)
    ndvi += 0.20 * np.exp(-(rc_dist ** 2) / (2 * 0.018 ** 2))
    # Boost near Anacostia River corridor
    ana_dist = np.sqrt((lons - (-76.96)) ** 2 + (lats - 38.86) ** 2)
    ndvi += 0.15 * np.exp(-(ana_dist ** 2) / (2 * 0.015 ** 2))
    # Boost near National Arboretum
    arb_dist = np.sqrt((lons - (-76.97)) ** 2 + (lats - 38.91) ** 2)
    ndvi += 0.12 * np.exp(-(arb_dist ** 2) / (2 * 0.010 ** 2))
    ndvi = np.clip(ndvi, 0.05, 0.85).astype(np.float32)

    # --- Temperature raster ---
    # Temperature is driven primarily by vegetation (through a correlated
    # but distinct proxy), with a modest distance component and localized
    # hot/cool anomalies. This keeps the ML problem non-trivial.
    rng2 = np.random.default_rng(seed=99)

    # Vegetation proxy — similar to NDVI but with its own noise and
    # slightly shifted park centers, so NDVI is a good-but-imperfect predictor
    veg_proxy = 0.15 + 0.35 * norm_dist + rng2.normal(0, 0.10, (n_rows, n_cols))
    veg_proxy += 0.18 * np.exp(
        -((lons - (-77.045)) ** 2 + (lats - 38.925) ** 2) / (2 * 0.016 ** 2)
    )
    veg_proxy += 0.13 * np.exp(
        -((lons - (-76.965)) ** 2 + (lats - 38.865) ** 2) / (2 * 0.013 ** 2)
    )
    veg_proxy = np.clip(veg_proxy, 0.05, 0.85)

    # Temperature: vegetation is the dominant driver, distance adds a
    # secondary gradient, plus localized anomalies and noise
    temperature = 37.0 - 1.5 * norm_dist - 9.0 * veg_proxy

    # Localized hot/cool spots (parking lots, plazas, ponds, etc.)
    n_spots = 25
    spot_lons = rng2.uniform(lon_min + 0.01, lon_max - 0.01, n_spots)
    spot_lats = rng2.uniform(lat_min + 0.01, lat_max - 0.01, n_spots)
    spot_temps = rng2.uniform(-2.5, 2.5, n_spots)
    for i in range(n_spots):
        sd = np.sqrt((lons - spot_lons[i]) ** 2 + (lats - spot_lats[i]) ** 2)
        temperature += spot_temps[i] * np.exp(-(sd ** 2) / (2 * 0.008 ** 2))

    temperature += rng2.normal(0, 1.0, (n_rows, n_cols))
    temperature = np.clip(temperature, 25.0, 42.0).astype(np.float32)

    # Mask pixels outside DC boundary
    try:
        dc_union = dc_gdf.union_all()
    except AttributeError:
        dc_union = dc_gdf.geometry.unary_union

    outside_dc = geometry_mask(
        [mapping(dc_union)],
        out_shape=(n_rows, n_cols),
        transform=transform,
        invert=False,  # True where OUTSIDE the geometry
    )
    temperature[outside_dc] = np.nan
    ndvi[outside_dc] = np.nan

    # Write GeoTIFFs
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": n_cols,
        "height": n_rows,
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": float("nan"),
    }

    with rasterio.open(temp_path, "w", **profile) as dst:
        dst.write(temperature, 1)
    with rasterio.open(ndvi_path, "w", **profile) as dst:
        dst.write(ndvi, 1)

    print(f"Generated synthetic rasters: {n_cols}x{n_rows} pixels at ~{res}° resolution")
    return temp_path, ndvi_path


# ---------------------------------------------------------------------------
# Zonal statistics
# ---------------------------------------------------------------------------

def compute_zonal_stats(gdf, raster_path, column_name):
    """Compute the mean raster value within each polygon.

    Uses rasterio.mask to extract pixels per geometry. Polygons that contain
    no valid (non-NaN) pixels get NaN.
    """
    means = []
    with rasterio.open(raster_path) as src:
        for _, row in gdf.iterrows():
            geom = [mapping(row.geometry)]
            try:
                out_image, _ = rasterio_mask(src, geom, crop=True, nodata=np.nan)
                valid = out_image[0][~np.isnan(out_image[0])]
                means.append(float(np.mean(valid)) if len(valid) > 0 else np.nan)
            except ValueError:
                means.append(np.nan)

    result = gdf.copy()
    result[column_name] = means
    return result


# ---------------------------------------------------------------------------
# Build the full feature table
# ---------------------------------------------------------------------------

def build_feature_dataframe(gdf, temp_path, ndvi_path):
    """Combine zonal stats, distance, and area into one GeoDataFrame.

    Each row is a census tract with columns:
      GEOID, mean_temp, mean_ndvi, distance_to_center, area_km2, geometry
    """
    print("  Computing mean temperature per tract...")
    gdf = compute_zonal_stats(gdf, temp_path, "mean_temp")
    print("  Computing mean NDVI per tract...")
    gdf = compute_zonal_stats(gdf, ndvi_path, "mean_ndvi")

    # Distance from each tract's centroid to the Capitol (project first to
    # avoid the geographic-CRS centroid warning)
    gdf_proj = gdf.to_crs(epsg=32618)
    centroids_proj = gdf_proj.geometry.centroid
    centroids_4326 = centroids_proj.to_crs(epsg=4326)
    gdf["distance_to_center"] = np.sqrt(
        (centroids_4326.x - CENTER_LON) ** 2
        + (centroids_4326.y - CENTER_LAT) ** 2
    )

    # Tract area in km²
    gdf_utm = gdf.to_crs(epsg=32618)
    gdf["area_km2"] = gdf_utm.geometry.area / 1e6

    n_missing = gdf[["mean_temp", "mean_ndvi"]].isna().any(axis=1).sum()
    print(f"  Feature table ready: {len(gdf)} tracts, {n_missing} with missing data")
    return gdf
