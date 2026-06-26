import numpy as np
import requests
import time
import logging
from shapely.geometry import Polygon, Point
from shapely.ops import transform as shapely_transform
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def auto_detect_utm_crs(lon, lat):
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    logger.info(f"Detected UTM EPSG: {epsg}")
    return CRS.from_epsg(epsg)

def generate_grid_points_in_polygon(polygon, spacing_m=30):
    src_crs = CRS.from_epsg(4326)
    utm_crs = auto_detect_utm_crs(*polygon.centroid.xy)
    transformer_to_utm = Transformer.from_crs(src_crs, utm_crs, always_xy=True).transform
    transformer_to_wgs = Transformer.from_crs(utm_crs, src_crs, always_xy=True).transform

    poly_utm = shapely_transform(transformer_to_utm, polygon)
    minx, miny, maxx, maxy = poly_utm.bounds
    xs = np.arange(minx, maxx, spacing_m)
    ys = np.arange(miny, maxy, spacing_m)
    points_utm = [Point(x, y) for x in xs for y in ys if poly_utm.contains(Point(x, y))]
    points_latlon = [transformer_to_wgs(pt.x, pt.y) for pt in points_utm]
    # OpenTopoData and OpenTopography expect (lat,lon)
    points_latlon = [(lat, lon) for lon, lat in points_latlon]
    return points_latlon, xs, ys, poly_utm

def fetch_elevation_otd(points_latlon, dataset="srtm90m", batch_size=100, pause_sec=1.05):
    url = f"https://api.opentopodata.org/v1/{dataset}"
    elevations = []
    session = requests.Session()
    logger.info(f"Fetching {len(points_latlon)} points from OpenTopoData ({dataset})")
    for i in range(0, len(points_latlon), batch_size):
        batch = points_latlon[i:i+batch_size]
        locs = "|".join(f"{lat},{lon}" for lat, lon in batch)
        params = {"locations": locs}
        try:
            r = session.get(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
            for res in data.get("results", []):
                elev = res.get("elevation")
                elevations.append(elev if elev is not None else np.nan)
        except Exception as e:
            logger.error(f"OTD API batch {i//batch_size+1} failed: {e}")
            elevations.extend([np.nan]*len(batch))
        time.sleep(pause_sec)
    session.close()
    return np.array(elevations)

def fetch_opentopography_raster(polygon, demtype="NASADEM", api_key=None, output_dir=None):
    minx, miny, maxx, maxy = polygon.bounds
    url = (
        "https://portal.opentopography.org/API/globaldem"
        f"?demtype={demtype}"
        f"&south={miny}&north={maxy}&west={minx}&east={maxx}"
        "&outputFormat=GTiff"
        f"&API_Key={api_key}"
    )
    logger.info(f"Downloading OpenTopography DEM: {url}")
    r = requests.get(url)
    if r.status_code == 200 and r.content:
        out_dir = output_dir or os.getcwd()
        out_path = os.path.join(out_dir, f"opentopo_{demtype}.tif")
        with open(out_path, "wb") as f:
            f.write(r.content)
        logger.info(f"Saved OpenTopography DEM: {out_path}")
        return out_path
    else:
        logger.error(f"OpenTopography DEM download failed: {r.status_code} {r.text}")
        return None

def synthesize_virtual_dem(polygon_coords, spacing_m=30, plot=True, use_opentopo=False, api_key=None):
    polygon = Polygon(polygon_coords)
    points_latlon, xs, ys, poly_utm = generate_grid_points_in_polygon(polygon, spacing_m)
    if not use_opentopo:
        elevations = fetch_elevation_otd(points_latlon)
        grid_shape = (len(np.arange(poly_utm.bounds[0], poly_utm.bounds[2], spacing_m)),
                      len(np.arange(poly_utm.bounds[1], poly_utm.bounds[3], spacing_m)))
        if elevations.size != grid_shape[0] * grid_shape[1]:
            logger.warning("Elevation count does not match grid shape, filling with nan")
            z = np.full((grid_shape[0], grid_shape[1]), np.nan)
            for idx, elev in enumerate(elevations):
                ix = idx % grid_shape[0]
                iy = idx // grid_shape[0]
                if ix < z.shape[0] and iy < z.shape[1]:
                    z[ix, iy] = elev
        else:
            z = elevations.reshape(grid_shape)
        x = np.arange(poly_utm.bounds[0], poly_utm.bounds[2], spacing_m)
        y = np.arange(poly_utm.bounds[1], poly_utm.bounds[3], spacing_m)
        if plot:
            plt.figure(figsize=(10, 8))
            plt.imshow(z.T, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()], cmap="terrain")
            plt.colorbar(label="Elevation (m)")
            plt.title("Virtual DEM (OpenTopoData bulk query)")
            plt.xlabel("UTM Easting (m)")
            plt.ylabel("UTM Northing (m)")
            plt.show()
        return x, y, z
    else:
        if not api_key:
            raise ValueError("API key required for OpenTopography DEM.")
        tiff_path = fetch_opentopography_raster(polygon, demtype="NASADEM", api_key=api_key)
        logger.info(f"OpenTopography DEM file saved: {tiff_path}")
        return tiff_path

# Example usage:
if __name__ == "__main__":
    # Your AOI polygon, WGS84 lon/lat
    polygon_coords = [(-122.5, 37.7), (-122.45, 37.7), (-122.45, 37.75), (-122.5, 37.75), (-122.5, 37.7)]
    # Use OpenTopoData (free, 100pts/sec) for "virtual raster"
    x, y, z = synthesize_virtual_dem(polygon_coords, spacing_m=90, plot=True)
    # To download a true raster from OpenTopography (needs API key), pass your own key:
    # tiff_path = synthesize_virtual_dem(polygon_coords, use_opentopo=True, api_key="YOUR_API_KEY")
