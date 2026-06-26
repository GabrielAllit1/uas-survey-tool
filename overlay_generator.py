import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer, CRS
import os
import logging
from utils import get_utm_crs_for_geometry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.expanduser("~/UAS_Survey_Tool_Logs"), 'overlay_generator.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_overlay_map(
    polygon_coords,
    gcp_points,
    verification_points,
    output_path,
    layout_mode="grid"
):
    # GeoDataFrame for AOI
    aoi_polygon = Polygon(polygon_coords)
    gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[aoi_polygon])
    detected_crs = get_utm_crs_for_geometry(polygon_coords)
    logger.info(f"Detected UTM CRS: {detected_crs}")

    transformer = Transformer.from_crs("EPSG:4326", detected_crs, always_xy=True)
    # AOI in UTM
    transformed_coords = [transformer.transform(lon, lat) for lon, lat in polygon_coords]
    transformed_polygon = Polygon(transformed_coords)
    gdf_transformed = gpd.GeoDataFrame(index=[0], crs=detected_crs, geometry=[transformed_polygon])

    fig, ax = plt.subplots(figsize=(8, 8))
    gdf_transformed.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.5)

    # Helper: allow dict or tuple for points
    def extract_xy(pt):
        if isinstance(pt, dict):
            lon, lat = pt.get('lon'), pt.get('lat')
        else:
            lon, lat = pt
        return transformer.transform(lon, lat)

    # Plot GCP/VCPs (only one legend label each)
    gcp_x, gcp_y = zip(*[extract_xy(pt) for pt in gcp_points]) if gcp_points else ([], [])
    vcp_x, vcp_y = zip(*[extract_xy(pt) for pt in verification_points]) if verification_points else ([], [])
    if gcp_x:
        ax.scatter(gcp_x, gcp_y, c='g', marker='o', s=50, label='GCP')
    if vcp_x:
        ax.scatter(vcp_x, vcp_y, c='r', marker='x', s=50, label='VCP')

    ax.set_title(f"Overlay Map (Layout: {layout_mode}) — UTM: {detected_crs}")
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved overlay map to: {output_path}")