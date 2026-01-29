import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from datetime import datetime
import os
from pyproj import CRS

# -----------------------
# CONFIG
# -----------------------
GREEN_BAND = 2   # Sentinel-2 B3
NIR_BAND   = 3   # Sentinel-2 B8
NDWI_THRESHOLD = 0.0


# -----------------------
# Reproject raster to UTM if needed
# -----------------------



def reproject_to_utm(src):
    bounds = src.bounds
    lon = (bounds.left + bounds.right) / 2
    lat = (bounds.top + bounds.bottom) / 2

    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    utm_crs = CRS.from_epsg(epsg)

    print("Reprojecting to:", utm_crs)

    transform, width, height = calculate_default_transform(
        src.crs, utm_crs, src.width, src.height, *src.bounds
    )

    profile = src.profile.copy()
    profile.update({
        "crs": utm_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    data = np.zeros((src.count, height, width), dtype=np.float32)

    for i in range(1, src.count + 1):
        reproject(
            source=rasterio.band(src, i),
            destination=data[i - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=utm_crs,
            resampling=Resampling.nearest
        )

    return data, profile


# -----------------------
# MAIN PIPELINE
# -----------------------
def process_water_from_image(image_path, lake_id, date_str, output_dir):

    date = datetime.strptime(date_str, "%Y-%m-%d").date()

    os.makedirs(output_dir, exist_ok=True)
    ndwi_dir = os.path.join(output_dir, "ndwi")
    mask_dir = os.path.join(output_dir, "masks")
    poly_dir = os.path.join(output_dir, "polygons")
    os.makedirs(ndwi_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(poly_dir, exist_ok=True)

    with rasterio.open(image_path) as src:
        print("Original CRS:", src.crs)

        if src.crs is None:
            raise ValueError("Raster has no CRS")

        # Reproject if geographic (EPSG:4326 etc.)
        if not src.crs.is_projected:
            data, profile = reproject_to_utm(src)
        else:
            data = src.read().astype("float32")
            profile = src.profile

        green = data[GREEN_BAND - 1]
        nir   = data[NIR_BAND - 1]
        transform = profile["transform"]

        # NDWI
        ndwi = (green - nir) / (green + nir + 1e-6)
        print("NDWI stats:", np.nanmin(ndwi), np.nanmax(ndwi))

        # Save NDWI
        ndwi_path = os.path.join(ndwi_dir, f"{lake_id}_{date}_ndwi.tif")
        ndwi_profile = profile.copy()
        ndwi_profile.update(dtype="float32", count=1, nodata=np.nan)

        with rasterio.open(ndwi_path, "w", **ndwi_profile) as dst:
            dst.write(ndwi, 1)

        # Threshold → water mask
        water_mask = (ndwi >= NDWI_THRESHOLD).astype(np.uint8)
        print("Water pixels:", np.sum(water_mask))

        mask_path = os.path.join(mask_dir, f"{lake_id}_{date}_mask.tif")
        mask_profile = profile.copy()
        mask_profile.update(dtype="uint8", count=1, nodata=0)

        with rasterio.open(mask_path, "w", **mask_profile) as dst:
            dst.write(water_mask, 1)

        # Polygonize
        results = (
            {"properties": {"value": v}, "geometry": s}
            for s, v in shapes(water_mask, transform=transform)
            if v == 1
        )

        geoms = list(results)

        if not geoms:
            print("⚠ No water detected")
            return 0.0

        gdf = gpd.GeoDataFrame.from_features(geoms, crs=profile["crs"])

        # Keep only largest water body
        gdf["area_m2"] = gdf.geometry.area
        gdf = gdf.sort_values("area_m2", ascending=False).head(1)

        # Area in hectares
        gdf["area_ha"] = gdf.geometry.area / 10000
        total_area = float(gdf["area_ha"].iloc[0])

        # Save polygon
        poly_path = os.path.join(poly_dir, f"{lake_id}_{date}_water.geojson")
        gdf.to_file(poly_path, driver="GeoJSON")

        # Save CSV
        csv_path = os.path.join(output_dir, "results.csv")
        df = pd.DataFrame([{
            "lake_id": lake_id,
            "date": date,
            "area_ha": round(total_area, 2)
        }])
        df.to_csv(csv_path, index=False)

        print(f"\n✅ Done → Area = {total_area:.2f} ha")
        return total_area



# -----------------------
# RUN
# -----------------------
if __name__ == "__main__":
    process_water_from_image(
        image_path="data/images/Satellite1.tif",
        lake_id="L001",
        date_str="2023-12-15",
        output_dir="outputs"
    )
