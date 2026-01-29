import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.io import MemoryFile
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape
from datetime import datetime
import os
from pyproj import CRS
from s2cloudless import S2PixelCloudDetector

# -----------------------
# CLOUD MASK (s2cloudless)
# -----------------------
def apply_s2cloudless_mask(data):
    """
    data: (bands, H, W) -> expects B2,B3,B4,B8 or compatible sequence.
    returns: cloud_mask (H, W)
    """
    # Move bands to last axis: (H, W, bands)
    img = np.moveaxis(data, 0, -1)
    
    # Scale to 0-1 (Sentinel-2 usually 0-10000 range, s2cloudless expects reflectance < 1 logic or trained on it)
    # The user provided snippet divides by 10000.0, so we do that.
    img = img / 10000.0

    cloud_detector = S2PixelCloudDetector(
        threshold=0.4,
        average_over=4,
        dilation_size=2
    )

    try:
        # Note: default S2PixelCloudDetector expects 10 bands. 
        # If input has fewer, this might fail unless bands="timestamp" or similar customization is used.
        # But per user instruction, we use this code.
        probs = cloud_detector.get_cloud_probability_maps(img)
        cloud_mask = probs > 0.4
        print(f"☁️ Cloud pixels masked: {np.sum(cloud_mask)}")
        return cloud_mask
    except ValueError as ve:
        print(f"⚠️ Cloud detection skipped (band mismatch?): {ve}")
        return np.zeros(img.shape[:2], dtype=bool)
    except Exception as e:
        print(f"⚠️ Cloud detection failed: {e}")
        return np.zeros(img.shape[:2], dtype=bool)

# -----------------------
# CONFIG
# -----------------------
GREEN_BAND = 2   # Sentinel-2 B3
NIR_BAND   = 4   # Sentinel-2 B8
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


def calculate_lake_volume(dem_path, polygon_path, base_level=None, current_level=None, target_area_ha=None):
    # -----------------------------
    # 1. Load polygon
    # -----------------------------
    gdf = gpd.read_file(polygon_path)

    if gdf.crs is None:
        raise ValueError("Polygon has no CRS")

    # Reproject polygon to UTM if needed
    if not gdf.crs.is_projected:
        gdf = gdf.to_crs(epsg=32644)

    # -----------------------------
    # 2. Load DEM
    # -----------------------------
    with rasterio.open(dem_path) as src:
        src_array = src.read(1)
        src_transform = src.transform
        src_crs = src.crs
        src_meta = src.meta.copy()

    # Reproject DEM to match polygon CRS
    dst_crs = gdf.crs

    transform, width, height = calculate_default_transform(
        src_crs, dst_crs, src.width, src.height, *src.bounds
    )

    with MemoryFile() as memfile:
        meta = src_meta.copy()
        meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "nodata": -32767
        })

        with memfile.open(**meta) as dst:
            reproject(
                source=src_array,
                destination=rasterio.band(dst, 1),
                src_transform=src_transform,
                src_crs=src_crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear
            )

            # -----------------------------
            # 3. Clip DEM to polygon
            # -----------------------------
            geoms = [geom for geom in gdf.geometry]
            try:
                clipped_img, clipped_transform = mask(dst, geoms, crop=True)
                dem_clipped = clipped_img[0]
            except ValueError:
                 print("⚠️ Polygon does not overlap with DEM")
                 return {"min_elevation": 0, "max_elevation": 0, "max_volume_m3": 0}

    # -----------------------------
    # 4. Zonal stats
    # -----------------------------
    valid = dem_clipped != -32767
    vals = dem_clipped[valid]
    
    if vals.size == 0:
        return {"min_elevation": 0, "max_elevation": 0, "max_volume_m3": 0}

    min_elev = float(vals.min())
    max_elev = float(vals.max())

    # -----------------------------
    # 5. Volume curve
    # -----------------------------
    step = 0.1
    levels = np.arange(min_elev, max_elev + step, step)

    pixel_area = abs(clipped_transform[0] * clipped_transform[4])

    records = []
    for level in levels:
        depth = level - dem_clipped
        mask_arr = (dem_clipped != -32767) & (dem_clipped < level)
        
        if np.any(mask_arr):
            volume_m3 = np.sum(depth[mask_arr]) * pixel_area
            area_m2 = np.sum(mask_arr) * pixel_area
        else:
            volume_m3 = 0.0
            area_m2 = 0.0

        volume_tmc = volume_m3 / 28316846.592
        area_ha = area_m2 / 10000.0
        records.append([level, volume_m3, volume_tmc, area_ha])

    df = pd.DataFrame(records, columns=["water_level", "volume_m3", "volume_tmc", "area_ha"])
    
    # CSV saving is handled by caller (analyze_lake)
    # if result_csv_path: ...

    # Example: return final max volume
    final_volume = float(df["volume_m3"].max())
    
    volume_at_base = None
    if base_level is not None:
        print(f"[DEBUG] Interpolating volume for Base Level: {base_level}")
        # Interpolate or find closest
        try:
             # Ensure sorted for interpolation
            df_sorted = df.sort_values("water_level")
            
            min_lvl = df_sorted["water_level"].min()
            max_lvl = df_sorted["water_level"].max()
            print(f"[DEBUG] Volume Curve Range: {min_lvl} to {max_lvl}")
            
            if base_level < min_lvl:
                print(f"[DEBUG] Base Level {base_level} < Min Level {min_lvl} -> 0")
                volume_at_base = 0.0
            elif base_level > max_lvl:
                print(f"[DEBUG] Base Level {base_level} > Max Level {max_lvl} -> Max Vol")
                volume_at_base = final_volume
            else:
                volume_at_base = np.interp(base_level, df_sorted["water_level"], df_sorted["volume_m3"])
                print(f"[DEBUG] Interpolated Volume: {volume_at_base}")
        except Exception as e:
            print(f"[ERROR] Interpolation error: {e}")
            volume_at_base = 0.0

    # Interpolate for Current Water Level (Detected either by Area or Mask Stats)
    volume_at_current = None
    water_level_at_current = None
    
    # If target_area (from Satellite) is provided, use Area-Elevation mapping (Most Accurate)
    if current_level is not None and isinstance(current_level, str) and current_level == "AREA_LOOKUP":
        pass # Logic handled below if target_area provided
        
    # Interpolate for Current Water Level
    # Priority: Area Lookup > Elev Lookup
    
    if target_area_ha is not None:
        print(f"[DEBUG] Calculating Volume via Area Lookup (Target: {target_area_ha:.2f} Ha)")
        try:
            df_sorted = df.sort_values("area_ha")
            max_area = df_sorted["area_ha"].max()
            
            if target_area_ha <= 0:
                water_level_at_current = min_elev
                volume_at_current = 0.0
            elif target_area_ha > max_area:
                 # Be careful, max_area might be smaller than satellite area if DEM is clipped tight
                 print(f"[WARN] Target Area {target_area_ha} > Max Curv Area {max_area}. Using Max.")
                 water_level_at_current = max_elev
                 volume_at_current = final_volume
            else:
                # Find Level for Area
                water_level_at_current = np.interp(target_area_ha, df_sorted["area_ha"], df_sorted["water_level"])
                # Find Volume for that Level (or just interp area->vol directly)
                volume_at_current = np.interp(target_area_ha, df_sorted["area_ha"], df_sorted["volume_m3"])
                
            print(f"[DEBUG] Matched Level: {water_level_at_current:.2f}m, Vol: {volume_at_current}")
            
        except Exception as e:
            print(f"[ERROR] Area lookup failed: {e}")
            volume_at_current = 0.0
            water_level_at_current = min_elev

    elif current_level is not None and isinstance(current_level, (int, float)):
        # Fallback to Elev Lookup
        water_level_at_current = current_level
        try:
            if current_level < min_elev:
                volume_at_current = 0.0
            elif current_level > max_elev:
                volume_at_current = final_volume
            else:
                volume_at_current = np.interp(current_level, df["water_level"], df["volume_m3"])
        except:
             volume_at_current = 0.0

    return {
        "min_elevation": min_elev,
        "max_elevation": max_elev,
        "max_volume_m3": final_volume,
        "volume_at_base_level": volume_at_base,
        "volume_at_current_level": volume_at_current,
        "water_level_recalculated": water_level_at_current,
        "curve_df": df
    }


# -----------------------
# MAIN PIPELINE
# -----------------------
def process_water_from_image(image_path, lake_id, date_str, output_dir):
    # ... (existing code)
    date = datetime.strptime(date_str, "%Y-%m-%d").date()
    # ...
    # (Leaving this function mostly alone, just ensuring it returns correct dict as before)
    # But wait, I'm replacing lines, so I need to be careful not to delete process_water_from_image body.
    # Actually, I am only replacing the end of calculate_lake_volume.
    pass

# Redoing chunk to be safe and simple.

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
            
        print("[DEBUG] Raster Shape:", data.shape)

        # -----------------------------
        # CLOUD MASKING
        # -----------------------------
        cloud_mask = apply_s2cloudless_mask(data)

        # -----------------------------
        # DYNAMIC BAND SELECTION
        # -----------------------------
        num_bands = data.shape[0]
        print(f"[DEBUG] Input has {num_bands} bands.")
        
        if num_bands >= 12:
            # Full Sentinel-2 (L1C/L2A usually 13 bands)
            # Band 3 = Green (Index 2)
            # Band 8 = NIR (Index 7)
            green_idx = 2
            nir_idx = 7
            print("[DEBUG] Using Full Sentinel-2 Mapping: Green=B3(idx 2), NIR=B8(idx 7)")
        else:
            # 4-Band Subset (export default)
            # [Blue, Green, Red, NIR]
            # Green = Index 1
            # NIR = Index 3
            green_idx = 1
            nir_idx = 3
            print(f"[DEBUG] Using Subset Mapping: Green=idx {green_idx}, NIR=idx {nir_idx}")

        # Bands
        green = data[green_idx]
        nir   = data[nir_idx]
        
        # Apply mask
        green = np.where(cloud_mask, np.nan, green)
        nir   = np.where(cloud_mask, np.nan, nir)
        
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
            return {"area_ha": 0.0, "polygon_path": None}

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
        return {
            "area_ha": total_area, 
            "polygon_path": poly_path
        }


def analyze_lake(sat_path, dem_path, lake_id, date_str, output_dir, base_level=None):
    # 1. Run water detection
    result = process_water_from_image(
        image_path=sat_path,
        lake_id=lake_id,
        date_str=date_str,
        output_dir=output_dir
    )
    
    area_ha = result["area_ha"]
    polygon_path = result["polygon_path"]
    
    volume_data = {
        "min_elevation": 0,
        "max_elevation": 0,
        "max_volume_m3": 0,
        "volume_at_base_level": None,
        "volume_at_current_level": 0
    }

    # 2. Run volume logic if DEM provided
    if dem_path:
        current_water_level = None
        
        # A. Detect Current Water Level from Mask
        if polygon_path:
            try:
                gdf = gpd.read_file(polygon_path)
                with rasterio.open(dem_path) as src:
                    # Reproject GDF to DEM crs if needed
                    if src.crs and gdf.crs and src.crs != gdf.crs:
                        gdf_proj = gdf.to_crs(src.crs)
                    else:
                        gdf_proj = gdf
                        
                    # Mask DEM with water polygon
                    geoms = [g for g in gdf_proj.geometry]
                    out_image, out_transform = mask(src, geoms, crop=True)
                    
                    nodata = src.nodata
                    if nodata is None:
                        nodata = -32767 # Fallback
                        print("[WARN] DEM nodata is None, assuming -32767")
                    
                    print(f"[DEBUG] DEM Nodata Value: {nodata}")
                    
                    valid_mask = (out_image[0] != nodata) & (~np.isnan(out_image[0]))
                    valid_pixels = out_image[0][valid_mask]
                    
                    # Sanity check: Filter extremely low values that might be legacy nodata
                    valid_pixels = valid_pixels[valid_pixels > -10000] 
                    
                    if valid_pixels.size > 0:
                        # ROBUST SHORELINE DETECTION (V3 - Conservative)
                        # Reject top 5% (Land/Noise) and bottom 85% (Bed)
                        p85 = np.percentile(valid_pixels, 85)
                        p95 = np.percentile(valid_pixels, 95)
                        
                        shoreline_pixels = valid_pixels[(valid_pixels >= p85) & (valid_pixels <= p95)]
                        
                        if shoreline_pixels.size > 0:
                             current_water_level = float(np.median(shoreline_pixels))
                             print(f"[DEBUG] Detected Water Level (P85-P95 Median): {current_water_level:.2f} m")
                        else:
                             current_water_level = float(np.percentile(valid_pixels, 90))
                             print(f"[DEBUG] Fallback Water Level (P90): {current_water_level:.2f} m")

                    else:
                        print("[DEBUG] No valid DEM pixels found under water mask.")
            except Exception as e:
                print(f"[ERROR] Failed to detect water level from mask: {e}")

        # B. Calculate Full Basin Curve & Lookup
        try:
            # Pass polygon_path (calculated above) to volume logic
            # Use Area-Based Lookup for robust volume
            volume_data = calculate_lake_volume(
                dem_path, 
                polygon_path, 
                base_level=base_level, 
                current_level=current_water_level, # Still pass detected level as debug/backup
                target_area_ha=area_ha # PRIMARY lookup
            )
            
            # Update current_water_level if it was recalculated from Area
            if volume_data.get("water_level_recalculated"):
                current_water_level = volume_data.get("water_level_recalculated")
            
            # Save Volume Curve CSV if returned
            if "curve_df" in volume_data:
                df_curve = volume_data.pop("curve_df")
                curve_path = os.path.join(output_dir, f"{lake_id}_{date_str}_volume_curve.csv")
                df_curve.to_csv(curve_path, index=False)
                print(f"[DEBUG] Saved volume curve to {curve_path}")
        except Exception as e:
             print(f"[ERROR] Full Basin Volume Calculation failed: {e}")


    # 3. Construct Final Result
    # "volume_m3" = Volume at Current Water Level (Best Estimate)
    current_vol = volume_data.get("volume_at_current_level")
    # If None (e.g. no intersection), fallback to 0
    if current_vol is None: current_vol = 0.0
    
    volume_tmc = current_vol / 28316846.592
    
    final_res = {
        "area_ha": round(area_ha, 2),
        "water_level": round(current_water_level, 2) if current_water_level is not None else 0,
        "volume_m3": round(current_vol, 2),
        "volume_tmc": round(volume_tmc, 4),
        "min_elevation": round(float(volume_data["min_elevation"]), 2),
        "max_elevation": round(float(volume_data["max_elevation"]), 2),
        "message": "Analysis successful"
    }

    # If base level was requested, add it
    base_vol = volume_data.get("volume_at_base_level")
    if base_vol is not None:
         final_res["base_level"] = round(float(base_level), 2)
         final_res["volume_at_level_m3"] = round(float(base_vol), 2)
         final_res["volume_at_level_tmc"] = round(float(base_vol) / 28316846.592, 4)
    
    return final_res



