import os
import sys
import glob
import torch
from tqdm import tqdm
import rasterio
import numpy as np
import pandas as pd
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import geopandas as gpd
from shapely.geometry import Polygon
from pyproj import Transformer
import cv2
import shutil
from shapely.validation import make_valid

# Add local SAM to Python path
sam_path = "/home/teulade/segment-anything"
if sam_path not in sys.path:
    sys.path.insert(0, sam_path)

def cumulative_count_cut(band, min_percentile=2, max_percentile=98):
    """Apply contrast enhancement stretch similar to QGIS"""
    new_band = band[band != 0]
    min_val = np.nanpercentile(new_band, min_percentile)
    max_val = np.nanpercentile(new_band, max_percentile)
    return (band - min_val) / (max_val - min_val) * 255

def build_nrg_composite(tif_path, output_dir):
    """Build NRG composite from Sentinel-2 GeoTIFF bands"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'nrg_composite.tif')
    
    # Read Sentinel-2 bands from GeoTIFF
    with rasterio.open(tif_path) as src:
        # Read bands (1-based index)
        green = src.read(1)  # Band 1
        red = src.read(2)    # Band 2
        nir = src.read(3)    # Band 3
        
        # Stack bands in NIR-Red-Green order
        image = np.stack([
            cumulative_count_cut(nir),
            cumulative_count_cut(red),
            cumulative_count_cut(green),
        ], axis=0)
        
        profile = src.profile.copy()
        profile.update(count=3)
        
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(image)
    
    return output_path

def get_georeferenced_polygons_from_image(path, mask_generator):
    """Extract georeferenced polygons from a satellite GeoTIFF image using SAM"""
    with rasterio.open(path) as src:
        image = src.read()
        crs = src.crs
        transform = src.transform

    # Normalize image for model compatibility
    image = image.transpose(1, 2, 0)
    image = (image / 255.0).astype(np.float32)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    masks = mask_generator.generate(image)
    georeferenced_data = []

    for mask_data in masks:
        mask = mask_data['segmentation'].astype(np.uint8) * 255
        confidence = mask_data.get('predicted_iou', 1.0)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if len(contour) >= 3:
                polygon_points = contour.reshape(-1, 2)
                geo_points = []
                
                for x, y in polygon_points:
                    lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
                    geo_points.append((lon, lat))

                geo_polygon = Polygon(geo_points)
                if crs != "EPSG:4326":
                    geo_polygon = Polygon([transformer.transform(x, y) for x, y in geo_polygon.exterior.coords])
                
                georeferenced_data.append({
                    'geometry': geo_polygon,
                    'confidence': confidence
                })

    return georeferenced_data

def process_sentinel_folder(folder_path, output_base_path, mask_generator):
    """Process a single Sentinel-2 folder"""
    print(f"Processing folder: {folder_path}")
    
    # Get folder name (e.g., 8077_5007_13)
    folder_name = os.path.basename(folder_path)
    
    # Get all 2018 TIF files
    tif_files = glob.glob(os.path.join(folder_path, "2018_*.tif"))
    if not tif_files:
        print(f"No 2018 TIF files found in {folder_path}")
        return
    
    # Process each 2018 image
    for tif_file in tqdm(tif_files, desc="Processing images"):
        try:
            # Get the date part from the filename (e.g., 2018_01)
            date_part = os.path.splitext(os.path.basename(tif_file))[0]
            
            # Create output directory structure
            output_dir = os.path.join(output_base_path, folder_name, date_part)
            os.makedirs(output_dir, exist_ok=True)
            
            # Build NRG composite
            nrg_path = build_nrg_composite(tif_file, output_dir)
            
            # Get polygons using SAM
            polygons = get_georeferenced_polygons_from_image(nrg_path, mask_generator)
            
            if polygons:
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
                
                # Save results in the same output directory
                gdf.to_file(os.path.join(output_dir, f"polygons_{date_part}.shp"))
                gdf.to_parquet(os.path.join(output_dir, f"polygons_{date_part}.parquet"))
                
                print(f"Processed {tif_file}: {len(polygons)} polygons")
            
        except Exception as e:
            print(f"Error processing {tif_file}: {str(e)}")

def get_pixel_area(tif_path):
    """Get pixel area from GeoTIFF file"""
    with rasterio.open(tif_path) as src:
        transform = src.transform
        return abs(transform[0] * transform[4])

def compute_folder_intersection(folder_path, output_base_path, min_pixels=100):
    """Compute intersection of all polygons within a folder"""
    folder_name = os.path.basename(folder_path)
    print(f"Computing intersection for folder: {folder_name}")
    
    # Get all parquet files in the folder's subdirectories
    parquet_files = []
    for root, _, files in os.walk(os.path.join(output_base_path, folder_name)):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        print(f"No parquet files found for folder {folder_name}")
        return
    
    # Read all GeoDataFrames
    gdfs = []
    for parquet_file in parquet_files:
        try:
            gdf = gpd.read_parquet(parquet_file)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Error reading {parquet_file}: {str(e)}")
    
    if not gdfs:
        print(f"No valid GeoDataFrames found for folder {folder_name}")
        return
    
    # Combine all polygons
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs="EPSG:4326")
    print(f"After merging, number of segments: {len(combined_gdf)}")
    
    # Fix invalid geometries and clean them
    def clean_geometry(geom):
        if not geom.is_valid:
            # Try to make valid first
            geom = make_valid(geom)
            if not geom.is_valid:
                # If still invalid, try buffer(0) trick
                geom = geom.buffer(0)
                if not geom.is_valid:
                    # If still invalid, try a small positive buffer
                    geom = geom.buffer(0.1)
                    # Then clean with negative buffer
                    geom = geom.buffer(-0.1)
        return geom
    
    # Clean geometries before reprojection
    combined_gdf['geometry'] = combined_gdf['geometry'].apply(clean_geometry)
    
    # Get pixel area from the first 2018 TIF file in the folder
    tif_files = glob.glob(os.path.join(folder_path, "2018_*.tif"))
    if not tif_files:
        print(f"No 2018 TIF files found in {folder_path}")
        return
    
    pixel_area = get_pixel_area(tif_files[0])
    print(f"Using pixel area: {pixel_area} m²")
    
    # Reproject to UTM zone 32N (EPSG:32632) for accurate area calculations
    combined_gdf = combined_gdf.to_crs("EPSG:32632")
    
    # Clean geometries again after reprojection
    combined_gdf['geometry'] = combined_gdf['geometry'].apply(clean_geometry)
    
    # Filter and process polygons
    filtered_gdf = combined_gdf[combined_gdf.geometry.area / pixel_area >= min_pixels]
    print(f"After removing segments smaller than {min_pixels} pixels, number of segments: {len(filtered_gdf)}")
    
    # Sort by area
    filtered_gdf["area"] = filtered_gdf.geometry.area
    filtered_gdf = filtered_gdf.sort_values(by="area").reset_index(drop=True)
    
    # Create spatial index for efficient intersection computation
    spatial_index = filtered_gdf.sindex
    final_geometries = []
    processed_indices = set()
    
    # Counters for statistics
    count_no_intersection = 0
    count_intersection = 0
    count_skipped = 0
    count_errors = 0
    
    # Process polygons
    for i, row in tqdm(enumerate(filtered_gdf.iterrows()), total=len(filtered_gdf), desc="Computing intersections"):
        if i in processed_indices:
            count_skipped += 1
            continue
            
        geom = row[1].geometry
        confidence = row[1].confidence
        processed_indices.add(i)
        
        possible_matches_index = list(spatial_index.intersection(geom.bounds))
        found_intersection = False
        
        for j in possible_matches_index:
            if j in processed_indices:
                continue
            
            other_row = filtered_gdf.iloc[j]
            other_geom = other_row.geometry
            other_confidence = other_row.confidence
            
            try:
                # Clean both geometries before intersection
                clean_geom = clean_geometry(geom)
                clean_other_geom = clean_geometry(other_geom)
                
                # Try intersection with cleaned geometries
                intersection = clean_geom.intersection(clean_other_geom)
                
                if not intersection.is_empty:
                    # Clean the intersection result
                    intersection = clean_geometry(intersection)
                    
                    # Check if intersection is significant
                    if intersection.area > 0.25 * clean_geom.area:
                        if not found_intersection:
                            final_geometries.append({
                                'geometry': intersection,
                                'confidence': max(confidence, other_confidence),
                                'source_folder': folder_path
                            })
                        found_intersection = True
                        processed_indices.add(j)
                        count_intersection += 1
            except Exception as e:
                print(f"Error computing intersection: {str(e)}")
                count_errors += 1
                continue
        
        if not found_intersection:
            final_geometries.append({
                'geometry': geom,
                'confidence': confidence,
                'source_folder': folder_path
            })
            count_no_intersection += 1
    
    # Create final GeoDataFrame and reproject back to EPSG:4326
    intersection_gdf = gpd.GeoDataFrame(
        geometry=[item['geometry'] for item in final_geometries if isinstance(item['geometry'], Polygon)],
        data={
            'confidence': [item['confidence'] for item in final_geometries if isinstance(item['geometry'], Polygon)],
            'source_folder': [item['source_folder'] for item in final_geometries if isinstance(item['geometry'], Polygon)]
        },
        crs="EPSG:32632"
    )
    
    # Clean geometries one final time before reprojection
    intersection_gdf['geometry'] = intersection_gdf['geometry'].apply(clean_geometry)
    
    # Reproject back to EPSG:4326 for saving
    intersection_gdf = intersection_gdf.to_crs("EPSG:4326")
    
    # Print statistics
    print(f"\nStatistics:")
    print(f"- Polygons without significant intersection: {count_no_intersection}")
    print(f"- Polygons reduced to intersection (>25%): {count_intersection}")
    print(f"- Skipped polygons (already processed): {count_skipped}")
    print(f"- Intersection errors: {count_errors}")
    print(f"- Remaining polygons: {len(intersection_gdf)}")
    
    # Save intersection results
    intersection_dir = os.path.join(output_base_path, folder_name, "intersection")
    os.makedirs(intersection_dir, exist_ok=True)
    
    intersection_gdf.to_file(os.path.join(intersection_dir, f"intersection_{folder_name}.shp"))
    intersection_gdf.to_parquet(os.path.join(intersection_dir, f"intersection_{folder_name}.parquet"))
    
    print(f"Saved intersection results for {folder_name}: {len(intersection_gdf)} polygons")

def concat_intersection_polygons(output_base_path, polygons_name="all_intersection_polygons", min_pixels=100):
    """Concatenate all intersection polygons from all folders"""
    print("\nConcatenating all intersection polygons...")
    
    # Get all intersection parquet files
    parquet_files = []
    for root, _, files in os.walk(output_base_path):
        for file in files:
            if file.startswith("intersection_") and file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    
    if not parquet_files:
        print("No intersection parquet files found")
        return
    
    # Read all GeoDataFrames
    gdfs = []
    for parquet_file in tqdm(parquet_files, desc="Loading intersection files"):
        try:
            gdf = gpd.read_parquet(parquet_file)
            gdfs.append(gdf)
            print(f"Loaded {parquet_file}: {len(gdf)} polygons")
        except Exception as e:
            print(f"Error loading {parquet_file}: {str(e)}")
    
    if gdfs:
        # Combine all GeoDataFrames
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        print(f"\nTotal number of polygons before filtering: {len(combined_gdf)}")
        
        # Reproject to UTM for accurate area calculations
        combined_gdf = combined_gdf.to_crs("EPSG:32632")
        
        # Calculate areas in square meters
        combined_gdf['area_m2'] = combined_gdf.geometry.area
        
        # Calculate pixel count (assuming 9m² per pixel)
        pixel_area = 9.0  # Sentinel-2 pixel area in m²
        combined_gdf['pixel_count'] = combined_gdf['area_m2'] / pixel_area
        
        # Filter out small polygons
        combined_gdf = combined_gdf[combined_gdf['pixel_count'] >= min_pixels]
        print(f"Number of polygons after filtering (min {min_pixels} pixels): {len(combined_gdf)}")
        
        # Create output directory
        output_dir = os.path.join(output_base_path, "combined_intersections")
        os.makedirs(output_dir, exist_ok=True)
        
        # Reproject back to EPSG:4326 for saving
        combined_gdf = combined_gdf.to_crs("EPSG:4326")
        
        # Save combined results
        combined_gdf.to_parquet(os.path.join(output_dir, f"{polygons_name}.parquet"))
        combined_gdf.to_file(os.path.join(output_dir, f"{polygons_name}.shp"))
        
        print(f"\nSaved combined files to {output_dir}")
    else:
        print("No data found to merge")

def main():
    # Define input and output paths
    tif_file = "/home/teulade/images/sentinel2/6688_3456_13/2019_02.tif"
    output_base_path = "/home/teulade/images/sentinel2_output"
    
    # Initialize SAM model
    sam_checkpoint = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sam = sam_model_registry[model_type]()
    sam.load_state_dict(torch.load(sam_checkpoint))
    sam.to(device)
    
    # Initialize mask generator with specified parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        points_per_batch=64,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.6,
        crop_nms_thresh=0,
        crop_overlap_ratio=1,
        crop_n_layers=1,
        min_mask_region_area=20,
    )
    
    try:
        # Get the date part from the filename (e.g., 2019_02)
        date_part = os.path.splitext(os.path.basename(tif_file))[0]
        folder_name = os.path.basename(os.path.dirname(tif_file))
        
        # Create output directory structure
        output_dir = os.path.join(output_base_path, folder_name, date_part)
        os.makedirs(output_dir, exist_ok=True)
        
        # Build NRG composite
        nrg_path = build_nrg_composite(tif_file, output_dir)
        
        # Get polygons using SAM
        polygons = get_georeferenced_polygons_from_image(nrg_path, mask_generator)
        
        if polygons:
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame(polygons, crs="EPSG:4326")
            
            # Save results in the same output directory
            gdf.to_file(os.path.join(output_dir, f"polygons_{date_part}.shp"))
            gdf.to_parquet(os.path.join(output_dir, f"polygons_{date_part}.parquet"))
            
            print(f"Processed {tif_file}: {len(polygons)} polygons")
        else:
            print(f"No polygons found for {tif_file}")
            
    except Exception as e:
        print(f"Error processing {tif_file}: {str(e)}")

if __name__ == "__main__":
    main() 