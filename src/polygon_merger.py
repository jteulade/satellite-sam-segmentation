import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
import rasterio
from tqdm import tqdm
import os
import logging
from shapely.validation import make_valid  # Add this import
import shutil

logger = logging.getLogger("logger")

def delete_files_in_directory(directory):
    """Supprime les fichiers dans un dossier """
    if os.path.exists(directory):
        logging.info(f"Deleting files in {directory} ...")
        shutil.rmtree(directory)

def get_pixel_area(tile_path, quarters, year):
    # Get pixel area from any available quarter's B02.tif
    pixel_area = None
    for quarter in quarters:
        b02_path =os.path.join(
            tile_path,
            f"Sentinel-2_mosaic_{year}_Q{quarter}_{os.path.basename(tile_path)}_0_0",
            "B02.tif"
        )
        if os.path.exists(b02_path):
            with rasterio.open(b02_path) as src:
                transform = src.transform
                return abs(transform[0] * transform[4])
    return None


def create_intersection_gdf(filtered_gdf):
    """
    Create a GeoDataFrame of merged geometries based on intersections.

    Args:
        filtered_gdf: GeoDataFrame containing filtered polygons.

    Returns:
        result_gdf: GeoDataFrame of merged geometries.
    """
    spatial_index = filtered_gdf.sindex
    final_geometries = []
    processed_indices = set()
    
    # Counters
    count_no_intersection = 0
    count_intersection = 0
    count_skipped = 0
    
    # Process polygons
    for i, row in tqdm(enumerate(filtered_gdf.iterrows()), total=len(filtered_gdf), desc="Processing Polygons"):
        # Skip already processed polygons:
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
            
            intersection = geom.intersection(other_geom)
            if not intersection.is_empty and intersection.area > 0.25 * geom.area:
                if not found_intersection:
                    final_geometries.append({
                        'geometry': make_valid(intersection),
                        'confidence': max(confidence, other_confidence)
                    })
                found_intersection = True
                processed_indices.add(j)
                count_intersection += 1
        
        if not found_intersection:
            final_geometries.append({
                'geometry': geom,
                'confidence': confidence
            })
            count_no_intersection += 1
    
    # Create final GeoDataFrame
    intersection_gdf = gpd.GeoDataFrame(
        geometry=[item['geometry'] for item in final_geometries if isinstance(item['geometry'], Polygon)],
        data={'confidence': [item['confidence'] for item in final_geometries if isinstance(item['geometry'], Polygon)]},
        crs=filtered_gdf.crs
    )
    
    # Print statistics
    logger.info(f"Statistics:")
    logger.info(f"- Polygons without significant intersection: {count_no_intersection}")
    logger.info(f"- Polygons reduced to intersection (>25%): {count_intersection}")
    logger.info(f"- Skipped polygons (already processed): {count_skipped}")
    logger.info(f"- Remaining polygons: {len(intersection_gdf)}")
    
    return intersection_gdf

def merge_overlapping_segments(tile_path, quarters, year, color_type='nrg', grid_size=10, min_pixels=100):
    """
    Merge quarterly polygons for a single tile from a list of quarters.
    
    Args:
        tile_path: Full path to the tile directory
        quarters: List of quarters to process
        year: Year of the Sentinel data (e.g., 2023)
        color_type: Either 'rgb' for true color or 'nrg' for NIR-Red-Green
        grid_size: Grid size used for tiling (default: 10)
        min_pixels: Minimum number of pixels for a segment to be kept (default: 100)
    """
    tile_id = os.path.basename(tile_path)
    logger.info(f"Processing tile {tile_id} for quarters: {quarters}")
    pixel_area = get_pixel_area(tile_path, quarters, year)
    
    if pixel_area is None:
        logger.info(f"Could not find B02.tif for tile {tile_id}")
        return
    
    # Load quarterly parquet files with new path structure
    geodfs = []
    for quarter in quarters:
        quarter_path = os.path.join(
            tile_path,
            f"Sentinel-2_mosaic_{year}_Q{quarter}_{tile_id}_0_0",
            color_type,
            f"polygons_{grid_size}x{grid_size}.parquet"
        )
        if os.path.exists(quarter_path):
            geodfs.append(gpd.read_parquet(quarter_path).to_crs("EPSG:32632"))
    
    if not geodfs:
        logger.info(f"No parquet files found for tile {tile_id}")
        return
    
    # Concatenate GeoDataFrames
    combined_gdf = gpd.GeoDataFrame(pd.concat(geodfs, ignore_index=True), crs="EPSG:32632")
    
    # Fix invalid geometries
    combined_gdf['geometry'] = combined_gdf['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)
    logger.info(f"After merging, number of segments: {len(combined_gdf)}")
    
    # Filter and process polygons
    filtered_gdf = combined_gdf[combined_gdf.geometry.area / pixel_area >= min_pixels]
    logger.info(f"After removing segments smaller than {min_pixels} pixels, number of segments: {len(filtered_gdf)}")
    
    filtered_gdf["area"] = filtered_gdf.geometry.area
    filtered_gdf = filtered_gdf.sort_values(by="area").reset_index(drop=True)
    
    # Create result GeoDataFrame using the new function
    intersection_gdf = create_intersection_gdf(filtered_gdf)
    
    # Create merge directory in color-specific folder
    merge_path = os.path.join(tile_path, "intersection_polygons")
    os.makedirs(merge_path, exist_ok=True)
    
    intersection_gdf.to_file(os.path.join(merge_path, f"{tile_id}_intersection.shp"))
    intersection_gdf.to_parquet(os.path.join(merge_path, f"{tile_id}_intersection.parquet"))

def concat_polygons(tile_paths, color_type='nrg', grid_size=10, polygons_name="all_polygons"):
    gdfs = []
    for tile_path in tqdm(tile_paths, desc="Loading tiles"):
        parquet_path = os.path.join(
            tile_path, 
            "intersection_polygons", 
            f"{os.path.basename(tile_path)}_intersection.parquet"
        )
        if os.path.exists(parquet_path):
            try:
                gdf = gpd.read_parquet(parquet_path)
                gdfs.append(gdf)
                logger.info(f"Loaded {parquet_path}: {len(gdf)} polygons")
            except Exception as e:
                logger.error(f"Error loading {parquet_path}: {str(e)}")
        else:
            logger.info(f"No parquet file found for {parquet_path}")

    if gdfs:
        combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
        logger.info(f"Total number of polygons: {len(combined_gdf)}")
        
        # Create output directory in the same location as the input tiles
        output_dir = os.path.join(
            f"{polygons_name}_{color_type}_{grid_size}x{grid_size}"
        )
        delete_files_in_directory(output_dir) 
        os.makedirs(output_dir, exist_ok=True)
        
        # Save combined results
        combined_gdf.to_parquet(os.path.join(output_dir, f"{polygons_name}.parquet"))
        combined_gdf.to_file(os.path.join(output_dir, f"{polygons_name}.shp"))
        
        logger.info(f"Saved merged files to {output_dir}")
    else:
        logger.info("No data found to merge")