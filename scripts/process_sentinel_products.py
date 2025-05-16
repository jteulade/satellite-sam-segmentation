import os
import sys
import zipfile
import glob
from pathlib import Path
import time
from datetime import datetime
import argparse
import logging

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.logger import configure_logger

from src.sentinel_preprocessing import preprocess_imagery
from src.sam_satellite_processor import segment_satellite_imagery
from src.polygon_merger import merge_overlapping_segments, concat_polygons
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

def unzip_sentinel_products(base_dir):
    """Unzip all Sentinel product zip files in the given directory."""
    start_time = time.time()
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting unzip process...")
    logger.info(f"Checking Sentinel products in {base_dir}")
    zip_files = glob.glob(os.path.join(base_dir, "**/*.zip"), recursive=True)
    
    failed_files = []

    for zip_path in zip_files:
        file_start_time = time.time()
        # Check if already unzipped
        output_dir = os.path.splitext(zip_path)[0]
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            logger.info(f"Directory {output_dir} already exists, skipping unzip...")
            continue
            
        logger.info(f"Unzipping {zip_path} to {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                first_dir = zip_ref.namelist()[0].split('/')[0]
                zip_ref.extractall(output_dir)
                nested_dir = os.path.join(output_dir, first_dir)
                if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
                    for item in os.listdir(nested_dir):
                        os.rename(os.path.join(nested_dir, item), os.path.join(output_dir, item))
                    os.rmdir(nested_dir)
            logger.info(f"Unzipped {os.path.basename(zip_path)} in {time.time() - file_start_time:.2f} seconds")
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            logger.error(f"ERROR: Failed to unzip {zip_path}: {str(e)}")
            failed_files.append(zip_path)
            # Clean up the partially created directory if it exists
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
            continue
        except Exception as e:
            logger.error(f"ERROR: Unexpected error while unzipping {zip_path}: {str(e)}")
            failed_files.append(zip_path)
            # Clean up the partially created directory if it exists
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
            continue
    
    if failed_files:
        logger.info("The following files failed to unzip:")
        for failed_file in failed_files:
            logger.error(f"- {failed_file}")
        logger.info("You may want to check these files and re-download them if necessary.")
    
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unzip process completed in {time.time() - start_time:.2f} seconds")


def is_preprocessing_done(quarter_dir):
    """Check if preprocessing has already been done."""
    # Check for color composite and tiles
    color_composite = os.path.join(quarter_dir, "nrg", "sentinel_composite.tif")
    tiles_dir = os.path.join(quarter_dir, "nrg", "tiles_10x10")
    return os.path.exists(color_composite) and os.path.exists(tiles_dir)

def is_sam_done(quarter_dir):
    """Check if SAM segmentation has already been done."""
    # Check for SAM output files
    parquet_file = os.path.join(quarter_dir, "nrg", "polygons_10x10.parquet")
    shapefile_dir = os.path.join(quarter_dir, "nrg", "shapefiles_10x10")
    return os.path.exists(parquet_file) and os.path.exists(shapefile_dir)

def is_merging_done(tile_dir, quarter):
    """Check if polygon merging has already been done."""
    # Check for merged polygon files
    tile_id = os.path.basename(tile_dir)
    parquet_file = os.path.join(tile_dir, "nrg", "intersection_polygons", f"{tile_id}_intersection.parquet")
    shapefile = os.path.join(tile_dir, "nrg", "intersection_polygons", f"{tile_id}_intersection.shp")
    return os.path.exists(parquet_file) and os.path.exists(shapefile)

def setup_sam_model():
    """Initialize and return the SAM model and mask generator."""
    # Get the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sam_checkpoint = os.path.join(project_root, "models", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=64,
        points_per_batch=192,
        pred_iou_thresh=0.6,
        stability_score_thresh=0.6,
        crop_nms_thresh=0,
        crop_overlap_ratio=1,
        crop_n_layers=1,
        min_mask_region_area=20,
    )
    
    return mask_generator

def process_sentinel_products(base_dir, year, n_samples=None, overwrite=False):
    """Process all Sentinel products in the given directory."""
    total_start_time = time.time()
    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Sentinel products processing...")
    
    # Get all tile directories
    tile_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

    # Setup SAM model
    model_start_time = time.time()
    mask_generator = setup_sam_model()
    logger.info(f"SAM model setup completed in {time.time() - model_start_time:.2f} seconds")
    
    for tile_dir in tile_dirs:
        tile_start_time = time.time()
        tile_id = os.path.basename(tile_dir)
        
        logging.info(f"Processing {(tile_dirs)} tiles...")
        
        # Process each quarter
        for quarter in range(1, 5):
            quarter_start_time = time.time()
            # Construct the correct quarter directory path
            quarter_dir = os.path.join(tile_dir, f"Sentinel-2_mosaic_{year}_Q{quarter}_{tile_id}_0_0")
            logger.info(f"Quarter directory: {quarter_dir}")
            if not os.path.exists(quarter_dir):
                logger.info(f"Quarter {quarter} not found for tile {tile_id}, skipping...")
                continue
                
            logging.info(f"Processing quarter {quarter}")
            
            # Step 1: Preprocess
            step_start_time = time.time()
            if not overwrite and is_preprocessing_done(quarter_dir):
                logging.info("Step 1: Preprocessing already done, skipping...")
            else:
                logger.info("Step 1: Preprocessing imagery...")
                preprocess_imagery(quarter_dir)
            logger.info(f"Preprocessing step completed in {time.time() - step_start_time:.2f} seconds")
            
            # Step 2: SAM Segmentation
            step_start_time = time.time()
            if not overwrite and is_sam_done(quarter_dir):
                logging.info("Step 2: SAM segmentation already done, skipping...")
            else:
                logger.info("Step 2: Running SAM segmentation...")
                segment_satellite_imagery(quarter_dir, mask_generator, n_samples=n_samples, random_seed=sum(map(ord, tile_id)))
            logger.info(f"SAM segmentation step completed in {time.time() - step_start_time:.2f} seconds")
            logger.info(f"Quarter {quarter} processing completed in {time.time() - quarter_start_time:.2f} seconds")
        

        # Step 3: Merge polygons for this tile (all quarters)
        step_start_time = time.time()
        if not overwrite and is_merging_done(tile_dir, quarter):
            logging.info("Step 3: Polygon merging already done for this tile, skipping...")
        else:
            logger.info("Step 3: Merging polygons for all quarters...")
            merge_overlapping_segments(tile_dir, list(range(1, 5)), year)
        logger.info(f"Polygon merging step completed in {time.time() - step_start_time:.2f} seconds")
        logger.info(f"Tile {tile_id} processing completed in {time.time() - tile_start_time:.2f} seconds")
    
    # Final step: Concatenate all polygons
    concat_start_time = time.time()
    logging.info("Concatenating all polygons...")
    concat_polygons(tile_dirs)
    logger.info(f"Polygon concatenation completed in {time.time() - concat_start_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    logging.info(f"Total processing completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process Sentinel products")
    parser.add_argument("--base-dir", type=str, required=True, help="Path to the base directory with Tiles")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--sam-path", type=str, default="models/sam_vit_h_4b8939.pth", help="Path to the SAM model directory")
    parser.add_argument("--year", type=int, help="Year of the Sentinel data (e.g., 2023)")
    parser.add_argument("--in-file", help="boolean to enable file logging", action="store_true")
    
    return parser.parse_args()
if __name__ == "__main__":
    script_start_time = time.time()

    args = parse_args()
    logger = configure_logger(is_file=args.in_file)

    logger.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script started")

    if not os.path.isdir(args.base_dir):
        logger.error(f"The provided base directory does not exist or is not a directory: {args.base_dir}")
        sys.exit(1)

    if not args.year:
        try:
            args.year = int(os.path.basename(args.base_dir))
        except ValueError:
            logger.error("Year not provided and could not be inferred from base_dir. Please provide a valid year.")
            sys.exit(1)

    # Add local SAM to Python path
    sam_path = args.sam_path
    if sam_path not in sys.path:
        sys.path.insert(0, sam_path)

    # First unzip all products (if needed)
    unzip_sentinel_products(args.base_dir)
    
    # Then process all products with n_samples=10
    process_sentinel_products(args.base_dir, args.year, n_samples=10,overwrite=args.overwrite)
    
    total_script_time = time.time() - script_start_time
    logger.info(f" Script completed in {total_script_time:.2f} seconds ({total_script_time/3600:.2f} hours)") 