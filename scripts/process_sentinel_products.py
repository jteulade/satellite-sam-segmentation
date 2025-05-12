import os
import sys
import zipfile
import glob
from pathlib import Path
import time
from datetime import datetime
import logging
import hydra


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

# Add local SAM to Python path
sam_path = "/home/teulade/segment-anything"
if sam_path not in sys.path:
    sys.path.insert(0, sam_path)


try:
    from src.sentinel_preprocessing import preprocess_imagery
    from src.sam_satellite_processor import segment_satellite_imagery
    from src.polygon_merger import merge_overlapping_segments, concat_polygons
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

except ImportError as e:
    logging.error(f"Error importing required modules: {e}")
    logging.error(f"Current Python path: {sys.path}")
    logging.error(f"Project root: {project_root}")
    logging.error("\nPlease make sure you have:")
    logging.error("1. Installed all requirements from requirements.txt")
    logging.error("2. The project structure is correct with a 'src' directory containing the required modules")
    logging.error("3. You're running the script from the project root directory")
    sys.exit(1)

def unzip_sentinel_products(base_dir):
    """Unzip all Sentinel product zip files in the given directory."""
    start_time = time.time()
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting unzip process...")
    logging.info(f"Checking Sentinel products in {base_dir}")
    zip_files = glob.glob(os.path.join(base_dir, "**/*.zip"), recursive=True)
    
    failed_files = []

    for zip_path in zip_files:
        file_start_time = time.time()
        # Check if already unzipped
        output_dir = os.path.splitext(zip_path)[0]
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            logging.info(f"Directory {output_dir} already exists, skipping unzip...")
            continue
            
        logging.info(f"Unzipping {zip_path} to {output_dir}")
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
            logging.info(f"Unzipped {os.path.basename(zip_path)} in {time.time() - file_start_time:.2f} seconds")
        except (zipfile.BadZipFile, zipfile.LargeZipFile) as e:
            logging.error(f"ERROR: Failed to unzip {zip_path}: {str(e)}")
            failed_files.append(zip_path)
            # Clean up the partially created directory if it exists
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
            continue
        except Exception as e:
            logging.error(f"ERROR: Unexpected error while unzipping {zip_path}: {str(e)}")
            failed_files.append(zip_path)
            # Clean up the partially created directory if it exists
            if os.path.exists(output_dir) and not os.listdir(output_dir):
                os.rmdir(output_dir)
            continue
    
    if failed_files:
        logging.info("The following files failed to unzip:")
        for failed_file in failed_files:
            logging.error(f"- {failed_file}")
        logging.info("\nYou may want to check these files and re-download them if necessary.")
    
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unzip process completed in {time.time() - start_time:.2f} seconds")


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

def setup_sam_model(sam_config):
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
        points_per_side=sam_config.get("points_per_side"),
        points_per_batch=sam_config.get("points_per_batch"),
        pred_iou_thresh=sam_config.get("pred_iou_thresh"),
        stability_score_thresh=sam_config.get("stability_score_thresh"),
        crop_nms_thresh=sam_config.get("crop_nms_thresh"),
        crop_overlap_ratio=sam_config.get("crop_overlap_ratio"),
        crop_n_layers=sam_config.get("crop_n_layers"),
        min_mask_region_area=sam_config.get("min_mask_region_area")
    )
    
    return mask_generator

def process_sentinel_products(base_dir, year, n_samples=None, overwrite=False, sam_config=None):
    """Process all Sentinel products in the given directory."""
    total_start_time = time.time()
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Sentinel products processing...")
    
    # Get all tile directories
    tile_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]

    # Setup SAM model
    model_start_time = time.time()
    mask_generator = setup_sam_model(sam_config)
    logging.info(f"SAM model setup completed in {time.time() - model_start_time:.2f} seconds")
    
    for tile_dir in tile_dirs:
        tile_start_time = time.time()
        tile_id = os.path.basename(tile_dir)
        
        logging.info(f"Processing {(tile_dirs)} tiles...")
        
        # Process each quarter
        for quarter in range(1, 5):
            quarter_start_time = time.time()
            # Construct the correct quarter directory path
            quarter_dir = os.path.join(tile_dir, f"Sentinel-2_mosaic_{year}_Q{quarter}_{tile_id}_0_0")
            logging.info(f"Quarter directory: {quarter_dir}")
            if not os.path.exists(quarter_dir):
                logging.info(f"Quarter {quarter} not found for tile {tile_id}, skipping...")
                continue
                
            logging.info(f"Processing quarter {quarter}")
            
            # Step 1: Preprocess
            step_start_time = time.time()
            if not overwrite and is_preprocessing_done(quarter_dir):
                logging.info("Step 1: Preprocessing already done, skipping...")
            else:
                logging.info("Step 1: Preprocessing imagery...")
                preprocess_imagery(quarter_dir)
            logging.info(f"Preprocessing step completed in {time.time() - step_start_time:.2f} seconds")
            
            # Step 2: SAM Segmentation
            step_start_time = time.time()
            if not overwrite and is_sam_done(quarter_dir):
                logging.info("Step 2: SAM segmentation already done, skipping...")
            else:
                logging.info("Step 2: Running SAM segmentation...")
                segment_satellite_imagery(quarter_dir, mask_generator, n_samples=n_samples, random_seed=sum(map(ord, tile_id)))
            logging.info(f"SAM segmentation step completed in {time.time() - step_start_time:.2f} seconds")
            logging.info(f"Quarter {quarter} processing completed in {time.time() - quarter_start_time:.2f} seconds")
        

        # Step 3: Merge polygons for this tile (all quarters)
        step_start_time = time.time()
        if not overwrite and is_merging_done(tile_dir, quarter):
            logging.info("Step 3: Polygon merging already done for this tile, skipping...")
        else:
            logging.info("Step 3: Merging polygons for all quarters...")
            merge_overlapping_segments(tile_dir, list(range(1, 5)), year)
        logging.info(f"Polygon merging step completed in {time.time() - step_start_time:.2f} seconds")
        logging.info(f"Tile {tile_id} processing completed in {time.time() - tile_start_time:.2f} seconds")
    
    # Final step: Concatenate all polygons
    concat_start_time = time.time()
    logging.info("Concatenating all polygons...")
    concat_polygons(tile_dirs)
    logging.info(f"Polygon concatenation completed in {time.time() - concat_start_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    logging.info(f"Total processing completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")


@hydra.main(config_path="../conf", config_name="config",version_base=None)
def main(cfg : dict):
    script_start_time = time.time()
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script started")
    
    # Convert base_dir to string if it's not already
    if not isinstance(cfg.base_dir, str):
        cfg.base_dir = str(cfg.base_dir)

    if not cfg.base_dir or not os.path.isdir(cfg.base_dir):
        logging.error("base_dir is mandatory and must be defined.")
        sys.exit(1)

    if not cfg.year:
        try:
            cfg.year = int(os.path.basename(cfg.base_dir))
        except ValueError:
            logging.error("Year not provided and could not be inferred from base_dir. Please provide a valid year.")
            sys.exit(1)

    # Add local SAM to Python path
    sam_path = cfg.sam_path
    if sam_path not in sys.path:
        sys.path.insert(0, sam_path)

    # First unzip all products (if needed)
    unzip_sentinel_products(cfg.base_dir)
    
    # Then process all products with n_samples=10
    process_sentinel_products(cfg.base_dir, cfg.year, n_samples=10, overwrite=cfg.overwrite, sam_config=cfg.sam)
    
    total_script_time = time.time() - script_start_time
    logging.info(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script completed in {total_script_time:.2f} seconds ({total_script_time/3600:.2f} hours)")

if __name__ == "__main__":
    main()