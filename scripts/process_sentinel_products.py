import os
import sys
import zipfile
import glob
from pathlib import Path
import time
from datetime import datetime
import numpy as np

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
    print(f"Error importing required modules: {e}")
    print(f"Current Python path: {sys.path}")
    print(f"Project root: {project_root}")
    print("\nPlease make sure you have:")
    print("1. Installed all requirements from requirements.txt")
    print("2. The project structure is correct with a 'src' directory containing the required modules")
    print("3. You're running the script from the project root directory")
    sys.exit(1)

def unzip_sentinel_products(base_dir):
    """Unzip all Sentinel product zip files in the given directory."""
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting unzip process...")
    print(f"Checking Sentinel products in {base_dir}")
    zip_files = glob.glob(os.path.join(base_dir, "**/*.zip"), recursive=True)
    
    for zip_path in zip_files:
        file_start_time = time.time()
        # Check if already unzipped
        output_dir = os.path.splitext(zip_path)[0]
        if os.path.exists(output_dir) and os.path.isdir(output_dir):
            print(f"Directory {output_dir} already exists, skipping unzip...")
            continue
            
        print(f"Unzipping {zip_path} to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            first_dir = zip_ref.namelist()[0].split('/')[0]
            zip_ref.extractall(output_dir)
            nested_dir = os.path.join(output_dir, first_dir)
            if os.path.exists(nested_dir) and os.path.isdir(nested_dir):
                for item in os.listdir(nested_dir):
                    os.rename(os.path.join(nested_dir, item), os.path.join(output_dir, item))
                os.rmdir(nested_dir)
        print(f"Unzipped {os.path.basename(zip_path)} in {time.time() - file_start_time:.2f} seconds")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Unzip process completed in {time.time() - start_time:.2f} seconds")
    return output_dir

def is_preprocessing_done(quarter_dir):
    """Check if preprocessing has already been done."""
    # Check for color composite and tiles
    color_composite = os.path.join(quarter_dir, "nrg", "sentinel_composite.tif")
    tiles_dir = os.path.join(quarter_dir, "nrg", "tiles_10x10")
    return os.path.exists(color_composite) and os.path.exists(tiles_dir)

def is_sam_done(quarter_dir, use_existing_selections=False):
    """Check if SAM segmentation has already been done."""
    # Check for SAM output files
    if use_existing_selections:
        parquet_file = os.path.join(quarter_dir, "nrg", "polygons_10x10_additional.parquet")
        shapefile_dir = os.path.join(quarter_dir, "nrg", "shapefiles_10x10_additional")
    else:
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

def process_sentinel_products(base_dir, year, n_samples=None, use_existing_selections=False):
    """Process all Sentinel products in the given directory.
    
    Args:
        base_dir: Base directory containing Sentinel products
        year: Year of the Sentinel products
        n_samples: Number of quadrants to randomly sample
        use_existing_selections: If True, will use existing quadrant selections from previous run
    """
    total_start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Sentinel products processing...")
    
    # Get all tile directories
    tile_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    # Setup SAM model
    model_start_time = time.time()
    mask_generator = setup_sam_model()
    print(f"SAM model setup completed in {time.time() - model_start_time:.2f} seconds")
    
    for tile_dir in tile_dirs:
        tile_start_time = time.time()
        tile_id = os.path.basename(tile_dir)
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing tile: {tile_id}")
        
        # Process each quarter
        for quarter in range(1, 5):
            quarter_start_time = time.time()
            # Construct the correct quarter directory path
            quarter_dir = os.path.join(tile_dir, f"Sentinel-2_mosaic_{year}_Q{quarter}_{tile_id}_0_0")
            if not os.path.exists(quarter_dir):
                print(f"Quarter {quarter} not found for tile {tile_id}, skipping...")
                continue
                
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing quarter {quarter}")
            
            # Step 1: Preprocess
            step_start_time = time.time()
            if is_preprocessing_done(quarter_dir):
                print("Step 1: Preprocessing already done, skipping...")
            else:
                print("Step 1: Preprocessing imagery...")
                preprocess_imagery(quarter_dir)
            print(f"Preprocessing step completed in {time.time() - step_start_time:.2f} seconds")
            
            # Step 2: SAM Segmentation
            step_start_time = time.time()
            if is_sam_done(quarter_dir, use_existing_selections):
                print("Step 2: SAM segmentation already done, skipping...")
            else:
                print("Step 2: Running SAM segmentation...")
                # Generate all possible quadrants
                grid_size = 10
                all_quadrants = [(row + 1, col + 1) for row in range(grid_size) for col in range(grid_size)]
                
                # Set up seed for reproducibility
                seed = sum(map(ord, tile_id))
                np.random.seed(seed)
                
                # Handle quadrant selection based on use_existing_selections parameter
                if use_existing_selections:
                    # Since we've only run once before, we know we processed n_samples quadrants
                    n_previous_selections = n_samples
                    previous_selections = np.random.choice(len(all_quadrants), size=n_previous_selections, replace=False)
                    print(f"Using existing selections: {previous_selections}")
                    segment_satellite_imagery(quarter_dir, mask_generator, n_samples=n_samples, random_seed=seed, exclude_selections=previous_selections, use_existing_selections=True)
                else:
                    print("Starting fresh selection of quadrants")
                    segment_satellite_imagery(quarter_dir, mask_generator, n_samples=n_samples, random_seed=seed)
            print(f"SAM segmentation step completed in {time.time() - step_start_time:.2f} seconds")
            print(f"Quarter {quarter} processing completed in {time.time() - quarter_start_time:.2f} seconds")
        
        # Step 3: Merge polygons for this tile (all quarters)
        # step_start_time = time.time()
        # if is_merging_done(tile_dir, quarter):
        #     print("Step 3: Polygon merging already done for this tile, skipping...")
        # else:
        #     print("Step 3: Merging polygons for all quarters...")
        #     merge_overlapping_segments(tile_dir, list(range(1, 5)), year)
        # print(f"Polygon merging step completed in {time.time() - step_start_time:.2f} seconds")
        
        print(f"Tile {tile_id} processing completed in {time.time() - tile_start_time:.2f} seconds")
    
    # Final step: Concatenate all polygons
    concat_start_time = time.time()
    print("\nConcatenating all polygons...")
    concat_polygons(tile_dirs, os.path.basename(base_dir))
    print(f"Polygon concatenation completed in {time.time() - concat_start_time:.2f} seconds")
    
    total_time = time.time() - total_start_time
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total processing completed in {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

if __name__ == "__main__":
    script_start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script started")
    
    base_dir = "/home/teulade/dataset_download/downloads/2023/"
    year = 2023  # Extract year from base_dir or specify it explicitly
    
    # First unzip all products (if needed)
    unzip_sentinel_products(base_dir)
    
    # Then process all products with n_samples=10
    # Set use_existing_selections=True to use existing quadrant selections
    process_sentinel_products(base_dir, year, n_samples=20, use_existing_selections=True)
    
    total_script_time = time.time() - script_start_time
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script completed in {total_script_time:.2f} seconds ({total_script_time/3600:.2f} hours)") 