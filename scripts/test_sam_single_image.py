import os
import sys
from pathlib import Path
import numpy as np
import rasterio
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import geopandas as gpd
import json

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.sam_satellite_processor import get_georeferenced_polygons_from_image

def setup_sam_model(params):
    """Initialize and return the SAM model and mask generator with the given parameters."""
    sam_checkpoint = os.path.join(project_root, "models", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "cuda"

    print(f"Loading SAM model from {sam_checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # Create mask generator with the provided parameters
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        **params
    )
    
    return mask_generator

def get_param_name(params):
    """Generate a name based on the parameters used."""
    # Create a short name for each parameter that is explicitly specified
    param_name = ""
    
    if "points_per_side" in params:
        param_name += f"ps{params['points_per_side']}"
    
    if "pred_iou_thresh" in params:
        param_name += f"_piou{params['pred_iou_thresh']}"
    
    if "stability_score_thresh" in params:
        param_name += f"_ss{params['stability_score_thresh']}"
    
    if "crop_nms_thresh" in params:
        param_name += f"_cnms{params['crop_nms_thresh']}"
    
    if "crop_overlap_ratio" in params:
        param_name += f"_cor{params['crop_overlap_ratio']}"
    
    if "crop_n_layers" in params:
        param_name += f"_cnl{params['crop_n_layers']}"
    
    if "min_mask_region_area" in params:
        param_name += f"_mmra{params['min_mask_region_area']}"
    
    return param_name

def save_results(image_path, output_dir, mask_generator, params):
    """Save results with parameter-specific naming."""
    # Get input filename without extension
    input_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Generate parameter name
    param_name = get_param_name(params)
    
    # Create shapefile directory specific to this image and parameter set
    shapefile_dir = os.path.join(output_dir, f"shapefiles_{input_name}_{param_name}")
    os.makedirs(shapefile_dir, exist_ok=True)
    
    # Get polygons from image
    georeferenced_polygons = get_georeferenced_polygons_from_image(image_path, mask_generator)
    print(f"Found {len(georeferenced_polygons)} polygons for parameters: {param_name}")
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(georeferenced_polygons, crs="EPSG:4326")
    
    # Save output with specific naming
    output_shapefile = os.path.join(shapefile_dir, f"polygons_{input_name}_{param_name}.shp")
    gdf.to_file(output_shapefile, driver='ESRI Shapefile')
    
    # Save parameters to a JSON file for reference
    params_file = os.path.join(shapefile_dir, f"parameters_{param_name}.json")
    with open(params_file, 'w') as f:
        json.dump(params, f, indent=4)
    
    print(f"Results saved to {output_shapefile}")
    print(f"Parameters saved to {params_file}")
    
    return gdf, len(georeferenced_polygons)

def main():
    try:
        # Define parameter sets to test
        parameter_sets = [
            {
                "name": "default",
                "params": {}  # Empty params to use SAM's built-in defaults
            },
            {
                "name": "original",
                "params": {
                    "points_per_side": 20,
                    "pred_iou_thresh": 0.6,
                    "stability_score_thresh": 0.6,
                    "crop_nms_thresh": 0,
                    "crop_overlap_ratio": 1,
                    "crop_n_layers": 1,
                    "min_mask_region_area": 20,
                }
            },
            {
                "name": "new",
                "params": {
                    "points_per_side": 20,
                    "pred_iou_thresh": 0.5,
                    "stability_score_thresh": 0.5,
                    "crop_nms_thresh": 0.3,
                    "crop_overlap_ratio": 0.8,
                    "crop_n_layers": 1,
                    "min_mask_region_area": 100,
                }
            }
            # Add more parameter sets here as needed
            
        ]
        
        # Load the image
        image_path = "/home/teulade/images/split_images/output_rgb_q5_3.tif"
        print(f"Loading image: {image_path}")
        
        # Create output directory
        output_dir = os.path.join(os.path.dirname(image_path), "results")
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each parameter set
        results = []
        for param_set in parameter_sets:
            print(f"\nProcessing parameter set: {param_set['name']}")
            
            # Setup SAM model with these parameters
            mask_generator = setup_sam_model(param_set['params'])
            
            # Generate and save results
            with tqdm(total=100, desc=f"Generating masks with {param_set['name']} parameters") as pbar:
                gdf, polygon_count = save_results(image_path, output_dir, mask_generator, param_set['params'])
                pbar.update(100)
            
            # Store results
            results.append({
                "name": param_set['name'],
                "params": param_set['params'],
                "polygon_count": polygon_count
            })
        
        # Print summary of all results
        print("\n=== SUMMARY OF RESULTS ===")
        for result in results:
            print(f"\n{result['name']} parameters:")
            print(f"  - Number of polygons: {result['polygon_count']}")
            print(f"  - Parameters: {json.dumps(result['params'], indent=2)}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 