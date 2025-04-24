import os
import sys
from pathlib import Path
import glob
from tqdm import tqdm

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.polygon_merger import merge_overlapping_segments, concat_polygons

def main():
    # Base directory containing the Sentinel products
    base_dir = "/home/teulade/dataset_download/downloads/2023"
    year = 2023
    use_additional = True  # Set this to True to use additional polygons
    
    # Get all tile directories
    tile_dirs = [d for d in glob.glob(os.path.join(base_dir, "*")) if os.path.isdir(d)]
    
    # Process each tile
    for tile_dir in tqdm(tile_dirs, desc="Processing tiles"):
        tile_id = os.path.basename(tile_dir)
        print(f"\nProcessing tile: {tile_id}")
        
        # Create the new intersection polygons directory if using additional polygons
        if use_additional:
            intersection_dir = os.path.join(tile_dir, "nrg", "intersection_polygons2")
            os.makedirs(intersection_dir, exist_ok=True)
        
        # Merge polygons for all quarters
        merge_overlapping_segments(tile_dir, list(range(1, 5)), year, use_additional=use_additional)
    
    # Concatenate all polygons and save to new location
    output_dir = "/home/teulade/dataset_download/downloads/2023/merged_polygons2"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all intersection polygon files
    intersection_files = []
    for tile_dir in tile_dirs:
        parquet_path = os.path.join(
            tile_dir,
            "nrg",
            "intersection_polygons2" if use_additional else "intersection_polygons",
            f"{os.path.basename(tile_dir)}_intersection.parquet"
        )
        if os.path.exists(parquet_path):
            intersection_files.append(parquet_path)
    
    if intersection_files:
        # Concatenate all polygons
        concat_polygons(
            [os.path.dirname(os.path.dirname(os.path.dirname(f))) for f in intersection_files],
            color_type='nrg',
            grid_size=10,
            polygons_name="all_polygons",
            use_additional=use_additional
        )
        print(f"\nMerged polygons saved to {output_dir}")
    else:
        print("No intersection files found to merge")

if __name__ == "__main__":
    main() 