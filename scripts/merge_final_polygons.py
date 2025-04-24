import os
import sys
from pathlib import Path
import glob
from tqdm import tqdm
import geopandas as gpd
import pandas as pd

# Add the project root directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

def merge_final_polygons():
    # Input directories
    additional_dir = "/home/teulade/dataset_download/downloads/2023/all_polygons_nrg_10x10_additional"
    original_dir = "/home/teulade/dataset_download/downloads/2023/merged_polygons"
    
    # Output directory
    output_dir = "/home/teulade/dataset_download/downloads/2023/final_merged_polygons"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load polygons from both directories
    print("Loading polygons from additional directory...")
    additional_gdf = gpd.read_parquet(os.path.join(additional_dir, "all_polygons.parquet"))
    print(f"Loaded {len(additional_gdf)} polygons from additional directory")
    
    print("\nLoading polygons from original directory...")
    original_gdf = gpd.read_parquet(os.path.join(original_dir, "intersection_polygons.parquet"))
    print(f"Loaded {len(original_gdf)} polygons from original directory")
    
    # Concatenate both GeoDataFrames
    print("\nMerging polygons...")
    combined_gdf = gpd.GeoDataFrame(pd.concat([additional_gdf, original_gdf], ignore_index=True))
    print(f"Total number of polygons after merging: {len(combined_gdf)}")
    
    # Save merged results
    print("\nSaving merged results...")
    combined_gdf.to_parquet(os.path.join(output_dir, "all_polygons.parquet"))
    combined_gdf.to_file(os.path.join(output_dir, "all_polygons.shp"))
    print(f"\nSaved merged files to {output_dir}")

if __name__ == "__main__":
    merge_final_polygons() 