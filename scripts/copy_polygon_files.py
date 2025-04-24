import os
import sys
import glob
import shutil
from pathlib import Path
import time
from datetime import datetime

def copy_polygon_files(source_dir, target_dir):
    """Copy all shapefile and parquet files while maintaining directory structure."""
    start_time = time.time()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting file copy...")
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Find all shapefiles and parquet files
    shapefiles = glob.glob(os.path.join(source_dir, "**/*.shp"), recursive=True)
    parquet_files = glob.glob(os.path.join(source_dir, "**/*.parquet"), recursive=True)
    
    # Process shapefiles and their associated files
    for shp_file in shapefiles:
        # Get the relative path from source directory
        rel_path = os.path.relpath(shp_file, source_dir)
        target_path = os.path.join(target_dir, rel_path)
        
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        # Copy the shapefile and its associated files
        base_name = os.path.splitext(shp_file)[0]
        for ext in ['.shp', '.dbf', '.shx', '.prj', '.cpg']:
            src_file = base_name + ext
            if os.path.exists(src_file):
                dst_file = os.path.join(target_dir, rel_path).replace('.shp', ext)
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")
                except Exception as e:
                    print(f"Error copying {src_file}: {str(e)}")
    
    # Process parquet files
    for parquet_file in parquet_files:
        # Get the relative path from source directory
        rel_path = os.path.relpath(parquet_file, source_dir)
        target_path = os.path.join(target_dir, rel_path)
        
        # Create target directory if it doesn't exist
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        
        try:
            shutil.copy2(parquet_file, target_path)
            print(f"Copied {parquet_file} to {target_path}")
        except Exception as e:
            print(f"Error copying {parquet_file}: {str(e)}")
    
    total_time = time.time() - start_time
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Copy completed in {total_time:.2f} seconds")

def main():
    source_dir = "/home/teulade/dataset_download/downloads/2023"
    target_dir = "/home/teulade/dataset_download/downloads_copy"
    
    copy_polygon_files(source_dir, target_dir)

if __name__ == "__main__":
    main() 