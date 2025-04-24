import os
import glob
import geopandas as gpd
from tqdm import tqdm
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon

def clean_geometry(geom):
    """Clean a geometry to make it valid and ensure it's a polygon"""
    if geom is None:
        return None
        
    try:
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
        
        # Convert to polygon if it's not already
        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            # Try to convert to polygon using buffer(0)
            geom = geom.buffer(0)
            if geom.geom_type not in ['Polygon', 'MultiPolygon']:
                return None
                
        return geom
    except Exception as e:
        print(f"Warning: Error cleaning geometry: {str(e)}")
        return None

def update_segments_source():
    # Define paths
    sentinel_base_path = "/home/teulade/images/sentinel2"
    output_base_path = "/home/teulade/images/sentinel2_output"
    combined_dir = os.path.join(output_base_path, "combined_intersections")
    
    # Read existing segments
    print("Reading existing segments...")
    segments_path = os.path.join(combined_dir, "all_intersection_polygons.parquet")
    if not os.path.exists(segments_path):
        print(f"Error: Could not find segments file at {segments_path}")
        return
    
    segments_gdf = gpd.read_parquet(segments_path)
    print(f"Loaded {len(segments_gdf)} segments")
    
    # Clean geometries in the main segments GeoDataFrame
    print("Cleaning geometries in main segments...")
    segments_gdf['geometry'] = segments_gdf['geometry'].apply(clean_geometry)
    
    # Remove rows with None geometries
    segments_gdf = segments_gdf.dropna(subset=['geometry'])
    print(f"After removing invalid geometries: {len(segments_gdf)} segments")
    
    # Get all intersection parquet files to map source folders
    print("Mapping source folders...")
    source_mapping = {}
    for root, _, files in os.walk(output_base_path):
        for file in files:
            if file.startswith("intersection_") and file.endswith(".parquet"):
                folder_name = file.replace("intersection_", "").replace(".parquet", "")
                source_folder = os.path.join(sentinel_base_path, folder_name)
                source_mapping[folder_name] = source_folder
    
    # Create spatial index for efficient spatial queries
    print("Creating spatial index...")
    spatial_index = segments_gdf.sindex
    
    # Update source folders
    print("Updating source folders...")
    segments_gdf['source_folder'] = None  # Keep original name for Parquet
    
    # Process each source folder
    for folder_name, source_folder in tqdm(source_mapping.items(), desc="Processing source folders"):
        # Read the intersection file for this folder
        intersection_file = os.path.join(output_base_path, folder_name, "intersection", f"intersection_{folder_name}.parquet")
        if not os.path.exists(intersection_file):
            print(f"Warning: Could not find intersection file for {folder_name}")
            continue
            
        folder_gdf = gpd.read_parquet(intersection_file)
        
        # Clean geometries in the folder GeoDataFrame
        folder_gdf['geometry'] = folder_gdf['geometry'].apply(clean_geometry)
        
        # Remove rows with None geometries
        folder_gdf = folder_gdf.dropna(subset=['geometry'])
        
        # Find overlapping segments
        for idx, row in folder_gdf.iterrows():
            try:
                # Get potential matches using spatial index
                possible_matches = list(spatial_index.intersection(row.geometry.bounds))
                
                for match_idx in possible_matches:
                    try:
                        if segments_gdf.iloc[match_idx].geometry.intersects(row.geometry):
                            # If there's a significant overlap, update the source folder
                            intersection = segments_gdf.iloc[match_idx].geometry.intersection(row.geometry)
                            if intersection.area > 0.25 * segments_gdf.iloc[match_idx].geometry.area:
                                segments_gdf.at[match_idx, 'source_folder'] = source_folder
                    except Exception as e:
                        print(f"Warning: Error processing match {match_idx} for folder {folder_name}: {str(e)}")
                        continue
            except Exception as e:
                print(f"Warning: Error processing row {idx} for folder {folder_name}: {str(e)}")
                continue
    
    # Clean geometries one final time before saving
    print("Final geometry cleaning...")
    segments_gdf['geometry'] = segments_gdf['geometry'].apply(clean_geometry)
    
    # Remove any remaining rows with None geometries
    segments_gdf = segments_gdf.dropna(subset=['geometry'])
    
    # Print geometry type statistics
    print("\nGeometry type statistics:")
    print(segments_gdf.geometry.type.value_counts())
    
    # Save updated segments
    print("\nSaving updated segments...")
    # Save to Parquet with full column names
    segments_gdf.to_parquet(segments_path)
    
    # Create a copy for Shapefile with compatible column names
    shp_gdf = segments_gdf.copy()
    
    # Select only the columns we want to keep in the Shapefile
    columns_to_keep = ['geometry', 'source_folder', 'pixel_count']
    shp_gdf = shp_gdf[columns_to_keep]
    
    # Rename columns to be compatible with Shapefile format (10 char limit)
    column_mapping = {
        'source_folder': 'src_folder',
        'pixel_count': 'pix_count'
    }
    shp_gdf = shp_gdf.rename(columns=column_mapping)
    
    # Save to Shapefile with compatible column names
    shp_path = os.path.join(combined_dir, "all_intersection_polygons.shp")
    shp_gdf.to_file(shp_path)
    
    # Print statistics
    total_segments = len(segments_gdf)
    segments_with_source = segments_gdf['source_folder'].notna().sum()
    print(f"\nUpdate complete:")
    print(f"- Total segments: {total_segments}")
    print(f"- Segments with source folder: {segments_with_source}")
    print(f"- Segments without source folder: {total_segments - segments_with_source}")
    print("\nNote: Shapefile format has 10-character limit for field names.")
    print("Full field names are preserved in the Parquet file.")

if __name__ == "__main__":
    update_segments_source() 