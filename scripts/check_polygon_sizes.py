import geopandas as gpd
import glob
import os
from tqdm import tqdm

def check_polygon_sizes(combined_dir):
    """Check sizes of polygons in combined intersections directory"""
    # Get all parquet files
    parquet_files = glob.glob(os.path.join(combined_dir, "*.parquet"))
    
    if not parquet_files:
        print("No parquet files found in combined directory")
        return
    
    # Read the combined file
    gdf = gpd.read_parquet(parquet_files[0])
    
    # Print CRS information
    print("\nOriginal CRS Information:")
    print(f"Current CRS: {gdf.crs}")
    print(f"CRS type: {type(gdf.crs)}")
    
    # Check if CRS is WGS84
    if gdf.crs.to_epsg() == 4326:
        print("✓ CRS is correctly set to EPSG:4326 (WGS84)")
    else:
        print(f"⚠ WARNING: CRS is not EPSG:4326 (WGS84), found {gdf.crs.to_epsg()} instead")
    
    # Reproject to UTM for accurate area calculations
    print("\nReprojecting to UTM zone 32N (EPSG:32632) for area calculations...")
    gdf_utm = gdf.to_crs("EPSG:32632")
    
    # Calculate areas in square meters
    gdf_utm['area_m2'] = gdf_utm.geometry.area
    
    # Calculate pixel count (assuming 9m² per pixel)
    pixel_area = 9.0  # Sentinel-2 pixel area in m²
    gdf_utm['pixel_count'] = gdf_utm['area_m2'] / pixel_area
    
    # Print statistics
    print("\nPolygon size statistics:")
    print(f"Total number of polygons: {len(gdf_utm)}")
    print(f"Minimum area: {gdf_utm['area_m2'].min():.2f} m² ({gdf_utm['pixel_count'].min():.2f} pixels)")
    print(f"Maximum area: {gdf_utm['area_m2'].max():.2f} m² ({gdf_utm['pixel_count'].max():.2f} pixels)")
    print(f"Mean area: {gdf_utm['area_m2'].mean():.2f} m² ({gdf_utm['pixel_count'].mean():.2f} pixels)")
    
    # Check for polygons smaller than 100 pixels
    small_polygons = gdf_utm[gdf_utm['pixel_count'] < 100]
    if len(small_polygons) > 0:
        print(f"\nWARNING: Found {len(small_polygons)} polygons smaller than 100 pixels!")
        print("Smallest polygons:")
        print(small_polygons.nsmallest(5, 'pixel_count')[['pixel_count', 'area_m2']])
    else:
        print("\nAll polygons are larger than 100 pixels ✓")

if __name__ == "__main__":
    combined_dir = "/home/teulade/images/sentinel2_output/combined_intersections"
    check_polygon_sizes(combined_dir) 