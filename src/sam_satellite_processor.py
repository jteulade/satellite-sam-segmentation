import rasterio
import numpy as np
import cv2
import os
import sys
import torch
import geopandas as gpd
from tqdm import tqdm

# Add local SAM to Python path
sam_path = "/home/teulade/segment-anything"
if sam_path not in sys.path:
    sys.path.insert(0, sam_path)

from segment_anything import SamAutomaticMaskGenerator
from shapely.geometry import Polygon
from pyproj import Transformer

    

def get_georeferenced_polygons_from_image(path, mask_generator: SamAutomaticMaskGenerator):
    """
    Extract georeferenced polygons from a satellite GeoTIFF image using a mask generator.
    Args:
        path (str): The file path to the satellite GeoTIFF image.
        mask_generator (SamAutomaticMaskGenerator): An instance of SamAutomaticMaskGenerator used to generate masks.
    Returns:
        list: A list of dictionaries containing georeferenced polygons and their confidence scores. 
              Each dictionary has the following keys:
              - 'geometry' (shapely.geometry.Polygon): The georeferenced polygon in EPSG:4326.
              - 'confidence' (float): The confidence score of the mask.
    """
    # Load the satellite GeoTIFF image
    with rasterio.open(path) as src:
        image = src.read()  # Read all image bands
        crs = src.crs  # Image coordinate system
        transform = src.transform  # Transform matrix for pixel to CRS

    # Normalize image for model compatibility
    image = image.transpose(1, 2, 0)
    image = (image / 255.0).astype(np.float32)
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    masks = mask_generator.generate(image)

    # Extract georeferenced polygons
    georeferenced_data = []
    # Extract and display polygon points for each mask
    for i, mask_data in enumerate(masks):
        mask = mask_data['segmentation'].astype(np.uint8) * 255  # Access binary mask and convert to 0 and 255 for OpenCV
        confidence = mask_data.get('predicted_iou', 1.0)  # Get confidence, default to 1.0 if missing

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Display polygon point coordinates for each contour
        for contour in contours:
            if len(contour) >= 3:  # Check if there are at least 3 points
                polygon_points = contour.reshape(-1, 2)  # Transform to (N, 2) format to get x, y points

                # Convert each point to geographic coordinates
                geo_points = []
                for x, y in polygon_points:
                    # Transform pixel coordinates to georeferenced coordinates
                    lon, lat = rasterio.transform.xy(transform, y, x, offset='center')
                    geo_points.append((lon, lat))

                # Transform polygon to EPSG:4326 if not the original CRS
                geo_polygon = Polygon(geo_points)
                if crs != "EPSG:4326":
                    geo_polygon = Polygon([transformer.transform(x, y) for x, y in geo_polygon.exterior.coords])
                
                georeferenced_data.append({
                    'geometry': geo_polygon,
                    'confidence': confidence
                })

    return georeferenced_data





def segment_satellite_imagery(sentinel_path, mask_generator: SamAutomaticMaskGenerator, color_type='nrg', grid_size=10, n_samples=None, random_seed=42, exclude_selections=None, use_existing_selections=False):
    """
    Process a tile using the provided mask generator.
    
    Args:
        sentinel_path: Full path to the tile directory
        mask_generator: Initialized SamAutomaticMaskGenerator instance
        color_type: Either 'rgb' for true color or 'nrg' for NIR-Red-Green
        grid_size: Number of tiles per row/column (default: 10)
        n_samples: Number of quadrants to randomly sample. If None, process all quadrants.
        random_seed: Random seed for reproducibility (only used if n_samples is not None)
        exclude_selections: List of indices to exclude from selection
        use_existing_selections: If True, will write results in a new directory
    """
    # Get the path to the color-specific directory
    color_dir = os.path.join(sentinel_path, color_type)
    if not os.path.exists(color_dir):
        print(f"Skipping {color_dir} - directory not found")
        return

    # Generate all possible quadrant combinations
    all_quadrants = [(row + 1, col + 1) for row in range(grid_size) for col in range(grid_size)]
    
    # Remove excluded quadrants if any
    if exclude_selections is not None:
        available_indices = [i for i in range(len(all_quadrants)) if i not in exclude_selections]
        if not available_indices:
            print("No available quadrants after excluding previously selected ones")
            return
        all_quadrants = [all_quadrants[i] for i in available_indices]
    
    # Select quadrants based on n_samples
    if n_samples is not None:
        np.random.seed(random_seed)
        # Ensure we don't try to select more quadrants than available
        n_samples = min(n_samples, len(all_quadrants))
        selected_quadrants = np.random.choice(len(all_quadrants), size=n_samples, replace=False)
        selected_quadrants = [all_quadrants[i] for i in selected_quadrants]
        print(f"Selected {n_samples} random quadrants: {selected_quadrants}")
    else:
        selected_quadrants = all_quadrants
        print(f"Processing all {len(selected_quadrants)} quadrants")
   
    print(f"Processing {color_dir}")
    georeferenced_polygons = []
    
    # Process selected quadrants
    with tqdm(total=len(selected_quadrants), desc=f"Processing {color_dir}", unit="quadrant") as pbar:
        for row, col in selected_quadrants:
            path = os.path.join(color_dir, f"tiles_{grid_size}x{grid_size}", f"tile_{row}_{col}.tif")
            if os.path.exists(path):
                try:
                    georeferenced_polygons.extend(get_georeferenced_polygons_from_image(path, mask_generator))
                except Exception as e:
                    print(f"Error processing {path}: {str(e)}")
            pbar.update(1)
    
    print(f"Found {len(georeferenced_polygons)} polygons for {color_dir}")
    
    # Before creating the GeoDataFrame, check if we have any polygons
    if not georeferenced_polygons:
        print(f"No polygons found for {color_dir}. Skipping GeoDataFrame creation.")
        return
    
    # Create output directories with different names if using existing selections
    if use_existing_selections:
        shapefile_dir = os.path.join(color_dir, f"shapefiles_{grid_size}x{grid_size}_additional")
        parquet_name = f"polygons_{grid_size}x{grid_size}_additional.parquet"
    else:
        shapefile_dir = os.path.join(color_dir, f"shapefiles_{grid_size}x{grid_size}")
        parquet_name = f"polygons_{grid_size}x{grid_size}.parquet"
    
    os.makedirs(shapefile_dir, exist_ok=True)
    
    # Save to Parquet
    gdf = gpd.GeoDataFrame(georeferenced_polygons, crs="EPSG:4326")
    gdf.to_parquet(os.path.join(color_dir, parquet_name))
    
    # Save to Shapefile
    output_shapefile = os.path.join(shapefile_dir, f"polygons_{grid_size}x{grid_size}.shp")
    gdf.to_file(output_shapefile, driver='ESRI Shapefile')

def segment_single_image(input_path: str, output_dir: str, mask_generator: SamAutomaticMaskGenerator):
    """
    Segment a single image and save the results.
    
    Args:
        input_path (str): Path to the input image file
        output_dir (str): Directory where to save the output files
        mask_generator (SamAutomaticMaskGenerator): Initialized SAM mask generator
    
    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the segmentation polygons
    """
    print(f"Processing {input_path}")
    
    try:
        # Get polygons from image
        georeferenced_polygons = get_georeferenced_polygons_from_image(input_path, mask_generator)
        print(f"Found {len(georeferenced_polygons)} polygons")
        
        # Get input filename without extension
        input_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # Create shapefile directory specific to this image
        shapefile_dir = os.path.join(output_dir, f"shapefiles_{input_name}")
        os.makedirs(shapefile_dir, exist_ok=True)
        
        # Save to GeoDataFrame
        gdf = gpd.GeoDataFrame(georeferenced_polygons, crs="EPSG:4326")
        
        # Save output with specific naming
        output_shapefile = os.path.join(shapefile_dir, f"polygons_{input_name}.shp")
        gdf.to_file(output_shapefile, driver='ESRI Shapefile')
        
        print(f"Results saved to {output_shapefile}")
        return gdf
        
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return None