import rasterio
from rasterio.windows import Window
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import os
from tqdm import tqdm
import logging
from src.polygon_merger import delete_files_in_directory


logger = logging.getLogger("logger")

def cumulative_count_cut(band, min_percentile=2, max_percentile=98):
    """Apply contrast enhancement stretch similar to QGIS"""
    new_band = band[band != 0]
    min_val = np.nanpercentile(new_band, min_percentile)
    max_val = np.nanpercentile(new_band, max_percentile)
    #print(min_val,max_val)
    return (band - min_val) / (max_val - min_val) * 255
    
# Définir les chemins vers les fichiers de bandes

def build_rgb_from_sentinel(sentinel_path, color_type='nrg'):
    """
    Create color composite from Sentinel-2 bands.
    
    Args:
        sentinel_path (str): Path to the Sentinel-2 bands directory
        color_type (str): Either 'rgb' for true color or 'nrg' for NIR-Red-Green
    Returns:
        str: Path to the output TIF
    """
    output_dir = os.path.join(sentinel_path, color_type)
    delete_files_in_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'sentinel_composite.tif')
    
    # Read Sentinel-2 bands
    with rasterio.open(os.path.join(sentinel_path, 'B02.tif')) as src_blue, \
         rasterio.open(os.path.join(sentinel_path, 'B03.tif')) as src_green, \
         rasterio.open(os.path.join(sentinel_path, 'B04.tif')) as src_red, \
         rasterio.open(os.path.join(sentinel_path, 'B08.tif')) as src_nir:
        
        blue = src_blue.read(1)
        green = src_green.read(1)
        red = src_red.read(1)
        nir = src_nir.read(1)
        
        # Stack bands based on color type
        if color_type == 'rgb':
            image = np.stack([
                cumulative_count_cut(red),
                cumulative_count_cut(green),
                cumulative_count_cut(blue),
            ], axis=0)
        else:  # nrg
            image = np.stack([
                cumulative_count_cut(nir),
                cumulative_count_cut(red),
                cumulative_count_cut(green),
            ], axis=0)
        
        profile = src_red.profile.copy()
        profile.update(count=3)
        
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(image)
    
    return output_path

def build_color_from_JP2(jp2_path, color_type='nrg'):
    """
    Create color composite from Pléiades JP2 file using windows.
    
    Args:
        jp2_path (str): Path to JP2 file
        color_type (str): Either 'rgb' for true color or 'nrg' for NIR-Red-Green
    Returns:
        str: Path to output TIF file
    """
    output_dir = os.path.join(os.path.dirname(jp2_path), color_type)
    delete_files_in_directory(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'pleiades_composite.tif')
    
    try:
        with rasterio.open(jp2_path) as src:
            profile = src.profile.copy()
            profile.update(
                count=3, 
                dtype='uint8',
                crs=rasterio.crs.CRS.from_epsg(4326),
                transform=src.transform,
                driver='GTiff',
                compress='LZW'
            )
            
            # Calculate optimal window size
            window_size = 2048
            width = src.width
            height = src.height
            
            # Define band order based on color type
            if color_type == 'rgb':
                band_order = [3, 2, 1]  # RGB
            else:  # nrg
                band_order = [4, 3, 2]  # NIR, Red, Green
            
            # First pass: calculate global statistics
            sample_size = min(1000000, width * height)  # Limit sample size for large images
            row_indices = np.random.randint(0, height, size=int(np.sqrt(sample_size)))
            col_indices = np.random.randint(0, width, size=int(np.sqrt(sample_size)))
            
            # Calculate statistics for NIR (4), Red (3), Green (2) bands
            global_stats = []
            for band_number in band_order:
                sample_data = src.read(band_number, window=Window(
                    min(col_indices), min(row_indices),
                    max(col_indices) - min(col_indices),
                    max(row_indices) - min(row_indices)
                ))
                sample_data = sample_data[sample_data != 0]  # Remove no-data values
                min_val = np.nanpercentile(sample_data, 2)
                max_val = np.nanpercentile(sample_data, 98)
                global_stats.append((min_val, max_val))
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                # Process image in windows
                for y in range(0, height, window_size):
                    for x in range(0, width, window_size):
                        window = Window(x, y,
                                     min(window_size, width - x),
                                     min(window_size, height - y))
                        
                        window_data = np.zeros((3, window.height, window.width), 
                                             dtype='uint8')
                        
                        # Read and process bands in NIR, Red, Green order
                        for i, band_number in enumerate(band_order):
                            band_window = src.read(band_number, window=window)
                            min_val, max_val = global_stats[i]
                            normalized = np.clip((band_window - min_val) / (max_val - min_val) * 255, 0, 255)
                            window_data[i] = normalized.astype('uint8')
                            del band_window
                        
                        dst.write(window_data, window=window)
                        del window_data
            
        return output_path
        
    except Exception as e:
        logger.error(f"Error processing JP2 file: {str(e)}")
        return None
    
def split_image_in_tiles(input_file, grid_size=10):
    """
    Split the input image into tiles based on specified grid size.
    
    Args:
        input_file (str): Path to the composite TIF file to split
        grid_size (int): Number of tiles per row/column (e.g., 10 creates a 10x10 grid)
    """
    with rasterio.open(input_file) as src:
        width = src.width
        height = src.height
        
        output_dir = os.path.dirname(input_file)  # Will be in the color_type directory
        tiles_dir = os.path.join(output_dir, f"tiles_{grid_size}x{grid_size}")
        delete_files_in_directory(tiles_dir)
        os.makedirs(tiles_dir, exist_ok=True)

        # Calculate dimensions for each sub-image
        tile_width = width // grid_size
        tile_height = height // grid_size

        total_tiles = grid_size * grid_size
        logger.info(f"Splitting image into {total_tiles} tiles ({grid_size}x{grid_size} grid)")
        
        # Loop through each quadrant with progress bar
        with tqdm(total=total_tiles, desc="Splitting image", unit="tile") as pbar:
            for row in range(grid_size):
                for col in range(grid_size):
                    # Calculate coordinates for each window
                    x_offset = col * tile_width
                    y_offset = row * tile_height
                    window = Window(x_offset, y_offset, tile_width, tile_height)

                    # Create filename for each sub-image
                    output_file = os.path.join(tiles_dir, f"tile_{row + 1}_{col + 1}.tif")

                    # Update profile for the sub-image
                    profile = src.profile.copy()
                    profile.update({
                        "width": tile_width,
                        "height": tile_height,
                        "transform": rasterio.windows.transform(window, src.transform)
                    })

                    # Read window data and save
                    with rasterio.open(output_file, "w", **profile) as dst:
                        dst.write(src.read(window=window))
                    
                    pbar.update(1)

    logger.info(f"Splitting complete. Sub-images are saved in: {tiles_dir}")


def preprocess_imagery(input_path, color_type='nrg'):
    """
    Generic preprocessing function for both Sentinel-2 and Pléiades imagery.
    
    Args:
        input_path (str): Path to either Sentinel-2 directory or Pléiades JP2 file
        color_type (str): Either 'rgb' for true color or 'nrg' for NIR-Red-Green
    """
    logger.info(f"Processing {input_path}")
    
    try:
        if input_path.endswith('.JP2'):
            composite_path = build_color_from_JP2(input_path, color_type)
        else:
            composite_path = build_rgb_from_sentinel(input_path, color_type)

        logger.info(f"Finish color composite build")
        split_image_in_tiles(composite_path)
        logger.info(f"Successfully processed {input_path}")
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")