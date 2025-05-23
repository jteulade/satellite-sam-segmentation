# Satellite SAM Segmentation

This project uses the Segment Anything Model (SAM) to segment satellite imagery from Sentinel products.

## Setup

### 1. Environment Setup

Create and activate a conda environment:

```bash
conda create -n sentinel python=3.11
conda activate sentinel
```

### 2. Install Dependencies

Install the required packages using conda and pip:

```bash
# Install core dependencies via conda
conda install -c conda-forge rasterio geopandas shapely tqdm matplotlib
conda install pytorch torchvision -c pytorch

# Install remaining dependencies via pip
pip install segment-anything opencv-python pillow
```

Or install all dependencies at once using pip:

```bash
pip install -r requirements.txt
```

### 3. Download SAM Model

The project requires the SAM model checkpoint. Download it using:

```bash
# Create models directory
mkdir -p models

# Download the SAM model checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth
```

The model file is approximately 2.4GB.

## Usage

The project provides scripts to process Sentinel satellite imagery:

1. Place your Sentinel product zip files in a directory (e.g., `/path/to/sentinel/products/`)
2. Run the processing script:

```bash
python scripts/process_sentinel_products.py base-dir="your_path_to_data"
```

### Key Arguments

| Argument                     | Description                                                               | Default                          |
|------------------------------|---------------------------------------------------------------------------|----------------------------------|
| `base-dir`                   | Path to the directory containing the Sentinel product zip files.          | *Required*                       |
| `overwrite`                  | Boolean flag to overwrite existing files and directories.                 | `false`                          |
| `sam-model-path`             | Path to the SAM model checkpoint file.                                    | `"models/sam_vit_h_4b8939.pth"`  |
| `year`                       | Year corresponding to the tiles being processed.                          | `2023`                           |
| `in-file`                    | Boolean flag to enable logging to a file.                                 | `false`                          |
| `sam.model`                  | Model name used by SAM.                                                   | `"sam"`                          |
| `sam.points_per_side`        | Number of points sampled per image side.                                  | `64`                             |
| `sam.points_per_batch`       | Number of points processed per batch.                                     | `192`                            |
| `sam.pred_iou_thresh`        | Prediction IoU threshold for mask filtering.                              | `0.6`                            |
| `sam.stability_score_thresh` | Threshold for the stability score.                                        | `0.6`                            |
| `sam.crop_nms_thresh`        | NMS threshold for cropping.                                               | `0`                              |
| `sam.crop_overlap_ratio`     | Overlap ratio between crops.                                              | `1`                              |
| `sam.crop_n_layers`          | Number of cropping layers.                                                | `1`                              |
| `sam.min_mask_region_area`   | Minimum area of a mask region to be considered.                           | `20`                             |

The script will:

1. Unzip all Sentinel product zip files
2. Preprocess the imagery
3. Run SAM segmentation
4. Merge and concatenate the resulting polygons

## Project Structure

```
satellite-sam-segmentation/
├── models/
│   └── sam_vit_h_4b8939.pth    # SAM model checkpoint
├── scripts/
│   └── process_sentinel_products.py
├── src/
│   ├── sentinel_preprocessing.py
│   ├── sam_satellite_processor.py
│   └── polygon_merger.py
└── requirements.txt
```

## Notes

- The SAM model checkpoint is required and must be placed in the `models/` directory
- The script expects Sentinel product zip files in the specified input directory
- Processing is done per tile and quarter
- Results are saved in the same directory structure as the input

## Configuration

### Supported Tile IDs

```python
tile_ids = [
"30UVU", "30TXT", "30TYR", "31TCJ",
"31TFJ", "31TGL", "31TDM", "31UGP",
"31UDP", "31UDR"
]
```

### SAM Parameters
- Model: `vit_h`
- Points per side: 150
- IoU threshold: 0.6
- Stability score threshold: 0.6
- Minimum mask region area: 50 pixels

## Output Structure
```
/home/teulade/images/Sentinel-2_mosaic_2022/
└── {tile_id}/
├── Sentinel-2_mosaic_2022_Q{1,2,3,4}{tile_id}_0_0/
│ ├── polygons.parquet
│ └── shapefiles/
│ └── polygons.shp
└── merge/
├── merge.shp
└── polygons.parquet
```


## Processing Pipeline

1. **Quarterly Processing**
   - Each quarter (Q1-Q4) is processed independently
   - Images are split into 100 tiles (10x10 grid)
   - SAM generates segmentation masks
   - Masks are converted to georeferenced polygons

2. **Merging Process**
   - Combines all quarterly results
   - Filters out segments smaller than 100 pixels
   - Handles overlapping polygons
   - Produces final clean polygon set

## Performance

Typical processing metrics:
- Input: 4 quarters × 100 tiles per quarter
- Initial segments: ~192,328 total
- Final output: ~60,607 clean polygons after merging

## Common Errors

Here are some common errors you might encounter while using this project and how to resolve them:

### 1. FileNotFoundError: SAM Model Not Found

**Error Message**:

```plaintext
Traceback (most recent call last):
  File "C:\Users\bilal\Documents\BUT2\satellite-sam-segmentation\scripts\process_sentinel_products.py", line 235, in <module>
    process_sentinel_products(base_dir, year, n_samples=10, overwrite=overwrite)
  File "C:\Users\bilal\Documents\BUT2\satellite-sam-segmentation\scripts\process_sentinel_products.py", line 156, in process_sentinel_products
    mask_generator = setup_sam_model()
                     ^^^^^^^^^^^^^^^^^
  File "C:\Users\bilal\Documents\BUT2\satellite-sam-segmentation\scripts\process_sentinel_products.py", line 129, in setup_sam_model
    sam = sam_model_registrymodel_type
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\bilal\anaconda3\Lib\site-packages\segment_anything\build_sam.py", line 15, in build_sam_vit_h
    return _build_sam()
           ^^^^^^^^^^^
  File "C:\Users\bilal\anaconda3\Lib\site-packages\segment_anything\build_sam.py", line 104, in _build_sam
    with open(checkpoint, "rb") as f:
         ^^^^^^^^^^^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\Users\\bilal\\Documents\\BUT2\\satellite-sam-segmentation\\models\\sam_vit_h_4b8939.pth'
```

Download the SAM model file :

```bash
# Create models directory
mkdir -p models

# Download the SAM model checkpoint
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O models/sam_vit_h_4b8939.pth
```

### 2. AssertionError: Torch not compiled with CUDA enabled

**Error Message**:
```plaintext
Traceback (most recent call last):
  File "C:\Users\bilal\Documents\BUT2\satellite-sam-segmentation\scripts\process_sentinel_products.py", line 242, in <module>
    process_sentinel_products(base_dir, year, n_samples=10, overwrite=overwrite)
  File "C:\Users\bilal\Documents\BUT2\satellite-sam-segmentation\scripts\process_sentinel_products.py", line 163, in process_sentinel_products
    mask_generator = setup_sam_model()
                     ^^^^^^^^^^^^^^^^^
  File "C:\Users\bilal\Documents\BUT2\satellite-sam-segmentation\scripts\process_sentinel_products.py", line 137, in setup_sam_model
    sam.to(device=device)
  File "C:\Users\bilal\anaconda3\envs\bon\Lib\site-packages\torch\nn\modules\module.py", line 1340, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\bilal\anaconda3\envs\bon\Lib\site-packages\torch\nn\modules\module.py", line 900, in _apply
    module._apply(fn)
  File "C:\Users\bilal\anaconda3\envs\bon\Lib\site-packages\torch\nn\modules\module.py", line 900, in _apply
    module._apply(fn)
  File "C:\Users\bilal\anaconda3\envs\bon\Lib\site-packages\torch\nn\modules\module.py", line 927, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "C:\Users\bilal\anaconda3\envs\bon\Lib\site-packages\torch\nn\modules\module.py", line 1326, in convert
    return t.to(
           ^^^^^
  File "C:\Users\bilal\anaconda3\envs\bon\Lib\site-packages\torch\cuda\__init__.py", line 310, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
```

You installed Torch without CUDA or with an incompatible version. Follow these steps:

Check CUDA Version:
Run the following command in your terminal to check your CUDA version:

```bash
nvcc --version
```

Install Correct Version of PyTorch:
Based on your CUDA version, install the correct version of PyTorch. Go to the official PyTorch installation page and select the appropriate version for your setup.

For example, for CUDA 11.3, use:

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/cu
```

### 3. Out of Memory Error

**Error Message**:
```plaintext
CUDA out of memory. Tried to allocate 1024.00 MiB. GPU 0 has a total capacity of 3.81 GiB of which 152.19 MiB is free. Including non-PyTorch memory, this process has 3.65 GiB memory in use. Of the allocated memory 2.59 GiB is allocated by PyTorch, and 1.01 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. See documentation for Memory Management (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

#### Reduce the workload:


You can **reduce the memory usage** of the SAM (Segment Anything Model) by decreasing the `points_per_side` parameter. Lowering this value reduces the number of input points, which in turn decreases the memory required for processing.

To change this parameter, simply add it as an override in your command:

```bash
python ./scripts/process_sentinel_products.py base_dir="2023" sam.points_per_side=32
```

## License

[Your chosen license]

## Contributors

[Your name/organization]