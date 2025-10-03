# ClipCap Training with DINO Features - README

This guide provides instructions for training ClipCap with DINO visual features instead of CLIP features.

## Prerequisites

1. Ensure you have the required dependencies installed:
   - PyTorch
   - torchvision
   - transformers
   - tqdm
   - Pillow
   - scikit-image

2. Prepare your COCO dataset with the following structure:
   ```
   ./data/coco/
   ├── annotations/
   │   └── train_caption.json
   ├── train2014/
   │   └── COCO_train2014_*.jpg
   └── val2014/
       └── COCO_val2014_*.jpg
   ```

## Required Files for DINO Feature Extraction

To start the DINO feature extraction for the COCO dataset, you need:

### 1. **COCO Dataset Structure**:
```
/raid/datasets/coco/                     # Main COCO directory (default)
├── train2014/                           # REQUIRED: Training images
│   └── COCO_train2014_*.jpg            # Image files
├── val2014/                             # REQUIRED: Validation images
│   └── COCO_val2014_*.jpg               # Image files
└── train_split_karpathy.json            # REQUIRED: Karpathy format annotations (default)
```

### 2. **Required Files**:
- **`train_split_karpathy.json`**: COCO caption annotations in Karpathy format (default)
- **Training images**: COCO 2014 training set (COCO_train2014_*.jpg)
- **Validation images**: COCO 2014 validation set (COCO_val2014_*.jpg)

### 3. **Annotation Format Support**:

The script supports two annotation formats:

#### **A. Karpathy Format** (default, recommended):
```json
{
    "images": [
        {"id": 522418, "file_name": "COCO_val2014_000000522418.jpg"}
    ],
    "annotations": [
        {"image_id": 522418, "id": 0, "caption": "A woman wearing a net..."}
    ]
}
```

#### **B. ClipCap Format** (legacy):
```json
[
    {"image_id": 522418, "caption": "A woman wearing a net..."}
]
```

### 3. **Specifying Custom Input/Output Paths**:

You can customize the paths using command-line arguments:

```bash
python clipcap_dino_parse_coco.py \
    --dino_model_type dinov2_vitb14 \
    --coco_images_dir "/path/to/your/coco/dataset" \
    --captions_file "/path/to/your/train_caption.json" \
    --output_file "/path/to/output/dino_features.pkl"
```

**Available path arguments**:
- `--coco_images_dir`: Path to COCO images directory (should contain `train2014/` and `val2014/` subdirs) - **Default: `/raid/datasets/coco`**
- `--captions_file`: Path to COCO captions JSON file (supports both Karpathy and ClipCap formats) - **Default: `/raid/datasets/coco/train_split_karpathy.json`**
- `--output_file`: Custom output file path (optional, auto-generated if not specified)

### 4. **Default Behavior** (if no paths specified):
```bash
# This will use default paths for your setup:
python clipcap_dino_parse_coco.py --dino_model_type dinov2_vitb14

# Equivalent to:
python clipcap_dino_parse_coco.py \
    --dino_model_type dinov2_vitb14 \
    --coco_images_dir "/raid/datasets/coco" \
    --captions_file "/raid/datasets/coco/train_split_karpathy.json" \
    --output_file "/raid/datasets/coco/coco_karpathy_split_dinov2_vitb14_train.pkl"
```

## Step 1: Extract DINO Features

First, extract DINO features from the COCO images using the modified feature extraction script:

### For DINOv2-B/14 (768-dim features):
```bash
# Default paths (uses /raid/datasets/coco and Karpathy annotations)
python clipcap_dino_parse_coco.py --dino_model_type dinov2_vitb14 --resize_dim 518 --crop_dim 518

# Custom paths
python clipcap_dino_parse_coco.py \
    --dino_model_type dinov2_vitb14 \
    --coco_images_dir "/your/coco/path" \
    --captions_file "/your/coco/train_split_karpathy.json" \
    --output_file "/your/output/dino_vitb14_features.pkl"
```

### For DINOv2-L/14 (1024-dim features):
```bash
# Default paths
python clipcap_dino_parse_coco.py --dino_model_type dinov2_vitl14 --resize_dim 518 --crop_dim 518

# Custom paths
python clipcap_dino_parse_coco.py \
    --dino_model_type dinov2_vitl14 \
    --coco_images_dir "/your/coco/path" \
    --output_file "/your/output/dino_vitl14_features.pkl"
```

### For DINOv2-S/14 (384-dim features):
```bash
python clipcap_dino_parse_coco.py --dino_model_type dinov2_vits14 --resize_dim 518 --crop_dim 518
```

### For DINOv2-G/14 (1536-dim features):
```bash
python clipcap_dino_parse_coco.py --dino_model_type dinov2_vitg14 --resize_dim 518 --crop_dim 518
```

**Output**: This will create a file like `/raid/datasets/models_weights/clipcap/training-features/coco_karpathy_split_dinov2_vitb14_train.pkl` (or your custom path) containing the DINO features and captions.

### Check Available Arguments:
```bash
python clipcap_dino_parse_coco.py --help
```

## Step 2: Train ClipCap with DINO Features

### Basic Training Command (MLP with sequence length 10):

For **DINOv2-B/14** with **MLP mapping** and **prefix length 10**:
```bash
python clipcapTraining.py \
    --data /raid/datasets/models_weights/clipcap/training-features/coco_karpathy_split_dinov2_vitb14_train.pkl \
    --out_dir ./checkpoints_dino_vitb14_mlp_len10 \
    --prefix dino_vitb14_mlp_len10 \
    --epochs 10 \
    --save_every 2 \
    --prefix_length 10 \
    --bs 32 \
    --mapping_type mlp \
    --use_dino \
    --dino_model_type dinov2_vitb14 \
    --only_prefix
```

### Training Options for Different DINO Models:

#### DINOv2-L/14 (1024-dim):
```bash
python clipcapTraining.py \
    --data ./data/coco/coco_karpathy_split_dinov2_vitl14_train.pkl \
    --out_dir ./checkpoints_dino_vitl14_mlp_len10 \
    --prefix dino_vitl14_mlp_len10 \
    --epochs 10 \
    --save_every 2 \
    --prefix_length 10 \
    --bs 32 \
    --mapping_type mlp \
    --use_dino \
    --dino_model_type dinov2_vitl14 \
    --only_prefix
```

#### DINOv2-S/14 (384-dim):
```bash
python clipcapTraining.py \
    --data ./data/coco/coco_karpathy_split_dinov2_vits14_train.pkl \
    --out_dir ./checkpoints_dino_vits14_mlp_len10 \
    --prefix dino_vits14_mlp_len10 \
    --epochs 10 \
    --save_every 2 \
    --prefix_length 10 \
    --bs 32 \
    --mapping_type mlp \
    --use_dino \
    --dino_model_type dinov2_vits14 \
    --only_prefix
```

### Advanced Training Options:

#### Train both prefix and GPT (full model):
```bash
python clipcapTraining.py \
    --data /raid/datasets/models_weights/clipcap/training-features/coco_karpathy_split_dinov2_vitb14_train.pkl \
    --out_dir ./checkpoints_dino_vitb14_mlp_len10_full \
    --prefix dino_vitb14_mlp_len10_full \
    --epochs 10 \
    --save_every 2 \
    --prefix_length 10 \
    --bs 16 \
    --mapping_type mlp \
    --use_dino \
    --dino_model_type dinov2_vitb14
```

#### Use Transformer mapping instead of MLP:
```bash
python clipcapTraining.py \
    --data /raid/datasets/models_weights/clipcap/training-features/coco_karpathy_split_dinov2_vitb14_train.pkl \
    --out_dir ./checkpoints_dino_vitb14_transformer_len10 \
    --prefix dino_vitb14_transformer_len10 \
    --epochs 10 \
    --save_every 2 \
    --prefix_length 10 \
    --bs 32 \
    --mapping_type transformer \
    --num_layers 8 \
    --use_dino \
    --dino_model_type dinov2_vitb14 \
    --only_prefix
```

#### Custom feature dimension (if needed):
```bash
python clipcapTraining.py \
    --data ./data/coco/coco_karpathy_split_dinov2_vitb14_train.pkl \
    --out_dir ./checkpoints_dino_custom \
    --prefix dino_custom \
    --epochs 10 \
    --prefix_length 10 \
    --bs 32 \
    --mapping_type mlp \
    --use_dino \
    --dino_model_type dinov2_vitb14 \
    --dino_feature_dim 768 \
    --only_prefix
```

## Key Parameters Explanation:

- `--use_dino`: Enable DINO mode (required for DINO training)
- `--dino_model_type`: Specify which DINO model was used for feature extraction
- `--dino_feature_dim`: Override automatic feature dimension detection
- `--prefix_length`: Number of prefix tokens (set to 10 as requested)
- `--mapping_type`: Choose between 'mlp' or 'transformer' mapping
- `--only_prefix`: Train only the mapping layer, freeze GPT-2
- `--bs`: Batch size (adjust based on GPU memory)
- `--epochs`: Number of training epochs
- `--save_every`: Save checkpoint every N epochs

## Expected Feature Dimensions:

- **DINOv2-S/14**: 384 dimensions
- **DINOv2-B/14**: 768 dimensions  
- **DINOv2-L/14**: 1024 dimensions
- **DINOv2-G/14**: 1536 dimensions

## Training Tips:

1. **Memory Usage**: DINO features are typically larger than CLIP features, so you might need to reduce batch size
2. **Convergence**: DINO-based models may require different learning rates or longer training
3. **Prefix Length**: Experiment with different prefix lengths (5, 10, 20) for optimal performance
4. **Mapping Type**: MLP is faster, Transformer might give better results but requires more memory

## Output:

The training will save checkpoints in the specified output directory:
- `{prefix}-{epoch:03d}.pt`: Model checkpoint for each epoch
- `{prefix}_latest.pt`: Latest model checkpoint (updated every 10k iterations)
- `{prefix}.json`: Training configuration

## Example Full Workflow:

```bash
# 1. Extract DINO features
python clipcap_dino_parse_coco.py --dino_model_type dinov2_vitb14

# 2. Train ClipCap with DINO features (MLP, length 10, prefix-only)
python clipcapTraining.py \
    --data /raid/datasets/models_weights/clipcap/training-features/coco_karpathy_split_dinov2_vitb14_train.pkl \
    --out_dir ./checkpoints_dino_vitb14_mlp_len10 \
    --prefix dino_vitb14_mlp_len10 \
    --epochs 10 \
    --prefix_length 10 \
    --bs 32 \
    --mapping_type mlp \
    --use_dino \
    --dino_model_type dinov2_vitb14 \
    --only_prefix
```

This will train a ClipCap model using DINO features with MLP mapping and sequence length 10 as requested.