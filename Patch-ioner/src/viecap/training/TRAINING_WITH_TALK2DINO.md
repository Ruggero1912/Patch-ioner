# Training ViECap Decoder with Talk2DINO Textual Representations

This README provides comprehensive steps to train the ViECap decoder using Talk2DINO projected textual features instead of standard CLIP text embeddings.

## Overview

Talk2DINO is a projection layer that transforms CLIP textual representations into the DINO embedding space, enabling better alignment between textual and visual features. This integration allows the ViECap model to leverage more semantically rich textual representations for improved captioning performance.

## Prerequisites

### Dependencies
Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

Key dependencies include:
- PyTorch 
- CLIP (OpenAI)
- Transformers (HuggingFace)
- tqdm, pandas, numpy
- pycocotools

### Talk2DINO Components
The integration requires:
- `talk2dino/talk2dino.py` - ProjectionLayer implementation
- `talk2dino/configs/vitb_mlp_infonce.yaml` - Model configuration
- `talk2dino/weights/vitb_mlp_infonce.pth` - Pre-trained weights

## Step-by-Step Training Process

### Step 1: Prepare Entity Embeddings with Talk2DINO

Generate entity embeddings using Talk2DINO projections:

```bash
python generating_prompt_ensemble.py
```

This script:
- Loads CLIP model (ViT-B/16 by default)
- Loads Talk2DINO projection layer
- Generates ensemble prompts for entities
- Projects CLIP text embeddings to DINO space
- Saves embeddings with `_t2d_` suffix

**Key parameters in the script:**
```python
talk2dino_weights_path = 'talk2dino/weights/vitb_mlp_infonce.pth'
talk2dino_config = 'talk2dino/configs/vitb_mlp_infonce.yaml'
clip_type = 'ViT-B/16'
device = 'cuda:5'  # Adjust as needed
```

**Output:** Entity embeddings saved to:
- `/raid/datasets/viecap_files/annotations/vocabulary/coco_embeddings_ViT-B16_t2d_with_ensemble.pickle`

### Step 2: Extract Talk2DINO Text Features from Captions

Extract textual features from training captions using Talk2DINO:

```bash
python texts_features_extraction.py
```

This script:
- Loads caption datasets with entities
- Encodes captions using CLIP text encoder
- Projects embeddings through Talk2DINO layer
- Saves projected features for training

**Configuration in the script:**
```python
idx = 0  # 0 for COCO, 1 for Flickr30k
talk2dino_weights_path = 'talk2dino/weights/vitb_mlp_infonce.pth'
talk2dino_config = 'talk2dino/configs/vitb_mlp_infonce.yaml'
clip_type = 'ViT-B/16'
```

**Outputs:**
- COCO: `/raid/datasets/viecap_files/annotations/coco/coco_texts_features_ViT-B16_t2d_.pickle`
- Flickr30k: `/raid/datasets/viecap_files/annotations/flickr30k/flickr30k_texts_features_ViT-B16_t2d_.pickle`

### Step 3: Train ViECap Model with Talk2DINO Features

Create a modified training script based on `train_cocoB16.sh`:

```bash
#!/bin/bash
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

DEVICE=$1
EXP_NAME=train_coco_t2d_B16
LOG_FOLDER=logs/${EXP_NAME}
TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FILE="$LOG_FOLDER/${TIME_START}.log"
mkdir -p $LOG_FOLDER

echo "=========================================================="
echo "RUNNING EXPERIMENTS: $EXP_NAME with Talk2DINO features"
echo "=========================================================="

python main.py \
--bs 80 \
--lr 0.00002 \
--epochs 15 \
--device cuda:$DEVICE \
--random_mask \
--prob_of_random_mask 0.4 \
--clip_model ViT-B/16 \
--using_clip_features \
--language_model gpt2 \
--using_hard_prompt \
--soft_prompt_first \
--path_of_datasets /raid/datasets/viecap_files/annotations/coco/coco_texts_features_ViT-B16_t2d_.pickle \
--out_dir /raid/datasets/viecap_files/checkpoints/$EXP_NAME \
2>&1 | tee $LOG_FILE
```

Save this as `scripts/train_coco_t2d_B16.sh` and run:

```bash
bash scripts/train_coco_t2d_B16.sh 0  # Use GPU 0
```

### Step 4: Modify Inference for Talk2DINO

For inference, ensure your validation/inference scripts use the Talk2DINO projected embeddings:

1. **Entity Retrieval**: Use entity embeddings with `_t2d_` suffix
2. **Image Features**: Continue using standard CLIP visual features
3. **Text Generation**: The model will generate captions in the aligned space

## Talk2DINO Integration Details

### How Talk2DINO Works in the Pipeline

1. **Text Encoding**: CLIP processes text → text embeddings (512-dim for ViT-B/16)
2. **Projection**: Talk2DINO projects CLIP text → DINO space (768-dim)
3. **Training**: ViECap decoder learns from projected representations
4. **Alignment**: Better semantic alignment between visual and textual features

### Key Configuration Parameters

**Talk2DINO Model Config** (`vitb_mlp_infonce.yaml`):
```yaml
model:
  act: tanh                # Activation function
  hidden_layer: True       # Use hidden layer in projection
  dino_embed_dim: 768     # DINO embedding dimension
```

**Training Parameters**:
- CLIP model: ViT-B/16 (matches Talk2DINO weights)
- Learning rate: 2e-5
- Batch size: 80
- Epochs: 15
- Embedding dimension: 768 (DINO) vs 512 (CLIP)

## File Structure After Integration

```
ViECap/
├── talk2dino/
│   ├── talk2dino.py                    # ProjectionLayer implementation
│   ├── configs/vitb_mlp_infonce.yaml   # Model configuration
│   └── weights/vitb_mlp_infonce.pth    # Pre-trained weights
├── generating_prompt_ensemble.py       # Entity embeddings with Talk2DINO
├── texts_features_extraction.py       # Caption features with Talk2DINO
├── main.py                            # Training script
└── scripts/
    ├── train_coco_t2d_B16.sh          # Training with Talk2DINO features
    └── ...
```

## Expected Benefits

1. **Better Semantic Alignment**: DINO space provides richer semantic representations
2. **Improved Captioning**: More meaningful text-image associations
3. **Entity Recognition**: Enhanced entity-aware caption generation
4. **Cross-modal Understanding**: Better alignment between visual and textual features

## Troubleshooting

### Common Issues

1. **Dimension Mismatch**: Ensure CLIP model (ViT-B/16) matches Talk2DINO configuration
2. **Missing Weights**: Verify Talk2DINO weights are properly downloaded
3. **CUDA Memory**: Reduce batch size if encountering OOM errors
4. **File Paths**: Adjust paths in scripts based on your data location

### Verification Steps

1. **Check embeddings dimensions**:
   ```python
   import pickle
   with open('path/to/embeddings_t2d_.pickle', 'rb') as f:
       data = pickle.load(f)
   print(data[0][2].shape)  # Should be [768] for Talk2DINO
   ```

2. **Verify Talk2DINO loading**:
   ```python
   from talk2dino.talk2dino import ProjectionLayer
   model = ProjectionLayer.from_config('talk2dino/configs/vitb_mlp_infonce.yaml')
   print(f"Model loaded successfully: {len(model)} parameters")
   ```

## Performance Monitoring

Monitor training progress:
- Check loss convergence in log files
- Compare BLEU/CIDEr scores with baseline
- Validate on held-out datasets
- Monitor entity detection accuracy

## Next Steps

After successful training:
1. Evaluate on standard captioning benchmarks (COCO, Flickr30k)
2. Compare performance with baseline ViECap
3. Analyze entity-aware caption quality
4. Consider fine-tuning Talk2DINO weights jointly with ViECap decoder
