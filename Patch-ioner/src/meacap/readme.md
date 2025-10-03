pip install sentence-transformers


(3.2.1)


package for the usage of meacap invlm version (based on viecap decoder)

pip install nltk

huggingface-cli snapshot-download \
  --repo-id JoeyZoZ/MeaCap \
  --allow-pattern "memory/*" \
  --local-dir /raid/datasets/meacap_files/data/memory

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='JoeyZoZ/MeaCap',
    local_dir='/raid/datasets/meacap_files',
    local_dir_use_symlinks=False
)
"

# Preparing Embeddings for T2D Space

python prepare_embeddings.py --memory_id coco_B16_t2d --use_t2d True


# inference example on one image

python viecap_inference.py --memory_id coco --image_path "/raid/datasets/coco/train2017/000000000064.jpg"


memory concepts: ['clock', 'tree', 'sidewalk', 'city']
the generated caption:  clock on sidewalk in city with trees and sidewalk in background.
![http://images.cocodataset.org/train2017/000000000064.jpg](http://images.cocodataset.org/train2017/000000000064.jpg)



python viecap_inference.py --memory_id coco --image_path "/raid/datasets/coco/train2017/000000000071.jpg"


memory concepts: ['trains']
the generated caption:  Blue and yellow trains passing each other on the tracks.
![http://images.cocodataset.org/train2017/000000000071.jpg](http://images.cocodataset.org/train2017/000000000071.jpg)


