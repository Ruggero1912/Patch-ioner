<p align="center">
  <a href="https://arxiv.org/abs/2510.02898">
    <img src="https://img.shields.io/badge/arXiv-2510.02898-b31b1b.svg?style=for-the-badge&logo=arxiv" alt="arXiv Paper"/>
  </a>
  <a href="https://paciosoft.com/Patch-ioner">
    <img src="https://img.shields.io/badge/🌐%20Project%20Website-success.svg?style=for-the-badge&logo=google-chrome" alt="Project Website"/>
  </a>
  <a href="https://huggingface.co/spaces/Ruggero1912/Patch-ioner">
    <img src="https://img.shields.io/badge/🚀%20Demo-orange.svg?style=for-the-badge&logo=gradio" alt="Demo on Hugging Face Space"/>
  </a>
  <a href="https://huggingface.co/collections/Ruggero1912/patch-ioner-68e7ae42fed581777266b76a">
    <img src="https://img.shields.io/badge/Models-blue.svg?style=for-the-badge&logo=huggingface" alt="Hugging Face Collection"/>
  </a>
</p>

---
# Patch-ioner
## **"One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework"** 💍

Official repository containing the code for the paper **"One Patch to Caption Them All: A Unified Zero-Shot Captioning Franework"**.

---

## 🧩 Installation

You can install **Patch-ioner** directly from GitHub using `pip`:

```bash
pip install git+https://github.com/Ruggero1912/Patch-ioner
```

## 🚀 Loading a Pretrained Model

You can easily load a pretrained model from Hugging Face using the following API:

```python
from patchioner import Patchioner

MODEL_ID = "Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions"

model = Patchioner.from_config(MODEL_ID)
```

Patchioner also supports `AutoModel.from_pretrained` of the `transformers` library.

```python
from transformers import AutoModel

MODEL_ID = "Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions"

model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
```

You can browse all models in the Patch-ioner collection:  
[Patch-ioner Models Collection](https://huggingface.co/collections/Ruggero1912/patch-ioner-68e7ae42fed581777266b76a)

| Model Name | Description / Variant | Hugging Face Link |
|---|---|---|
| `Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions` | Talk2DINO + DeCap variant trained on COCO | [🔗](https://huggingface.co/Ruggero1912/Patch-ioner_talk2dino_decap_COCO_Captions) |
| `Ruggero1912/Patch-ioner_talk2dino_capdec_COCO_Captions` | Talk2DINO + CapDec variant trained on COCO | [🔗](https://huggingface.co/Ruggero1912/Patch-ioner_talk2dino_capdec_COCO_Captions) |
| `Ruggero1912/Patch-ioner_talk2dino_Viecap_COCO_Captions` | Talk2DINO + ViECap variant trained on COCO | [🔗](https://huggingface.co/Ruggero1912/Patch-ioner_talk2dino_viecap_COCO_Captions) |
| `Ruggero1912/Patch-ioner_talk2dino_Meacap_COCO_Captions` | Talk2DINO + MeaCap variant trained on COCO | [🔗](https://huggingface.co/Ruggero1912/Patch-ioner_talk2dino_meacap_COCO_Captions) |


## Trace Captioning Dataset Test Splits

The trace captioning dataset test splits are based on the Localized Narratives datasets for COCO and Flickr30k. These datasets are available inside this repository in the `eval-trace-captioning` folder.

### Available Datasets

- `trace_capt_coco_test.json`: This dataset contains trace captioning test splits based on the COCO dataset.
- `trace_capt_flickr30k_test.json`: This dataset contains trace captioning test splits based on the Flickr30k dataset.

### Dataset Description

The trace captioning datasets are derived from the Localized Narratives annotations, which provide detailed descriptions of images along with the corresponding mouse traces. These traces indicate the sequence in which different parts of the image are described, providing a rich source of information for training and evaluating captioning models.


## Quantitative Experiments

The repository includes code to run quantitative experiments on various captioning tasks. You can find the relevant code in the following folders:
- `eval-trace-captioning`: For evaluating trace captioning tasks.
- `eval-dense-captioning`: For evaluating dense captioning tasks.
- `eval-region-set-captioning`: For evaluating region set captioning tasks.
- `eval-image-captioning`: For evaluating image captioning tasks.


### Evaluating the Model on Trace Captioning

To evaluate the model on the trace captioning task, use the eval_trace_captioning.py script. This script runs the model on a specified dataset and computes relevant evaluation metrics.

#### Running the Evaluation

To perform the evaluation, use the following command:

```
python eval_trace_captioning.py --model_name <MODEL_NAME> \
                                --evaluation_dataset <DATASET_PATH> \
                                --batch_size 16 \
                                --device cuda
```


Replace <MODEL_NAME> with the name of the model and <DATASET_PATH> with the path to the dataset.

Available Options

- --model_name (str, required): The name of the model to evaluate.
- --evaluation_dataset (str, required): Path to the dataset used for evaluation.
- --batch_size (int, default=16): Number of samples per batch during evaluation.
- --device (str, default='cuda' if available): The device to run the evaluation on (cuda or cpu).
- --use_gaussian_weighting (flag): If set, applies Gaussian weighting to the captions.
- --gaussian_variance (float, default=1.0): Sets the variance for Gaussian weighting.
- --keep_img_ratio (flag): Maintains the image aspect ratio when resizing.
- --caption_bboxes_type (str, default=None): Specifies the type of bounding boxes for captions.
- --use_attention_weighting (flag): If set, weights patches using the attention map.
- --keep_n_best_sims (int, default=None): Stores the top-N similarities for visualization purposes.
- --caption_from (str, default='patches'): Specifies whether to generate captions from patches or cls tokens.
- --configs_dir (str, default='../configs'): Path to the configuration files directory.
- --use_attn_map_for_bboxes (flag): Uses the attention map to define bounding boxes.
- --csv_scores_output (str, default='evaluation_results.csv'): Path to save the evaluation results.


#### Example Usage

Running Evaluation with Gaussian Weighting
```
python eval_trace_captioning.py --model_name mlp.k \
                                --evaluation_dataset data/trace_captioning.json \
                                --batch_size 32 \
                                --use_gaussian_weighting \
                                --gaussian_variance 0.8 \
                                --device cuda
```
This command evaluates the mlp.k model on data/trace_captioning.json, applying Gaussian weighting with a variance of 0.8 and running on a GPU (cuda).


### Evaluating the Model on Dense Captioning

To evaluate the model on the dense captioning task, use the `eval_densecap.py` script. This script runs the model on a specified dataset and computes relevant evaluation metrics.

#### Running the Evaluation

To perform the evaluation, use the following command:

```
python eval_densecap.py --model_name <MODEL_NAME> \
                        --evaluation_dataset <DATASET_PATH> \
                        --batch_size 16 \
                        --device cuda
```

Replace `<MODEL_NAME>` with the name of the model and `<DATASET_PATH>` with the path to the dataset.

Available Options

- --model_name (str, required): The name of the model to evaluate.
- --evaluation_dataset (str, required): Path to the dataset used for evaluation.
- --batch_size (int, default=16): Number of samples per batch during evaluation.
- --device (str, default='cuda' if available): The device to run the evaluation on (cuda or cpu).
- --use_gaussian_weighting (flag): If set, applies Gaussian weighting to the captions.
- --gaussian_variance (float, default=1.0): Sets the variance for Gaussian weighting.
- --keep_img_ratio (flag): Maintains the image aspect ratio when resizing.
- --caption_bboxes_type (str, default=None): Specifies the type of bounding boxes for captions.
- --configs_dir (str, default='../configs'): Path to the configuration files directory.
- --compute_scores (bool, default=True): Computes the dense captioning MAP score.
- --compute_scores_verbose (bool, default=False): Verbose output for score computation.
- --overwrite (bool, default=True): Overwrites existing results.
- --overwrite_inference (str, default=None): Overwrites inference results.
- --compute_predictions_scores (bool, default=True): Computes prediction scores.
- --caption_from (str, default='patches'): Specifies whether to generate captions from patches or cls tokens.
- --use_attn_map_for_bboxes (bool, default=False): Uses the attention map to define bounding boxes.

#### Example Usage

Running Evaluation with Gaussian Weighting
```
python eval_densecap.py --model_name mlp.k \
                        --evaluation_dataset data/vg_test_dense_captioning.json \
                        --batch_size 32 \
                        --use_gaussian_weighting \
                        --gaussian_variance 0.8 \
                        --device cuda
```
This command evaluates the mlp.k model on `data/vg_test_dense_captioning.json`, applying Gaussian weighting with a variance of 0.8 and running on a GPU (cuda).

### Evaluating the Model on Region-Set Captioning

To evaluate the model on the region-set captioning task, use the `eval_region_set_captioning.py` script. This script runs the model on a specified dataset and computes relevant evaluation metrics.

#### Running the Evaluation

To perform the evaluation, use the following command:

```
python eval_region_set_captioning.py --model_name <MODEL_NAME> \
                                     --evaluation_dataset <DATASET_PATH> \
                                     --batch_size 16 \
                                     --device cuda
```

Replace `<MODEL_NAME>` with the name of the model and `<DATASET_PATH>` with the path to the dataset.

Available Options

- --model_name (str, required): The name of the model to evaluate.
- --evaluation_dataset (str, required): Path to the dataset used for evaluation.
- --batch_size (int, default=16): Number of samples per batch during evaluation.
- --device (str, default='cuda' if available): The device to run the evaluation on (cuda or cpu).
- --use_gaussian_weighting (flag): If set, applies Gaussian weighting to the captions.
- --gaussian_variance (float, default=1.0): Sets the variance for Gaussian weighting.
- --keep_img_ratio (flag): Maintains the image aspect ratio when resizing.
- --caption_bboxes_type (str, default=None): Specifies the type of bounding boxes for captions.
- --caption_from (str, default='patches'): Specifies whether to generate captions from patches or cls tokens.
- --configs_dir (str, default='../configs'): Path to the configuration files directory.
- --use_attn_map_for_bboxes (flag): Uses the attention map to define bounding boxes.
- --csv_scores_output (str, default='evaluation_results.csv'): Path to save the evaluation results.

#### Example Usage

Running Evaluation with Gaussian Weighting
```
python eval_region_set_captioning.py --model_name mlp.k \
                                     --evaluation_dataset data/region_set_captioning.json \
                                     --batch_size 32 \
                                     --use_gaussian_weighting \
                                     --gaussian_variance 1.0 \
                                     --device cuda
```
This command evaluates the mlp.k model on `data/region_set_captioning.json`, applying Gaussian weighting with a variance of 1.0 and running on a GPU (cuda).

### Evaluating the Model on Image Captioning

To evaluate the model on the image captioning task, use the `eval_image_captioning.py` script. This script runs the model on a specified dataset and computes relevant evaluation metrics.

#### Running the Evaluation

To perform the evaluation, use the following command:

```
python eval_image_captioning.py --model_name <MODEL_NAME> \
                                --evaluation_dataset <DATASET_PATH> \
                                --batch_size 16 \
                                --device cuda
```

Replace `<MODEL_NAME>` with the name of the model and `<DATASET_PATH>` with the path to the dataset.

Available Options

- --model_name (str, required): The name of the model to evaluate.
- --evaluation_dataset (str, required): Path to the dataset used for evaluation.
- --batch_size (int, default=16): Number of samples per batch during evaluation.
- --use_gaussian_weighting (flag): If set, applies Gaussian weighting to the captions.
- --gaussian_variance (float, default=1.0): Sets the variance for Gaussian weighting.
- --keep_img_ratio (flag): Maintains the image aspect ratio when resizing.
- --keep_n_best_sims (int, default=None): Stores the top-N similarities for visualization purposes.
- --caption_from (str, default='cls'): Specifies whether to generate captions from cls tokens, average self-attention, or patches.
- --configs_dir (str, default='../configs'): Path to the configuration files directory.
- --device (str, default='cuda' if available): The device to run the evaluation on (cuda or cpu).
- --no_scores (flag): If set, does not compute the scores for the captions.

#### Example Usage

Running Evaluation with Gaussian Weighting
```
python eval_image_captioning.py --model_name mlp.k \
                                --evaluation_dataset data/coco-test.json \
                                --batch_size 32 \
                                --use_gaussian_weighting \
                                --gaussian_variance 1.0 \
                                --device cuda
```
This command evaluates the mlp.k model on `data/coco-test.json`, applying Gaussian weighting with a variance of 1.0 and running on a GPU (cuda).

Available datasets:
- coco-test.json
- flickr30_test.json

## Setup Requirements

To set up the requirements for this repository, follow the steps below:

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Git
- Conda
- CUDA Toolkit (if using GPU)

### Installation

1. **Clone the repository:**
    ```bash
    git clone [REDACTED]
    cd Patch-ioner
    ```

2. **Create a Conda Environment**
```
    conda env create -f environment.yml
```

## Training the Decoder

You can train the decoder using the following commands:

### Talk2DINO, Memory (~DeCap)

```bash
python decoderTraining.py --out_dir weights_dino_b14_karpathy --not-distributed 1 --local-rank 1 --dataset coco_train_karpathy.json --prefix coco_karpathy --talk2dino_weights weights_talk2dino/vitb_mlp_infonce.pth --talk2dino_config configs_talk2dino/vitb_mlp_infonce.yaml --use_dino_feats --pre_extract_features
```

### Talk2DINO, Noise (~CapDec)

```bash
python decoderTraining.py --out_dir weights_dino_b14_noise_karpathy --not-distributed 1 --local-rank 1 --dataset coco_train_karpathy.json --prefix coco_karpathy --talk2dino_weights weights_talk2dino/vitb_mlp_infonce.pth --talk2dino_config configs_talk2dino/vitb_mlp_infonce.yaml --use_dino_feats --pre_extract_features --gaussian_noise 0.08
```



### CLIP B16, Memory (DeCap) Karpathy Train Split
```bash
python decoderTraining.py --out_dir weights_clip_b16_karpathy --not-distributed 1 --local-rank 0 --dataset coco_train_karpathy.json --prefix coco_karpathy
```

### CLIP B32, Memory (DeCap) Karpathy Train Split
```bash
python decoderTraining.py --out_dir weights_clip_b32_karpathy --not-distributed 1 --local-rank 0 --dataset coco_train_karpathy.json --prefix coco_karpathy --clip_model ViT-B/32
```

## Model Configuration

You can define a configuration for the model in the `configs` folder as a YAML file. The allowed options include:

- `decap_weights`: Path to the textual decoder weights file.
- `prefix_size`: Size of the textual embedding prefix.
- `linear_talk2dino`: Boolean flag to use the linear version talk2dino.
- `support_memory_size`: Size of the memory bank.
- `dino_model`: Model type for DINO.
- `normalize`: Boolean flag for normalization of the embeddings in input to the decoder.
- `kkv_attention`: Boolean flag for KKV attention.
- `projection_type`: Path to the projection type file.

Example configuration:
```yaml
decap_weights: '/raid/datasets/models_weights/decap_weights/talkingdino-ksplits/coco_karpathy-009.pt'
prefix_size: 768
linear_talk2dino: False
support_memory_size: 591753
dino_model: 'dinov2_vitb14_reg'
normalize: True
kkv_attention: False
projection_type: '/raid/datasets/im2txtmemories/coco_train_karpathy.json'
```

To setup a ViECap baseline, populate the nested fields at the key `viecap`. The available options are:

- `project_length`: Length of the learnable prefix projected from vision features.
- `top_k`: The number of detected objects to use as hard prompt.
- `name_of_entities_text`: The name of the collection of entities to use for the hard-prompt.
- `files_path`: Path to directory containing ViECap-related checkpoints and auxiliary data.
- `weight_path`: Path to the learned weights used for prefix projection.
- `using_hard_prompt`: True in the default configuration. 
- `soft_prompt_first`: True in the default configuration.
- `using_greedy_search`: True for greedy search, False for beam search.
- `language_model`: GPT-2 in the default configuration.

In the case of MeaCap baselines, set the nested fields at `viecap` -> `meacap`. The available options are:

- `memory_caption_num`: standard value for MeaCap is 5
- `vl_model`: the clip version 
- `wte_model_path`: the default value is "sentence-transformers/all-MiniLM-L6-v2"
- `parser_checkpoint`: Checkpoint for a scene graph parser, default is "lizhuang144/flan-t5-base-VG-factual-sg".
- `memory_id`: the id of the memory pool.
- `memory_base_path`: Path to directory containing MeaCap-related checkpoints and auxiliary data.



## Credits

This repository contains code from several other repositories, including:
- [DeCap](https://github.com/dhg-wei/DeCap)
- [MeaCap](https://github.com/joeyz0z/MeaCap)
- [ViECap](https://github.com/FeiElysia/ViECap)
- [Talk2DINO](https://github.com/lorebianchi98/Talk2DINO)
- [PAC-S](https://github.com/aimagelab/pacscore)
- [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP)
- and others.

We acknowledge and thank the authors of these repositories for their contributions.

## Reference
If you found this code useful, please cite the following paper:
```
@misc{bianchi2025patchcaptionallunified,
      title={One Patch to Caption Them All: A Unified Zero-Shot Captioning Framework}, 
      author={Lorenzo Bianchi and Giacomo Pacini and Fabio Carrara and Nicola Messina and Giuseppe Amato and Fabrizio Falchi},
      year={2025},
      eprint={2510.02898},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2510.02898}, 
}
```
