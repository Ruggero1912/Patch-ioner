from speaksee.evaluation import Bleu, Meteor, Rouge, Cider, Spice
from speaksee.evaluation import PTBTokenizer
import torch
import random
import numpy as np
import itertools
import argparse
import os, sys
import json
import munkres
from tqdm import tqdm
import pandas as pd
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
import re
from copy import deepcopy
if os.path.abspath(os.path.pardir) not in sys.path:
    sys.path.append(os.path.abspath(os.path.pardir))

from pacsMetric.pac_score import RefPACScore, PACScore

import clip

def draw_trace_points(img, traces, point_color=(255, 0, 0), point_size=3, alpha=122):
    img = img.convert("RGBA")  # Ensure original image has an alpha channel
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))  # Transparent layer to draw on
    draw = ImageDraw.Draw(overlay)
    width, height = img.size

    # Append alpha value to color tuple
    rgba_color = point_color + (alpha,)

    for point in traces:
        x, y = point['x'], point['y']
        if 0 <= x <= 1 and 0 <= y <= 1:
            abs_x, abs_y = int(x * width), int(y * height)
            draw.ellipse(
                (abs_x - point_size, abs_y - point_size, abs_x + point_size, abs_y + point_size),
                fill=rgba_color,
                outline=rgba_color
            )

    # Composite overlay on original image
    result = Image.alpha_composite(img, overlay)

    return result.convert("RGB")  # Convert back if needed

def preprocess(img, ann):
    return draw_trace_points(img, ann)

def get_coco_id(image_path: str) -> int:
    """Extracts the COCO ID from the given image path."""
    match = re.search(r'COCO_[a-z]+2014_(\d+)\.jpg', image_path)
    if match:
        return str(int(match.group(1)))
    raise ValueError("Invalid COCO image path format")

def get_img_path(img_path, idx, ann, tmp_path):
    out_path = os.path.join(tmp_path, f"{idx}.jpg")
    if os.path.exists(out_path):
        return out_path
    img = preprocess(Image.open(img_path), ann)
    img.save(out_path)
    return out_path

def extract_id_from_path(path: str) -> str:
    filename = os.path.basename(path)        
    id_str, _ = os.path.splitext(filename)     
    return str(int(id_str))

# Function to load the DataFrame from file (if it exists), or create a new one
def load_or_create_df(csv_file_path, result_df_columns) -> pd.DataFrame:
    if os.path.exists(csv_file_path):
        # Load the DataFrame from the existing file
        df = pd.read_csv(csv_file_path).fillna("")
    else:
        # Create a new DataFrame if the file doesn't exist
        base_columns = [
            'model_name',
            'evaluation_dataset',
            'use_gaussian_weighting',
            'gaussian_variance',
            'keep_img_ratio',
            'caption_bboxes_type',
            'double_dino_last_layer',
            'double_dino_feature_computation',
            'representation_cleaning_type',
            'representation_cleaning_alpha',
            'representation_cleaning_clean_from',
            'representation_cleaning_clean_after_projection',
            'caption_from', 'use_attn_map_for_bboxes', 'use_attention_weighting',
            # Timing columns
            'total_preprocessing_time', 'total_inference_time', 'total_time',
            'avg_preprocessing_time_per_batch', 'avg_inference_time_per_batch',
            'avg_preprocessing_time_per_image', 'std_preprocessing_time_per_image',
            'avg_inference_time_per_image', 'std_inference_time_per_image',
            'images_per_second_inference', 'images_per_second_total',
            # FLOP columns
            'flops_per_image_estimate', 'flops_per_forward_pass', 'flop_counter_used'
        ]
        
        all_columns = base_columns + list(result_df_columns)
        df = pd.DataFrame(columns=all_columns)
    return df

def get_combination_row(df, combination):
    filter_condition = True
    for key, value in combination.items():
        if key not in df.columns: return None
        if value is None: value = ""
        filter_condition &= (df[key] == value)  # Logical AND across all conditions
    return df[filter_condition]

# Check if the combination already exists in the DataFrame
def is_combination_existing(df, combination):
    comb_row = get_combination_row(df, combination)
    if comb_row is None: return False
    # Check if any rows match the filter
    return comb_row.shape[0] > 0

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
pacS_model = None

def get_pac_s_model(device, clip_model_name = "ViT-B/32"):
    global pacS_model

    if pacS_model is not None:
        return pacS_model

    _MODELS = {
        "ViT-B/32": "/raid/datasets/models_weights/pacs-metric/clip_ViT-B-32.pth",
        "open_clip_ViT-L/14": "/raid/datasets/models_weights/pacs-metric/openClip_ViT-L-14.pth"
    }

    pacS_model, clip_preprocess = clip.load(clip_model_name, device=device)

    pacS_model = pacS_model.to(device)
    pacS_model = pacS_model.float()

    checkpoint = torch.load(_MODELS[clip_model_name])
    pacS_model.load_state_dict(checkpoint['state_dict'])
    pacS_model.eval()
    return pacS_model, clip_preprocess

def get_clipscore_model(device, clip_model_name = "ViT-B/32"):
    model, clip_preprocess = clip.load(clip_model_name, device=device)

    model.to(device)
    model.float()
    model.eval()

    return model, clip_preprocess


import h5py

def get_CLIPScore(model, clip_preprocess, references_images, candidates, device='cuda', w=2.5, batch_size=64, cache_file='cache.hdf5'):
    from tqdm import tqdm
    """
    Compute CLIPScore between reference images and candidate captions.

    Args:
        model: CLIP model (e.g., from OpenAI or HuggingFace).
        clip_preprocess: preprocessing transform for the images (from CLIP).
        references_images: list of PIL images or image paths.
        candidates: list of candidate captions (without prompt).
        device: device for computation.
        w: weight factor for CLIPScore (default 2.5).
        batch_size: batch size for processing.
        cache_file: path to HDF5 file to cache or load image features.

    Returns:
        Tuple containing:
            - mean_clip_score: float
            - clip_scores: list of float, individual CLIPScore values per (image, caption) pair
    """
    assert len(references_images) == len(candidates), "Mismatch between number of images and captions."

    model.eval()
    model.to(device)

    prompts = ["A photo depicts " + caption for caption in candidates]

    # Try loading cached image features
    image_features = None

    cache_is_valid = False

    if os.path.exists(cache_file):
        with h5py.File(cache_file, 'r') as f:
            if 'image_features' in f and len(f['image_features']) == len(references_images):
                #print(f"Loading image features from cache {cache_file}...")
                image_features = torch.from_numpy(f['image_features'][:]).to(device)
                cache_is_valid = True
            else:
                print("Cache found but invalid or mismatched; recalculating features.")
    
    # If cache doesn't exist or is invalid, compute and save features
    if image_features is None and not cache_is_valid:
        # Load or preprocess images
        if isinstance(references_images[0], str):
            references_images = [Image.open(path).convert("RGB") for path in references_images]
        preprocessed_images = [clip_preprocess(image) for image in tqdm(references_images, desc="Preprocessing images")]
        #image_tensor = torch.stack(preprocessed_images)#.to(device)

        image_features_list = []

        with torch.no_grad():
            for i in tqdm(range(0, len(preprocessed_images), batch_size), desc="Encoding images"):
                #batch = image_tensor[i:i+batch_size].to(device)
                batch = torch.stack(preprocessed_images[i:i+batch_size]).to(device)
                features = model.encode_image(batch)
                features = features / features.norm(dim=-1, keepdim=True)
                image_features_list.append(features.cpu())

        image_features = torch.cat(image_features_list, dim=0)

        # Save to cache
        with h5py.File(cache_file, 'w') as f:
            print(f"Saving images at {cache_file}")
            f.create_dataset('image_features', data=image_features.cpu().numpy())

    # Tokenize captions
    text_tokens = clip.tokenize(prompts, truncate=True).to(device)

    clip_scores = []

    with torch.no_grad():
        for i in tqdm(range(0, len(candidates), batch_size), desc="Computing CLIPScore"):
            text_batch = text_tokens[i:i+batch_size]
            text_features = model.encode_text(text_batch)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            #if cache_is_valid:
            #     with h5py.File(cache_file, 'r') as f:
            #        image_batch = torch.from_numpy(f['image_features'][i:i+batch_size]).to(device)
            #else:
            image_batch = image_features[i:i+batch_size].to(device)

            cos_sim = (text_features * image_batch).sum(dim=1)
            scores = w * torch.clamp(cos_sim, min=0.0)

            clip_scores.extend(scores.cpu().tolist())

    mean_clip_score = sum(clip_scores) / len(clip_scores)

    return mean_clip_score, clip_scores

def _compute_scores_for_input_data(data, device, test_set=None, tmp_dir='tmp'):
    """
    - data : dict in form coco format containing predictions
    """
    os.makedirs(tmp_dir, exist_ok=True)
    gen = {}
    gts = {}
    gen_l = []
    gts_l = []
    references_images = []
    pacS_model, pacs_preprocess = get_pac_s_model(device)
    clips_model, clips_preprocess = get_clipscore_model(device)

    for i, (pred, gt, path) in tqdm(enumerate(zip(data['predictions'], data['gt_captions'], data['img_paths'])), total=len(data['predictions'])):
        gen[i] = [pred]
        gts[i] = [capt for capt in gt.split('\n')]
        gen_l.append(pred)
        gts_l.append([capt for capt in gt.split('\n')])
        try:
            idx = extract_id_from_path(path)
        except:
            idx = get_coco_id(path)
        img_entry = test_set[idx]
        img_entry = test_set[extract_id_from_path(path)]
        ann = img_entry['traces'][img_entry['captions'].index(gt)]
        references_images.append(get_img_path(path, len(references_images), ann, tmp_dir))

    print( "len(gen): ", len(gen), "len(gts): ", len(gts) )

    gts_t = PTBTokenizer.tokenize(gts)
    gen_t = PTBTokenizer.tokenize(gen)

    # Initialize score dictionary
    scores_dict = {}

    print(f"Going to compute CLIPScores")
    val_clip_score, clip_score_per_instance = get_CLIPScore(
        clips_model, clips_preprocess, references_images, candidates=gen_l, device=device, cache_file=f"{tmp_dir}_clips.hdf5")
    val_clip_score, clip_score_per_instance = float(val_clip_score), [float(x) for x in clip_score_per_instance]
    scores_dict['CLIP-S'] = val_clip_score
    print(f"CLIP-S: {val_clip_score}")

    print(f"Going to compute PACScores")
    val_pac_score, pac_score_per_instance, _, len_candidates = PACScore(
        pacS_model, pacs_preprocess, references_images, candidates=gen_l, device=device, w=2.0)
    val_pac_score, pac_score_per_instance = float(val_pac_score), [float(x) for x in pac_score_per_instance]
    scores_dict['PAC-S'] = val_pac_score
    print(f"PAC-S: {val_pac_score}")

    # Compute BLEU scores
    val_bleu, per_instance_bleu = Bleu(n=4).compute_score(gts_t, gen_t)
    method = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    for metric, score in zip(method, val_bleu):
        scores_dict[metric] = score
        print(f"{metric}: {score:.3f}")

    # Compute METEOR score
    val_meteor, per_instance_meteor = Meteor().compute_score(gts_t, gen_t)
    scores_dict['METEOR'] = val_meteor
    print(f"METEOR: {val_meteor:.3f}")

    # Compute ROUGE_L score
    val_rouge, per_instance_rouge = Rouge().compute_score(gts_t, gen_t)
    scores_dict['ROUGE_L'] = val_rouge
    print(f"ROUGE_L: {val_rouge:.3f}")

    # Compute CIDEr score
    val_cider, per_instance_cider = Cider().compute_score(gts_t, gen_t)
    scores_dict['CIDEr'] = val_cider
    print(f"CIDEr: {val_cider:.3f}")

    # Compute SPICE score
    val_spice, per_instance_spice = Spice().compute_score(gts_t, gen_t)
    scores_dict['SPICE'] = val_spice
    print(f"SPICE: {val_spice:.3f}")

    len_candidates = [len(c.split()) for c in gen_l] 
    val_ref_pac_score, ref_pac_score_per_instance = RefPACScore(pacS_model, references=gts_l, candidates=gen_l, device=device, len_candidates=len_candidates)
    scores_dict['RefPAC-S'] = val_ref_pac_score
    print(f"RefPAC-S: {val_ref_pac_score}")

    # Standard deviation for METEOR
    scores_dict['METEOR_std'] = float(np.std(per_instance_meteor))
    # Standard deviation for BLEU scores
    scores_dict['Bleu_1_std'] = float(np.std(per_instance_bleu[0]))
    scores_dict['Bleu_2_std'] = float(np.std(per_instance_bleu[1]))
    scores_dict['Bleu_3_std'] = float(np.std(per_instance_bleu[2]))
    scores_dict['Bleu_4_std'] = float(np.std(per_instance_bleu[3]))
    # Standard deviation for ROUGE_L
    scores_dict['ROUGE_L_std'] = float(np.std(per_instance_rouge))
    # Standard deviation for CIDEr
    scores_dict['CIDEr_std'] = float(np.std(per_instance_cider))
    # Standard deviation for SPICE
    scores_dict['SPICE_std'] = float(np.std([x['All']['f'] for x in per_instance_spice]))
    # Standard deviation for PAC-S
    scores_dict['PAC-S_std'] = float(np.std(pac_score_per_instance))
    # Standard deviation for CLIP-S
    scores_dict['CLIP-S_std'] = float(np.std(clip_score_per_instance))
    # Standard deviation for RefPAC-S
    scores_dict['RefPAC-S_std'] = float(np.std(ref_pac_score_per_instance))

    # Convert to DataFrame, round scores, and scale by 100
    result_df = pd.DataFrame([scores_dict]).round(3) * 100

    # Display DataFrame
    return result_df

def store_results(data, result_df, csv_file_path = "evaluation_results.csv"):
    # Load or create the DataFrame at the start
    df = load_or_create_df(csv_file_path, result_df_columns=result_df.columns)

    config_data = data['config_data']
    if 'caption_from' not in config_data:
        config_data['caption_from'] = 'patches'
    
    # Create result row starting with config data
    result_row = config_data.copy()
    
    # Add evaluation scores
    result_row.update(result_df.iloc[0].to_dict())
    
    # Filter out complex data types that can't be saved to CSV
    def filter_for_csv(data_dict):
        csv_compatible = {}
        for key, value in data_dict.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                csv_compatible[key] = value
            elif isinstance(value, (list, dict)):
                # Skip complex data types for CSV
                continue
            else:
                # Convert other types to string
                csv_compatible[key] = str(value)
        return csv_compatible
    
    result_row = filter_for_csv(result_row)

    if is_combination_existing(df, config_data):
        print(f"[!] The provided combination was already evaluated [!]")
        row = get_combination_row(df, config_data)
    
    # Convert result_row to DataFrame and concatenate
    result_row_df = pd.DataFrame([result_row])
    df = pd.concat([df, result_row_df], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    return df

def compute_and_stores_scores_main(input_file, device, csv_output = "evaluation_results.csv", test_set=None):
    assert os.path.exists(input_file)

    data = json.load(open(input_file))
    
    # Try to load enhanced timing data if available
    enhanced_file = input_file.replace('.json', '_enhanced.pkl')
    if os.path.exists(enhanced_file):
        print(f"Loading enhanced timing data from {enhanced_file}")
        import pickle
        with open(enhanced_file, 'rb') as f:
            enhanced_data = pickle.load(f)
        
        # Extract timing and FLOP data from enhanced data
        computation_details = enhanced_data.get('computation_details', {})
        other_config_data = enhanced_data.get('other_config_data', {})
        
        # Update config_data with timing information
        timing_data = {}
        for key, value in computation_details.items():
            if key not in ['preprocessing_times', 'inference_times', 'flop_measurements']:  # Skip complex data
                timing_data[key] = value
        
        data['config_data'].update(timing_data)
        print(f"Added {len(timing_data)} timing metrics to config data")

    if isinstance(test_set, str):
        tmp_dir = f'{test_set.split(".")[0]}_tmp'
        with open(test_set, 'r') as f:
            test_set = json.load(f)
    else:
        tmp_dir = 'tmp'
    result_df = _compute_scores_for_input_data(data, device, test_set, tmp_dir=tmp_dir)
    print("Score computed! going to store the results...\n\n")
    print(result_df)
    print()
    df = store_results(data, result_df, csv_file_path=csv_output)

    return df

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate captioning annotations.")
    parser.add_argument(
        "--evaluated_predictions_path",
        required=True,
        type=str,
        help="Path to the evaluated annotations file (JSON format)."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for RefPAC-S score computation")
    parser.add_argument("--output_file", default="evaluation_results.csv", type=str, help="Path to the output csv file containing the scores")

    args = parser.parse_args()

    test_set = 'coco_track_recaptioned_complete.json'
    compute_and_stores_scores_main(args.evaluated_predictions_path, args.device, args.output_file, test_set)