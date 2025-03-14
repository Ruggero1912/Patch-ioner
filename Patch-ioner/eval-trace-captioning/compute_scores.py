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


if os.path.abspath(os.path.pardir) not in sys.path:
    sys.path.append(os.path.abspath(os.path.pardir))

from pacsMetric.pac_score import RefPACScore

import clip

# Function to load the DataFrame from file (if it exists), or create a new one
def load_or_create_df(csv_file_path, result_df_columns) -> pd.DataFrame:
    if os.path.exists(csv_file_path):
        # Load the DataFrame from the existing file
        df = pd.read_csv(csv_file_path).fillna("")
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=[
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
            'caption_from', 'use_attn_map_for_bboxes', 'use_attention_weighting'] + 
            list(result_df_columns)
            )
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
    return pacS_model




def _compute_scores_for_input_data(data, device):
    """
    - data : dict in form coco format containing predictions
    """

    gen = {}
    gts = {}
    gen_l = []
    gts_l = []

    pacS_model = get_pac_s_model(device)

    for i, (pred, gt) in enumerate(zip(data['predictions'], data['gt_captions'])):
        gen[i] = [pred]
        gts[i] = [capt for capt in gt.split('\n')]
        gen_l.append(pred)
        gts_l.append([capt for capt in gt.split('\n')])
    print( "len(gen): ", len(gen), "len(gts): ", len(gts) )

    gts_t = PTBTokenizer.tokenize(gts)
    gen_t = PTBTokenizer.tokenize(gen)

    # Initialize score dictionary
    scores_dict = {}

    # Compute BLEU scores
    val_bleu, _ = Bleu(n=4).compute_score(gts_t, gen_t)
    method = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    for metric, score in zip(method, val_bleu):
        scores_dict[metric] = score
        print(f"{metric}: {score:.3f}")

    # Compute METEOR score
    val_meteor, _ = Meteor().compute_score(gts_t, gen_t)
    scores_dict['METEOR'] = val_meteor
    print(f"METEOR: {val_meteor:.3f}")

    # Compute ROUGE_L score
    val_rouge, _ = Rouge().compute_score(gts_t, gen_t)
    scores_dict['ROUGE_L'] = val_rouge
    print(f"ROUGE_L: {val_rouge:.3f}")

    # Compute CIDEr score
    val_cider, _ = Cider().compute_score(gts_t, gen_t)
    scores_dict['CIDEr'] = val_cider
    print(f"CIDEr: {val_cider:.3f}")

    # Compute SPICE score
    val_spice, _ = Spice().compute_score(gts_t, gen_t)
    scores_dict['SPICE'] = val_spice
    print(f"SPICE: {val_spice:.3f}")

    len_candidates = [len(c.split()) for c in gen_l] 
    val_ref_pac_score, ref_pac_score_per_instance = RefPACScore(pacS_model, references=gts_l, candidates=gen_l, device=device, len_candidates=len_candidates)
    scores_dict['RefPAC-S'] = val_ref_pac_score
    print(f"RefPAC-S: {val_ref_pac_score}")

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
    #result_data

    if is_combination_existing(df, config_data):
        print(f"[!] The provided combination was already evaluated [!]")
        row = get_combination_row(df, config_data)
        #if row.isnull().sum().sum() == 0:
        #    print("No missing values")
        #else:
        #    print(f"Added to df since there were missing values")
        #    df = pd.concat([df, pd.concat([pd.DataFrame.from_dict([result_data]), result_df], axis=1)], ignore_index=True)
        df = pd.concat([df, pd.concat([pd.DataFrame.from_dict([config_data]), result_df], axis=1)], ignore_index=True)
        
    else:
        df = pd.concat([df, pd.concat([pd.DataFrame.from_dict([config_data]), result_df], axis=1)], ignore_index=True)
        
    df.to_csv(csv_file_path, index=False)
    return df



def compute_and_stores_scores_main(input_file, device, csv_output = "evaluation_results.csv"):
    assert os.path.exists(input_file)

    data = json.load(open(input_file))

    result_df = _compute_scores_for_input_data(data, device)
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

    compute_and_stores_scores_main(parser.evaluated_predictions_path, parser.device, parser.output_file)