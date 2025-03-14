# Standard Library Imports
import argparse
import json
import math
import os
import sys
from copy import deepcopy
import subprocess
sys.path.append("../")

# Third-Party Library Imports
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torch
import torchvision.transforms as T
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

# Project-Specific Imports
from src.bbox_utils import adjust_bbox_for_transform, adjust_bbox_for_transform_no_scale
from src.model import Patchioner

import json
import matplotlib.pyplot as plt
import os
import re
import torch
import random
import math

from tqdm import tqdm
from copy import deepcopy
from PIL import Image
from src.bbox_utils import draw_bounding_boxes, adjust_bbox_for_transform, adjust_bbox_for_transform_no_scale, extract_bboxes_feats
from src.model import Patchioner

device = 'cuda'

def extract_coco_id(filename):
        numbers = re.findall(r'\d+', filename)

        if len(numbers) > 1:
            second_number = int(numbers[1])  # Get the second number
            return second_number
        assert "Invalid filename"

def evaluate(model_name, 
             evaluation_dataset,
             batch_size,
             use_gaussian_weighting=False,
             gaussian_variance=0.5,
             keep_img_ratio=True,
             caption_bboxes_type=None,
             double_dino_last_layer=False,
             double_dino_feature_computation='avg',
             representation_cleaning_type=None,
             representation_cleaning_alpha=0.4,
             representation_cleaning_clean_from="cls",
             representation_cleaning_clean_after_projection=True,
             caption_from='patches',
             configs_dir="../configs",
             keep_n_best_sims=None,
             device = "cuda" if torch.cuda.is_available() else 'cpu',
             use_attn_map_for_bboxes=False
             ):
    
    crop_str = "CROP" if keep_img_ratio else "NO-CROP"
    if caption_bboxes_type is None:
        if use_attn_map_for_bboxes:
            gaussian_str = f"-ATTN_WEIGHTED_PATCHES"
        elif not double_dino_last_layer:
            gaussian_str = f"-GAUSSIAN-var_{gaussian_variance}" if use_gaussian_weighting else ""
        else:
            gaussian_str = f"DOUBLEDINO-{double_dino_feature_computation}"
        if representation_cleaning_type is not None:
            gaussian_str += f"-{representation_cleaning_type}-{representation_cleaning_alpha}-{representation_cleaning_clean_from}-{'after-proj' if representation_cleaning_clean_after_projection else 'before-proj'}"
    else:
        gaussian_str = f"captioning-of-bboxes_{caption_bboxes_type}"
    if caption_from != 'patches':
        caption_from_str = '-captionfromCLS'
    else:
        caption_from_str = ""
    evaluation_dataset_name = 'coco' if 'coco' in evaluation_dataset else 'flickr30k'
    output_file = f"./annotations/predictions_{model_name}-{evaluation_dataset_name}-{crop_str}{gaussian_str}{caption_from_str}.json"
    output_file = os.path.abspath(output_file)
    model_config = os.path.join(configs_dir,f"{model_name}.yaml")
    
    print("\n=== Parameter Recap ===")
    print(f"Model Config: {model_config}")
    print(f"Evaluation Dataset: {evaluation_dataset}")
    print(f"Batch Size: {batch_size}")
    print(f"Use Gaussian Weighting: {use_gaussian_weighting}")
    print(f"Gaussian Variance: {gaussian_variance}")
    print(f"Keep Image Aspect Ratio: {keep_img_ratio}")
    print(f"Caption BBoxes Type: {caption_bboxes_type}")
    print(f"Double DINO Last Layer: {double_dino_last_layer}")
    if double_dino_last_layer:
        print(f"Double DINO Feature Computation: {double_dino_feature_computation}")
    print(f"Representation Cleaning Type: {representation_cleaning_type}")
    if representation_cleaning_type:
        print(f"Representation Cleaning Alpha: {representation_cleaning_alpha}")
        print(f"Representation Cleaning Clean From: {representation_cleaning_clean_from}")
        print(f"Representation Cleaning Clean After Projection: {representation_cleaning_clean_after_projection}")
    print(F"use self attn weighting: {use_attn_map_for_bboxes}")
    print(f"Caption from: {caption_from}")
    print(f"Out File: {output_file}")
    print("=======================\n")
    
    model = Patchioner.from_config(model_config, device=device)
    model.eval()
    model.to(device)
    
    with open(evaluation_dataset, 'r') as fp:  
        test_set = json.load(fp)
    
    if 'coco' in evaluation_dataset_name:
        base_path = "/raid/datasets/coco"
        train_path = os.path.join(base_path, "train2014")
        val_path = os.path.join(base_path, "val2014")
        test_path = os.path.join(base_path, "test2014")
        coco_filenames = {extract_coco_id(x): ('train2014/' if 'train2014' in x else 'test2014/' if 'test2014' in x else 'val2014/') + x for x in os.listdir(train_path) + os.listdir(val_path) + os.listdir(test_path)}
    else:
        base_path = '/raid/datasets/flickr30k-images'


    samples = {
        'gt_captions': [],
        'img_paths': [],
        'bboxes': []
    }

    for i, (img_id, img_obj) in enumerate(test_set.items()):
        if 'coco' in evaluation_dataset_name:
            img_path = os.path.join(base_path, coco_filenames[int(img_id)])
        else:
            img_path = os.path.join(base_path, f"{img_id}.jpg")
        for caption, bboxes, in img_obj.items():
            # convert x1y1x2y2 to xywh
            new_bboxes = [[bbox[0], bbox[1], bbox[2]- bbox[0], bbox[3] - bbox[1]] for bbox in bboxes] 
            
            samples['gt_captions'].append(caption)
            samples['img_paths'].append(img_path)
            samples['bboxes'].append(new_bboxes)
    
    preds = []
    n_imgs = len(samples['bboxes'])
    n_batch = math.ceil(n_imgs / batch_size)

    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs
        batch_size_ = end - start
        gt_captions = samples['gt_captions'][start:end]
        img_paths = samples['img_paths'][start:end]
        bboxes = samples['bboxes'][start:end]
        
        raw_imgs = [Image.open(img_path).convert('RGB') for img_path in img_paths]
        
        if keep_img_ratio:
            batch_imgs = torch.stack([model.image_transforms(img) for img in raw_imgs]).to(device)
        else:
            batch_imgs = torch.stack([model.image_transforms_no_crop(img) for img in raw_imgs]).to(device)
        
        adjusted_bboxes = []
        n_max_boxes = max(map(len, bboxes))
        for bbox, img in zip(bboxes, raw_imgs):
            if keep_img_ratio:
                adjusted_bbox = [adjust_bbox_for_transform(img, box, resize_dim=model.resize_dim, crop_dim=model.crop_dim) for box in bbox]    
            else:
                adjusted_bbox = [adjust_bbox_for_transform_no_scale(img, box, model.resize_dim, model.resize_dim) for box in bbox]
            if len(adjusted_bbox) < n_max_boxes:
                adjusted_bbox += [[-1, -1, -1, -1]] * (n_max_boxes - len(adjusted_bbox)) # add dummy bbox for batch computation
            adjusted_bboxes.append(adjusted_bbox)
        adjusted_bboxes = torch.tensor(adjusted_bboxes)
        
        with torch.no_grad():
            outs = model(batch_imgs,
                        get_cls_capt=caption_from == 'cls',
                        bboxes=adjusted_bboxes.to(device),
                        gaussian_avg=use_gaussian_weighting, 
                        gaussian_bbox_variance=gaussian_variance,
                        double_DINO_for_bboxes=double_dino_last_layer,
                        double_DINO_for_bboxes_return_type=double_dino_feature_computation,
                        cleaning_type=representation_cleaning_type,
                        alpha=representation_cleaning_alpha,
                        clean_from=representation_cleaning_clean_from,
                        clean_after_projection=representation_cleaning_clean_after_projection,
                        caption_bboxes_type=caption_bboxes_type,
                        return_n_best_sims=keep_n_best_sims,
                        get_controllable_capts=True,
                        use_attn_map_for_bboxes=use_attn_map_for_bboxes
            )
        preds += outs['set_controllable_capts'] if caption_from == 'patches' else outs['cls_capt']

    result_data = {
        "model_name": model_name,
        "evaluation_dataset": evaluation_dataset,
        "use_gaussian_weighting": use_gaussian_weighting,
        "gaussian_variance": gaussian_variance,
        "keep_img_ratio": keep_img_ratio,
        "caption_bboxes_type": caption_bboxes_type,
        "double_dino_last_layer": double_dino_last_layer,
        "double_dino_feature_computation": double_dino_feature_computation,
        "representation_cleaning_type": representation_cleaning_type,
        "representation_cleaning_alpha": representation_cleaning_alpha,
        "representation_cleaning_clean_from" : representation_cleaning_clean_from,
        "representation_cleaning_clean_after_projection" : representation_cleaning_clean_after_projection,
        "caption_from": caption_from,
        "use_attn_map_for_bboxes" : use_attn_map_for_bboxes
    }
    
    results = {
        'predictions': [caption.replace("<|startoftext|>", "").replace("<|endoftext|>", "").strip(" .") for caption in preds],
        'gt_captions': samples['gt_captions'],
        'bboxes': samples['bboxes'],
        'img_paths': samples['img_paths'],
        'config_data': result_data
    }
    
    # Save predictions to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f)
        
    print(f"Predictions saved to {output_file}")
    return output_file

from compute_scores import compute_and_stores_scores_main

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with specified configurations.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--evaluation_dataset", type=str, required=True, help="Path to the evaluation dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--use_gaussian_weighting", action="store_true", help="Whether to use Gaussian weighting.")
    parser.add_argument("--gaussian_variance", type=float, default=1.0, help="Variance for Gaussian weighting.")
    parser.add_argument("--keep_img_ratio", action="store_true", help="Whether to keep the image aspect ratio. False -> does not crop the image, resize the input image to (resize_dim, resize_dim), True  -> resize the image keeping ratio, then crops it")
    parser.add_argument("--caption_bboxes_type", type=str, default=None, help="Type of bounding boxes for captions, to be set to compute dense caption as bbox crop caption (default None, allowed values: 'avg_self_attn_capt' 'cls_capt').")
    parser.add_argument("--double_dino_last_layer", action="store_true", help="Whether to double the DINO last layer.")
    parser.add_argument("--keep_n_best_sims", type=int, default=None, help="Number of similairties to save in the prediction file (only for visualization purpose)")
    parser.add_argument("--double_dino_feature_computation", type=str, default="avg", 
                        help="Feature computation strategy for doubled DINO layer (default: 'avg').")
    parser.add_argument("--representation_cleaning_type", type=str, default=None, 
                        help="Type of representation cleaning (default: None).")
    parser.add_argument("--representation_cleaning_alpha", type=float, default=0.4, 
                        help="Alpha value for representation cleaning.")
    parser.add_argument("--representation_cleaning_clean_from", type=str, default="cls", 
                        help="clean_from : 'cls' | 'avg_self_attn'")
    parser.add_argument("--representation_cleaning_clean_after_projection", action="store_true", 
                        help="If set, it first projects the patch embeddings and general token in textual space and then applies cleaning.")
    parser.add_argument("--caption_from", type=str, default='patches', 
                        help="In case is equal to cls, it uses the caption from the cls instead of the patches")
    parser.add_argument("--configs_dir", type=str, default="../configs", help="Directory for configuration files.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--use_attn_map_for_bboxes", action="store_true", default=False, help="Whether to use the attention map for weighting the patches in the bounding boxes")

    parser.add_argument("--csv_scores_output", type=str, default="evaluation_results.csv")


    args = parser.parse_args()


    output_file_path = evaluate(
        model_name=args.model_name,
        evaluation_dataset=args.evaluation_dataset,
        batch_size=args.batch_size,
        use_gaussian_weighting=args.use_gaussian_weighting,
        gaussian_variance=args.gaussian_variance,
        keep_img_ratio=args.keep_img_ratio,
        caption_bboxes_type=args.caption_bboxes_type,
        double_dino_last_layer=args.double_dino_last_layer,
        double_dino_feature_computation=args.double_dino_feature_computation,
        representation_cleaning_type=args.representation_cleaning_type,
        representation_cleaning_alpha=args.representation_cleaning_alpha,
        representation_cleaning_clean_from=args.representation_cleaning_clean_from,
        representation_cleaning_clean_after_projection=args.representation_cleaning_clean_after_projection,
        caption_from=args.caption_from,
        configs_dir=args.configs_dir,
        keep_n_best_sims=args.keep_n_best_sims,
        device=args.device,
        use_attn_map_for_bboxes=args.use_attn_map_for_bboxes
    )

    scores_df = compute_and_stores_scores_main(output_file_path, args.device, args.csv_scores_output)

if __name__ == '__main__':
    main()
