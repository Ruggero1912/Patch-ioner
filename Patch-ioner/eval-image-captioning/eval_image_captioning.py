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
from src.model import Patchioner

import json
import matplotlib.pyplot as plt
import os
import re
import torch
import random
import math


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
             gaussian_variance=1.0,
             keep_img_ratio=True,
             caption_from='cls',
             configs_dir="../configs",
             keep_n_best_sims=None,
             device = "cuda" if torch.cuda.is_available() else 'cpu',
             limit=None,
             compute_scores=True,
             result_data=None
             ):
    
    crop_str = "CROP" if keep_img_ratio else "NO-CROP"

    gaussian_str = ""

    if use_gaussian_weighting:
        if gaussian_variance >= 100:
            gaussian_str = f"-avg-patches"
        elif gaussian_variance == 0:
            gaussian_str = F"-one-hot-center"
        else:
            gaussian_str = f"-GAUSSIAN-var_{gaussian_variance}"
    

    caption_from_str = f"caption_from_{caption_from}"

    if 'flickr30k' in evaluation_dataset:
        evaluation_dataset_name = 'flickr30k'
    elif 'coco' in evaluation_dataset:
        evaluation_dataset_name = 'coco'
    else:
        evaluation_dataset_name = 'unknown'

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
    print(f"Caption from: {caption_from}")
    print(f"Out File: {output_file}")
    print("=======================\n")
    
    model = Patchioner.from_config(model_config, device=device)
    model.eval()
    model.to(device)
    
    with open(evaluation_dataset, 'r') as fp:  
        test_set = json.load(fp)
    
    if 'flickr30k' in evaluation_dataset_name:
        base_path = '/raid/datasets/flickr30k-images'
    elif 'coco' in evaluation_dataset_name:
        base_path = "/raid/datasets/coco/val2014"
    else:
        base_path = None

    print(f"Base path: {base_path}")

    dataset_coco_obj = COCO(evaluation_dataset)
    image_ids = list(dataset_coco_obj.imgs.keys()) if (limit is None) or (limit <= 0) else list(dataset_coco_obj.imgs.keys())[:limit]

    coco_format_predictions = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "image_caption"}]
    }
    annotation_id = 1

    image_ids_set = set()
    
    preds = []
    n_imgs = len(image_ids)
    n_batch = math.ceil(n_imgs / batch_size)

    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs

        raw_imgs = []
        
        for img_i in range(start, end):
            image_id = image_ids[img_i]
            image_info = dataset_coco_obj.loadImgs(image_id)[0]
            tmp_file_name = image_info['file_name']
            img_path = os.path.join(base_path, tmp_file_name)
            #pil_img = get_pil_image(tmp_file_name, images_path, alt_images_path, evaluation_dataset)
        
            raw_imgs.append( Image.open(img_path).convert('RGB') )

            if image_id not in image_ids_set:
                        coco_format_predictions["images"].append({
                            "id": image_id,
                            "file_name": tmp_file_name
                        })
                        image_ids_set.add(image_id)

            annotations = dataset_coco_obj.imgToAnns[image_id]
        
        if keep_img_ratio:
            batch_imgs = torch.stack([model.image_transforms(img) for img in raw_imgs]).to(device)
        else:
            batch_imgs = torch.stack([model.image_transforms_no_crop(img) for img in raw_imgs]).to(device)
        

        
        with torch.no_grad():
            outs = model.forward(batch_imgs,
                        get_cls_capt=caption_from == 'cls',
                        get_avg_self_attn_capt=caption_from == 'avg_self_attn',
                        get_avg_patch_capt=use_gaussian_weighting, 
                        gaussian_img_variance=gaussian_variance,
                        return_n_best_sims=keep_n_best_sims,
                        compute_scores=compute_scores
            )
        
        for offset, j in enumerate(range(start, end)):
            image_id = image_ids[j]
            #annotations = dataset_coco_obj.imgToAnns[image_id]
            
            if use_gaussian_weighting:
                caption = outs['avg_patch_capt'][offset]
                score = outs['avg_patch_capt_scores'][offset]
            elif caption_from == 'cls':
                caption = outs['cls_capt'][offset]
                score = outs['cls_capt_scores'][offset]
            elif caption_from == 'avg_self_attn':
                caption = outs['avg_self_attn_capt'][offset]
                score = outs['avg_self_attn_capt_scores'][offset]
            
            caption = caption.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
            caption = caption.strip(" .")

            coco_format_predictions["annotations"].append({
                            "id": annotation_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "caption": caption, 
                            "score" : score
                        })
            annotation_id += 1

    
    coco_format_predictions["info"] = result_data
    
    # Save predictions to a JSON file
    with open(output_file, "w") as f:
        json.dump(coco_format_predictions, f)
        
    print(f"Predictions saved to {output_file}")
    return output_file


from eval_image_captioning_compute_scores import evaluate_captioning


def load_or_create_df(csv_file_path, columns):
    if os.path.exists(csv_file_path):
        # Load the DataFrame from the existing file
        df = pd.read_csv(csv_file_path).fillna("")
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=columns)
    return df

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with specified configurations.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--evaluation_dataset", type=str, required=True, help="Path to the evaluation dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--use_gaussian_weighting", action="store_true", help="Whether to use Gaussian weighting.")
    parser.add_argument("--gaussian_variance", type=float, default=1.0, help="Variance for Gaussian weighting.")
    parser.add_argument("--keep_img_ratio", action="store_true", help="Whether to keep the image aspect ratio. False -> does not crop the image, resize the input image to (resize_dim, resize_dim), True  -> resize the image keeping ratio, then crops it")
    parser.add_argument("--keep_n_best_sims", type=int, default=None, help="Number of similairties to save in the prediction file (only for visualization purpose)")
    parser.add_argument("--caption_from", type=str, default='cls', choices=['cls', 'avg_self_attn', 'patches'], 
                        help="In case is equal to cls, it uses the caption from the cls instead of the patches")
    parser.add_argument("--configs_dir", type=str, default="../configs", help="Directory for configuration files.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--no_scores", action="store_true", help="Whether to not compute the scores for the captions.")



    args = parser.parse_args()

    compute_scores = not args.no_scores

    result_data = {
        "model_name": args.model_name,
        "evaluation_dataset": args.evaluation_dataset,
        "use_gaussian_weighting": args.use_gaussian_weighting,
        "gaussian_variance": args.gaussian_variance,
        "keep_img_ratio": args.keep_img_ratio,
        "caption_from": args.caption_from
    }

    output_file_path = evaluate(
        model_name=args.model_name,
        evaluation_dataset=args.evaluation_dataset,
        batch_size=args.batch_size,
        use_gaussian_weighting=args.use_gaussian_weighting,
        gaussian_variance=args.gaussian_variance,
        keep_img_ratio=args.keep_img_ratio,
        caption_from=args.caption_from,
        configs_dir=args.configs_dir,
        keep_n_best_sims=args.keep_n_best_sims,
        device=args.device,
        compute_scores=compute_scores,
        result_data=result_data
    )

    print(f"Inference completed, now evaluating the predictions at {output_file_path}")

    from eval_image_captioning_compute_scores import datasets_paths_dict

    if 'flickr30k' in args.evaluation_dataset:
        k = 'flickr30k'
    elif 'coco' in args.evaluation_dataset:
        k = 'coco'
    else:
        k = None

    references_dataset_path = datasets_paths_dict[k]
    
    global_scores_dict = evaluate_captioning(
        annotations_path=output_file_path,
        reference_dataset_path=references_dataset_path,
        store_associated_gt_capts=True,
        store_single_scores=True,
        device=args.device
    )

    # copy the result_data dictionary
    result_row = result_data.copy()
    result_row.update(global_scores_dict) # Merge the two dictionaries

    csv_file_path = "./evaluation_results.csv"
    # Load or create the DataFrame at the start
    df = load_or_create_df(csv_file_path, columns=result_row.keys())
    df = pd.concat([df, pd.DataFrame([result_row])], ignore_index=True)
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    main()
