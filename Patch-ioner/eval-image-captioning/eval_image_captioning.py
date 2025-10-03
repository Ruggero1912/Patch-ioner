# Standard Library Imports
import argparse
import json
import math
import os
import sys
from copy import deepcopy
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Adjust path to include the parent directory

# Third-Party Library Imports
from PIL import Image
import numpy as np
from pycocotools.coco import COCO
import torch
import torchvision.transforms as T
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd
import time

# Try to import FLOP counting libraries
FLOP_COUNTER_AVAILABLE = False
try:
    from fvcore.nn import FlopCountAnalysis
    FLOP_COUNTER_AVAILABLE = True
    FLOP_COUNTER_TYPE = "fvcore"
except ImportError as e:
    print("Warning: No FLOP counting library found. Install fvcore, thop, or calflops for FLOP measurements.")
    print(e)
    FLOP_COUNTER_AVAILABLE = False

# Project-Specific Imports
from src.model import Patchioner

import json
import matplotlib.pyplot as plt
import os
import re
import torch
import random
import math


def count_model_flops(model, sample_input, model_kwargs=None):
    """
    Count FLOPs for a single forward pass of the model.
    
    Args:
        model: The model to analyze
        sample_input: Sample input tensor
        model_kwargs: Additional model arguments
    
    Returns:
        dict: Dictionary containing FLOP measurements
    """
    if not FLOP_COUNTER_AVAILABLE:
        return {"flops": None, "flop_counter": "unavailable"}
    
    model_kwargs = model_kwargs or {}
    
    try:
        if FLOP_COUNTER_TYPE == "fvcore":
            # Prepare inputs as expected by fvcore
            def model_forward_wrapper(*args, **kwargs):
                return model(sample_input, **model_kwargs)
            
            flops = FlopCountAnalysis(model, inputs=(sample_input))
            
            total_flops = flops.total()
        else:
            total_flops = None
            
        return {
            "flops": total_flops,
            "flop_counter": FLOP_COUNTER_TYPE,
            "flops_formatted": format_flops(total_flops) if total_flops else None
        }
        
    except Exception as e:
        print(f"Warning: FLOP counting failed with {FLOP_COUNTER_TYPE}: {e}")
        return {"flops": None, "flop_counter": f"{FLOP_COUNTER_TYPE}_failed", "error": str(e)}

def format_flops(flops):
    """Format FLOP count in human readable format."""
    if flops is None:
        return None
    
    if flops >= 1e12:
        return f"{flops / 1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops / 1e3:.2f} KFLOPs"
    else:
        return f"{flops:.0f} FLOPs"


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
             configs_dir=os.path.join(os.path.dirname(__file__), '../configs'),
             keep_n_best_sims=None,
             device = "cuda" if torch.cuda.is_available() else 'cpu',
             limit=None,
             compute_scores=True,
             result_data=None,
             measure_flops=True,
             overwrite_inference=True
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

    if overwrite_inference == False and os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping inference.")
        return output_file

    model_config = os.path.join(configs_dir,f"{model_name}.yaml")
    
    print("\n=== Parameter Recap ===")
    print(f"Model Config: {model_config}")
    print(f"Evaluation Dataset: {evaluation_dataset}")
    print(f"Batch Size: {batch_size}")
    print(f"Use Gaussian Weighting: {use_gaussian_weighting}")
    print(f"Gaussian Variance: {gaussian_variance}")
    print(f"Keep Image Aspect Ratio: {keep_img_ratio}")
    print(f"Caption from: {caption_from}")
    print(f"Measure FLOPs: {measure_flops}")
    print(f"FLOP Counter Available: {FLOP_COUNTER_AVAILABLE} ({FLOP_COUNTER_TYPE if FLOP_COUNTER_AVAILABLE else 'None'})")
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

    # Initialize timing measurements and FLOP measurement placeholder
    total_inference_time = 0.0
    total_preprocessing_time = 0.0
    inference_times = []
    preprocessing_times = []
    flop_measurements = {"flops_per_forward_pass": None}  # Initialize here

    for i in tqdm(range(n_batch)):
        start = i * batch_size
        end = start + batch_size if i < n_batch - 1 else n_imgs
        batch_size_ = end - start

        # Start preprocessing timing
        preprocess_start = time.time()

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

        # End preprocessing timing
        preprocess_end = time.time()
        preprocess_time = preprocess_end - preprocess_start
        preprocessing_times.append(preprocess_time)
        total_preprocessing_time += preprocess_time
        
        # Start inference timing
        inference_start = time.time()
        
        # Measure FLOPs on the first batch using real data
        if measure_flops and FLOP_COUNTER_AVAILABLE and i == 0:
            print(f"Measuring FLOPs using real data from first batch (batch size: {batch_size_})...")
            
            # Prepare model arguments for FLOP measurement
            model_kwargs = {
                'get_cls_capt': caption_from == 'cls',
                'get_avg_self_attn_capt': caption_from == 'avg_self_attn',
                'get_avg_patch_capt': use_gaussian_weighting, 
                'gaussian_img_variance': gaussian_variance,
                'return_n_best_sims': keep_n_best_sims,
                'compute_scores': compute_scores
            }
            
            try:
                flop_result = count_model_flops(model, batch_imgs, model_kwargs)
                flop_measurements.update(flop_result)
                
                if flop_result["flops"] is not None:
                    print(f"FLOPs per forward pass (batch size {batch_size_}): {flop_result['flops_formatted']}")
                    print(f"FLOPs per image: {format_flops(flop_result['flops'] / batch_size_)}")
                else:
                    print("FLOP measurement failed or unavailable")
                    
            except Exception as e:
                print(f"Error measuring FLOPs: {e}")
                flop_measurements["flop_error"] = str(e)
        
        elif measure_flops and not FLOP_COUNTER_AVAILABLE and i == 0:
            print("FLOP measurement requested but no FLOP counting library available.")
            flop_measurements["flops_error"] = "No FLOP counting library available"

        

        
        with torch.no_grad():
            outs = model.forward(batch_imgs,
                        get_cls_capt=caption_from == 'cls',
                        get_avg_self_attn_capt=caption_from == 'avg_self_attn',
                        get_avg_patch_capt=use_gaussian_weighting, 
                        gaussian_img_variance=gaussian_variance,
                        return_n_best_sims=keep_n_best_sims,
                        compute_scores=compute_scores
            )
        
        # End inference timing
        inference_end = time.time()
        inference_time = inference_end - inference_start
        inference_times.append(inference_time)
        total_inference_time += inference_time
        
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

    # Compute timing statistics
    avg_inference_time_per_batch = total_inference_time / n_batch
    avg_preprocessing_time_per_batch = total_preprocessing_time / n_batch
    avg_inference_time_per_image = total_inference_time / n_imgs
    # compute mean and std of preprocessing and inference times
    std_inference_time_per_image = np.std(inference_times)
    std_preprocessing_time_per_image = np.std(preprocessing_times)
    avg_preprocessing_time_per_image = total_preprocessing_time / n_imgs
    total_time = total_inference_time + total_preprocessing_time
    
    # Estimate total FLOPs if measurement was successful
    total_flops_estimate = None
    flops_per_image_estimate = None
    if flop_measurements.get("flops") is not None:
        # The FLOP measurement was done on the first batch, so we need to account for its actual batch size
        first_batch_size = min(batch_size, n_imgs)  # Size of the first batch
        flops_per_image_estimate = flop_measurements["flops"] / first_batch_size
        
        # Estimate total FLOPs based on total number of images
        total_flops_estimate = flops_per_image_estimate * n_imgs
    
    print(f"\n=== Timing Results ===")
    print(f"Total images processed: {n_imgs}")
    print(f"Total batches: {n_batch}")
    print(f"Average batch size: {n_imgs / n_batch:.1f}")
    print(f"Total preprocessing time: {total_preprocessing_time:.3f}s")
    print(f"Total inference time: {total_inference_time:.3f}s")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average preprocessing time per batch: {avg_preprocessing_time_per_batch:.3f}s")
    print(f"Average inference time per batch: {avg_inference_time_per_batch:.3f}s")
    print(f"Average preprocessing time per image: {avg_preprocessing_time_per_image:.3f}s")
    print(f"Average inference time per image: {avg_inference_time_per_image:.3f}s")
    print(f"Images per second (inference only): {n_imgs / total_inference_time:.2f}")
    print(f"Images per second (total): {n_imgs / total_time:.2f}")
    
    if flop_measurements.get("flops") is not None:
        print(f"\n=== FLOP Results (measured on real data) ===")
        first_batch_size = min(batch_size, n_imgs)
        print(f"FLOPs per forward pass (batch size {first_batch_size}): {flop_measurements.get('flops_formatted', 'N/A')}")
        print(f"Estimated FLOPs per image: {format_flops(flops_per_image_estimate)}")
        print(f"Estimated total FLOPs: {format_flops(total_flops_estimate)}")
        print(f"FLOP counter used: {flop_measurements.get('flop_counter', 'N/A')}")
        print(f"Measurement based on: Real images and annotations from first batch")
    elif measure_flops:
        print(f"\n=== FLOP Results ===")
        print("FLOP measurement failed or unavailable")
        if "error" in flop_measurements:
            print(f"Error: {flop_measurements['error']}")
    print("=======================\n")

    # Create structured result data dictionaries like in controllable captioning
    result_data_enhanced = {
        "model_name": model_name,
        "evaluation_dataset": evaluation_dataset,
        "use_gaussian_weighting": use_gaussian_weighting,
        "gaussian_variance": gaussian_variance,
        "keep_img_ratio": keep_img_ratio,
        "caption_from": caption_from,
        # Timing measurements
        "total_preprocessing_time": total_preprocessing_time,
        "total_inference_time": total_inference_time,
        "total_time": total_time,
        "avg_preprocessing_time_per_batch": avg_preprocessing_time_per_batch,
        "avg_inference_time_per_batch": avg_inference_time_per_batch,
        "avg_preprocessing_time_per_image": avg_preprocessing_time_per_image,
        "std_preprocessing_time_per_image": std_preprocessing_time_per_image,
        "avg_inference_time_per_image": avg_inference_time_per_image,
        "std_inference_time_per_image": std_inference_time_per_image,
        "images_per_second_inference": n_imgs / total_inference_time,
        "images_per_second_total": n_imgs / total_time,
        # FLOP measurements
        "flops_per_image_estimate": flops_per_image_estimate,
        "flops_per_forward_pass": flop_measurements.get("flops"),
        "flop_counter_used": flop_measurements.get("flop_counter"),
    }

    computation_details = {
        "batch_size": batch_size,
        "n_imgs": n_imgs,
        "n_batch": n_batch,
        # Timing measurements
        "total_preprocessing_time": total_preprocessing_time,
        "total_inference_time": total_inference_time,
        "total_time": total_time,
        "avg_preprocessing_time_per_batch": avg_preprocessing_time_per_batch,
        "avg_inference_time_per_batch": avg_inference_time_per_batch,
        "avg_preprocessing_time_per_image": avg_preprocessing_time_per_image,
        "std_preprocessing_time_per_image": std_preprocessing_time_per_image,
        "avg_inference_time_per_image": avg_inference_time_per_image,
        "std_inference_time_per_image": std_inference_time_per_image,
        "images_per_second_inference": n_imgs / total_inference_time,
        "images_per_second_total": n_imgs / total_time,
        "preprocessing_times": preprocessing_times,
        "inference_times": inference_times,
        # FLOP measurements
        "flop_measurements": flop_measurements,
        "total_flops_estimate": total_flops_estimate,
        "flops_per_image_estimate": flops_per_image_estimate,
        "flops_per_forward_pass": flop_measurements.get("flops"),
        "flop_counter_used": flop_measurements.get("flop_counter"),
        "flop_measurement_method": "real_data_first_batch" if flop_measurements.get("flops") is not None else "unavailable",
        "flop_measurement_batch_size": min(batch_size, n_imgs) if flop_measurements.get("flops") is not None else None
    }

    other_config_data = {
        "use_gaussian_weighting": use_gaussian_weighting,
        "gaussian_variance": gaussian_variance,
        "keep_img_ratio": keep_img_ratio,
        "caption_from": caption_from,
        "keep_n_best_sims": keep_n_best_sims,
        "compute_scores": compute_scores,
        "limit": limit
    }
    
    # Use the enhanced result_data or fallback to the passed one
    final_result_data = result_data if result_data is not None else result_data_enhanced
    
    coco_format_predictions["info"] = final_result_data
    coco_format_predictions["computation_details"] = computation_details
    coco_format_predictions["other_config_data"] = other_config_data
    
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
    parser.add_argument("--configs_dir", type=str, default=os.path.join(os.path.dirname(__file__), '../configs'), help="Directory for configuration files.")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--no_scores", action="store_true", help="Whether to not compute the scores for the captions.")
    # add synonym for --csv_output_file : --csv_scores_output
    parser.add_argument("--csv_output_file", type=str, default="./evaluation_results.csv", help="Path to the output file for the scores.")
    parser.add_argument("--csv_scores_output", dest="csv_output_file", type=str, help="(Synonym) Path to the output file for the scores.")
    parser.add_argument("--measure_flops", action="store_true", default=False, help="Whether to measure FLOPs during evaluation. Requires fvcore, thop, or calflops to be installed.")
    parser.add_argument("--overwrite_inference", action="store_true", default=False, help="Whether to overwrite the inference results if they already exist. Default is True, meaning it will overwrite existing results.")

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
        result_data=result_data,
        measure_flops=args.measure_flops,
        overwrite_inference=args.overwrite_inference
    )

    print(f"Inference completed, now evaluating the predictions at {output_file_path}")

    from eval_image_captioning_compute_scores import datasets_paths_dict, images_paths_dict

    if 'flickr30k' in args.evaluation_dataset:
        k = 'flickr30k'
    elif 'coco' in args.evaluation_dataset:
        k = 'coco'
    else:
        k = None

    references_dataset_path = datasets_paths_dict[k]
    image_dataset_path = images_paths_dict[k]
    
    global_scores_dict = evaluate_captioning(
        annotations_path=output_file_path,
        reference_dataset_path=references_dataset_path,
        image_dataset_path=image_dataset_path,
        store_associated_gt_capts=True,
        store_single_scores=True,
        device=args.device
    )

    # copy the result_data dictionary
    result_row = result_data.copy()

    # take the computation time from the json in output_file_path
    with open(output_file_path, 'r') as f:
        output_data = json.load(f)
        computation_details = output_data.get('computation_details', {})
        other_config_data = output_data.get('other_config_data', {})
    
    result_row.update(computation_details)  # Merge computation details

    result_row.update(global_scores_dict) # Merge the two dictionaries
    result_row_df = pd.DataFrame([result_row]) # Convert to DataFrame
    print("Result Row DataFrame:")
    print(result_row_df)

    csv_file_path = args.csv_output_file
    # Load or create the DataFrame at the start
    df = load_or_create_df(csv_file_path, columns=result_row.keys())
    
    df = pd.concat([df, result_row_df], ignore_index=True)
    df.to_csv(csv_file_path, index=False)

if __name__ == '__main__':
    main()
