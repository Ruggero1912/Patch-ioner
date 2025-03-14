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
from denseCapEvaluator import DenseCapEvaluator
from src.bbox_utils import adjust_bbox_for_transform, adjust_bbox_for_transform_no_scale
from src.model import Patchioner


evaluator = DenseCapEvaluator()
dense_captioning_annotations_folder = os.getenv("DENSE_CAPTIONING_ANNOTATIONS_FOLDER")


# Function to load image
def load_image(image_path):
    return Image.open(image_path).convert("RGB")

loaded_from_alt_cnt = 0
skipped_cnt = 0

def get_pil_image(tmp_file_name, images_path, alt_images_path, evaluation_dataset) -> Image.Image:
    #print(filename)
    if evaluation_dataset == "refcocog":
        filename = f"{str(tmp_file_name.split('_')[-1].replace('.jpg', '')).zfill(12)}.jpg"
    else:
        filename = tmp_file_name

    image_path = os.path.join(images_path, filename)
    try:
        image = load_image(image_path)
    except Exception as e:
        if isinstance(e, KeyboardInterrupt): raise
        alt_image_path = os.path.join(alt_images_path, filename)

        try:
            image = load_image(alt_image_path)
            global loaded_from_alt_cnt
            loaded_from_alt_cnt += 1
        except Exception as e:
            if isinstance(e, KeyboardInterrupt): raise e
            print(f"\rskipping img '{filename}'", end="")
            global skipped_cnt
            skipped_cnt += 1
            return None
    return image


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
             configs_dir="../configs",
             keep_n_best_sims=None,
             overwrite=True,
             device = "cuda" if torch.cuda.is_available() else 'cpu',
             compute_scores=False,
             limit=None,
             caption_from='patches',
             use_attn_map_for_bboxes=False
             ):
    """
    - returns output_file if completed, otherwise None
    """
    load_dotenv()
    
    crop_str = "CROP" if keep_img_ratio else "NO-CROP"
    if caption_bboxes_type is None:
        if not double_dino_last_layer:
            gaussian_str = f"-GAUSSIAN-var_{gaussian_variance}" if use_gaussian_weighting else ""
        else:
            gaussian_str = f"DOUBLEDINO-{double_dino_feature_computation}"
        if representation_cleaning_type is not None:
            gaussian_str += f"-{representation_cleaning_type}-{representation_cleaning_alpha}-{representation_cleaning_clean_from}-{'after-proj' if representation_cleaning_clean_after_projection else 'before-proj'}"
        gaussian_str += f"captioning-from-{caption_from}"
    else:
        gaussian_str = f"captioning-of-bboxes_{caption_bboxes_type}"
    output_file = f"./annotations/predictions_{model_name}-{evaluation_dataset}-coco-format-new-{crop_str}{gaussian_str}-new.json"
    output_file = os.path.abspath(output_file)
    model_config = os.path.join(configs_dir,f"{model_name}.yaml")
    
    print("\n=== Parameter Recap ===")
    print(f"Model Config: {model_config}")
    print(f"Evaluation Dataset: {evaluation_dataset}")
    print(f"Caption From: {caption_from}")
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
    print(f"Out File: {output_file}")
    print("=======================\n")

    if os.path.exists(output_file):
        print("[!] annotation file already exists [!]")
        if overwrite is False:
            print(F"Returning without recomputing it")
            return output_file
        else:
            print(F"Going to overwrite the existing annotations")

    model = Patchioner.from_config(model_config, device=device)

    dense_captioning_annotations_folder = os.getenv("DENSE_CAPTIONING_ANNOTATIONS_FOLDER")

    if evaluation_dataset == "refcocog":
        # Dataset path  # dense-captioning-evaluation
        anns_path = os.path.join(dense_captioning_annotations_folder, "refcoco/controlcap/refcocog_val.json")#"data/refcocog/instances.json"  # Update with actual path
    elif evaluation_dataset == "vg12":
        anns_path = os.path.join(dense_captioning_annotations_folder, "vg/controlcap/vg1.2/test.json")
    elif evaluation_dataset == "vg12-GRiTbbox":
        anns_path = os.path.join('./', "annotations/GRIT_vg_instances_results-coco-format.json")
    elif evaluation_dataset == "vgcoco":
        anns_path = os.path.join(dense_captioning_annotations_folder, "vg/controlcap/vgcoco/test.json")


    # Load RefCOCOg dataset
    coco = COCO(anns_path)
    image_ids = list(coco.imgs.keys()) if (limit is None) or (limit <= 0) else list(coco.imgs.keys())[:limit]
    
    if evaluation_dataset == "refcocog":
        images_path = "/raid/datasets/coco/test2017"
        alt_images_path = "/raid/datasets/coco/train2017"#"../../coco-dataset/val2014/"
        
    elif evaluation_dataset in ["vg12", "vg12-GRiTbbox", "vgcoco"]:
        images_path = "/raid/datasets/vg1.2/VG_100K_2"
        alt_images_path = "/raid/datasets/vg1.2/VG_100K"

    

    image_ids_set = set()
    annotation_id = 1
    skipped_cnt = 0
    loaded_from_alt_cnt = 0


    coco_format_predictions = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "dense_caption"}]
    }


    with torch.no_grad():
        print("Starting the features extraction...")
        n_imgs = len(image_ids)
        n_batch = math.ceil(n_imgs / batch_size)
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = start + batch_size if i < n_batch - 1 else n_imgs
            raw_imgs = []

            adjusted_bboxes_per_batch = []
            plain_bboxes_per_batch = []

            for j in range(start, end):
                image_id = image_ids[j]
                image_info = coco.loadImgs(image_id)[0]
                tmp_file_name = image_info['file_name']
                pil_img = get_pil_image(tmp_file_name, images_path, alt_images_path, evaluation_dataset)
                
                if pil_img.mode != 'RGB':
                    pil_img = pil_img.convert('RGB')
                
                raw_imgs.append(pil_img)

                if image_id not in image_ids_set:
                    coco_format_predictions["images"].append({
                        "id": image_id,
                        "file_name": tmp_file_name
                    })
                    image_ids_set.add(image_id)

                # Get all annotations for this image
                annotations = coco.imgToAnns[image_id]

                adjusted_bboxes_per_image = []
                plain_bboxes_per_image = []
                
                for ann in annotations:
                    x1, y1, x2, y2 = ann['bbox']
                    if x1 == x2: x2 += 1
                    if y1 == y2: y2 += 1
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        print(f"Invalid bounding box {ann['bbox']} for image {image_id}")
                        raise Exception
                        #skipped_cnt += 1
                        #continue
                    bbox = [x1,y1, x2-x1, y2-y1]
                    plain_bboxes_per_image.append(bbox)
                    if keep_img_ratio:
                        adjusted_bbox = adjust_bbox_for_transform(pil_img, bbox, resize_dim=model.resize_dim, crop_dim=model.crop_dim)
                    else:
                        adjusted_bbox = adjust_bbox_for_transform_no_scale(pil_img, bbox, model.resize_dim, model.resize_dim)
                    
                    adjusted_bboxes_per_image.append(adjusted_bbox)

                plain_bboxes_per_batch.append(plain_bboxes_per_image)
                adjusted_bboxes_per_batch.append(adjusted_bboxes_per_image)
            
            try:
                max_len = max([len(adj_bb_img) for adj_bb_img in adjusted_bboxes_per_batch])
            except Exception as e:
                print("WWWW")
                raise e

            for i in range(len(adjusted_bboxes_per_batch)):
                cur_len = len(adjusted_bboxes_per_batch[i])
                if cur_len < max_len:
                    adjusted_bboxes_per_batch[i].extend([ [0,0,1,1] ] * (max_len - cur_len))
                    plain_bboxes_per_batch[i].extend([ [0,0,1,1] ] * (max_len - cur_len))

            plain_bboxes_per_batch = torch.tensor(plain_bboxes_per_batch)
            adjusted_bboxes_per_batch = torch.tensor(adjusted_bboxes_per_batch)


            if keep_img_ratio:
                batch_imgs = torch.stack([model.image_transforms(img) for img in raw_imgs]).to(device)
            else:
                batch_imgs = torch.stack([model.image_transforms_no_crop(img) for img in raw_imgs]).to(device)

            if caption_bboxes_type:
                output = model.caption_bboxes(raw_imgs, plain_bboxes_per_batch, crop_boxes=True)
                #output = {'bbox_capts' : output}
            elif caption_from in ['cls', 'avg_self_attn']:
                output = model.forward(batch_imgs, 
                                       get_cls_capt=(caption_from == 'cls'),
                                       get_avg_self_attn_capt=(caption_from == 'avg_self_attn'),
                                       compute_scores=compute_scores)
            else:
                output = model.forward(batch_imgs, 
                                    bboxes=adjusted_bboxes_per_batch, get_cls_capt=False,
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
                                    compute_scores=compute_scores,
                                    use_attn_map_for_bboxes=use_attn_map_for_bboxes)

            
            for offset, j in enumerate(range(start, end)):
                image_id = image_ids[j]
                annotations = coco.imgToAnns[image_id]

                for foo, ann in enumerate(annotations):
                    if caption_from == 'patches':
                        caption = output['bbox_capts'][offset][foo]
                        score = output['bbox_scores'][offset][foo] if(compute_scores is True and 'bbox_scores' in output.keys()) else 1.0
                    elif caption_from == 'cls':
                        caption = output['cls_capt'][offset]
                        score = output['cls_capt_scores'][offset]
                    elif caption_from == 'avg_self_attn':
                        caption = output['avg_self_attn_capt'][offset]
                        score = output['avg_self_attn_capt_scores'][offset]
                    caption = caption.replace("<|startoftext|>", "").replace("<|endoftext|>", "")
                    caption = caption.strip(" .")
                    
                    if keep_n_best_sims is not None:
                        projection_sims = output['bbox_sims'][foo]
                    else:
                        projection_sims = None
                    
                    coco_format_predictions["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": ann['bbox'],
                        "caption": caption, 
                        "score" : score,
                        "projection_sims": projection_sims 
                    })
                    annotation_id += 1

    # Save predictions to a JSON file
    with open(output_file, "w") as f:
        json.dump(coco_format_predictions, f)

    print(f"Predictions saved to {output_file} - {skipped_cnt = } - {loaded_from_alt_cnt = }")
    return output_file

# Function to load the DataFrame from file (if it exists), or create a new one
def load_or_create_df(csv_file_path) -> pd.DataFrame:
    if os.path.exists(csv_file_path):
        # Load the DataFrame from the existing file
        df = pd.read_csv(csv_file_path).fillna("")
    else:
        # Create a new DataFrame if the file doesn't exist
        df = pd.DataFrame(columns=[
            "model_name", "evaluation_dataset", "caption_from", "use_gaussian_weighting", 
            "gaussian_variance", "keep_img_ratio", "caption_bboxes_type", 
            "double_dino_last_layer", "double_dino_feature_computation", 
            "representation_cleaning_type", "representation_cleaning_alpha", "map_score", "use_attn_map_for_bboxes"
        ])
    return df

def get_combination_row(df : pd.DataFrame, combination):
    filter_condition = True
    for key, value in combination.items():
        if value is None: value = ""
        if key not in df.columns: return None
        filter_condition &= (df[key] == value)  # Logical AND across all conditions
    return df[filter_condition]

# Check if the combination already exists in the DataFrame
def is_combination_existing(df, combination):
    comb_row = get_combination_row(df, combination)
    if comb_row is None:
        return False
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with specified configurations.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--evaluation_dataset", type=str, required=True, help="Path to the evaluation dataset.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation.")
    parser.add_argument("--use_gaussian_weighting", action="store_true", help="Whether to use Gaussian weighting.")
    parser.add_argument("--gaussian_variance", type=float, default=1.0, help="Variance for Gaussian weighting.")
    parser.add_argument("--keep_img_ratio", action="store_true", help="Whether to keep the image aspect ratio. False -> does not crop the image, resize the input image to (resize_dim, resize_dim), True  -> resize the image keeping ratio, then crops it")
    parser.add_argument("--caption_bboxes_type", type=str, default=None, help="Type of bounding boxes for captions, to be set to compute dense caption as bbox crop caption (default None, allowed values: 'avg_self_attn_capt' 'cls_capt').", choices=['avg_self_attn_capt', 'cls_capt', None])
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
    parser.add_argument("--representation_cleaning_clean_after_projection", type=str2bool, default=True, 
                        help="bool - if True, it first projects the patch embeddings and general token in textual space and then apply cleaning")
    parser.add_argument("--configs_dir", type=str, default="../configs", help="Directory for configuration files.")
    parser.add_argument("--compute_scores", type=str2bool, default=True, help="Wether to compute the densecaptioning MAP score over the test dataset")
    parser.add_argument("--compute_scores_verbose", type=str2bool, default=False)
    parser.add_argument("--overwrite", type=str2bool, default=True)
    parser.add_argument("--overwrite_inference", type=str, default=None, choices=['true', 'false', '0', '1', None])
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--compute_predictions_scores", type=str2bool, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--caption_from", type=str, default='patches', 
                        help="['cls', 'patches', 'avg_self_attn'] In case is equal to cls, it uses the caption from the cls instead of the patches", choices=['cls', 'patches', 'avg_self_attn'])
    parser.add_argument("--use_attn_map_for_bboxes", type=str2bool, default=False, help="Whether to use the attention map for the bounding boxes")

    args = parser.parse_args()

    overwrite_inference = args.overwrite if args.overwrite_inference is None else True if args.overwrite_inference in ['true', '1'] else False

    result_data = {
        "model_name": args.model_name,
        "evaluation_dataset": args.evaluation_dataset,
        "caption_from" : args.caption_from,
        "use_gaussian_weighting": args.use_gaussian_weighting,
        "gaussian_variance": args.gaussian_variance,
        "keep_img_ratio": args.keep_img_ratio,
        "caption_bboxes_type": args.caption_bboxes_type,
        "double_dino_last_layer": args.double_dino_last_layer,
        "double_dino_feature_computation": args.double_dino_feature_computation,
        "representation_cleaning_type": args.representation_cleaning_type,
        "representation_cleaning_alpha": args.representation_cleaning_alpha,
        "representation_cleaning_clean_from" : args.representation_cleaning_clean_from,
        "representation_cleaning_clean_after_projection" : args.representation_cleaning_clean_after_projection,
        "use_attn_map_for_bboxes" : args.use_attn_map_for_bboxes,
    }

    csv_file_path = "./annotations/evaluation_results.csv"
    # Load or create the DataFrame at the start
    df = load_or_create_df(csv_file_path)

    if is_combination_existing(df, result_data):
        print(f"[!] The provided combination was already evaluated [!]")

        comb_row = get_combination_row(df, result_data)
        print(comb_row)

        if args.overwrite == False:
            print(f"[-] stopping [-]")
            return
        else:
            print(f"[!] going to add duplicate row [!]")

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
        configs_dir=args.configs_dir,
        overwrite=overwrite_inference,
        keep_n_best_sims=args.keep_n_best_sims,
        device=args.device,
        compute_scores=args.compute_predictions_scores,
        limit=args.limit,
        caption_from=args.caption_from,
        use_attn_map_for_bboxes=args.use_attn_map_for_bboxes
    )
    if output_file_path is None:
            print(f"predictions generation did not completed successfully")
            return
    
    if args.compute_scores:
        print(f"Going to compute the scores for the generated predictions...")
        
        
        USE_SUBPROCESS = False
        
        if USE_SUBPROCESS:
            parameters = [f"--evaluated_annotations_file_path {output_file_path} --eval_dataset_name {args.evaluation_dataset} --device {args.device}"]
            if args.limit is not None and args.limit > 0:
                parameters.append(f" --limit {args.limit}")
            script_path = os.path.abspath("./eval_densecap_score_computation.sh")
            
            try:
                print(f"going to launch the script '{script_path}'")
                process = subprocess.Popen(
                    ["bash", script_path, *parameters],  # Command to execute
                    #check=True,                         # Raise an exception on error
                    text=True,                          # Decode output as text
                    #capture_output=True, (for subprocess.run method) # Capture stdout and stderr
                    stdout=subprocess.PIPE,  # Capture stdout
                    stderr=subprocess.STDOUT
                )
                output_lines = []
                for line in process.stdout:
                    if args.compute_scores_verbose:
                        print(line, end="")
                    output_lines.append(line)
                
                process.wait()

                # Check if the process completed successfully
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, process.args)

                #print("Script output:", "".join(output_lines))
                map_score = output_lines[-1].strip()
                scores_dict = json.loads(output_lines[-3].strip())
                
            except subprocess.CalledProcessError as e:
                print(f"Error occurred during nested script execution!")
                print(f"Executed command: {e.cmd}")
                #print(f"cmd list: ", ["bash", script_path, *parameters])
                print("Error:", e.stderr)
                print("Output:", e.stdout)
        else: # case without subprocess
            from eval_densecap_score_computation import eval_densecap_compute_scores

            metrics = eval_densecap_compute_scores(evaluated_annotations_file_path=output_file_path, eval_dataset_name=args.evaluation_dataset, 
                                                   device=args.device, limit=(args.limit) if args.limit is not None else -1,
                                                   print_results=False)
            map_score = metrics["map"]
            scores_dict = metrics
        

        
        result_data["map_score"] = map_score
        for k, v in scores_dict['global_scores'].items():
            result_data[k] = v
        print("")
        print(f"MAP score: ", map_score)
        print("")
        print(json.dumps(result_data, indent=2))
        print("")
        df = load_or_create_df(csv_file_path) # reload the dataset from file if already exists, so that if it was modified during the script execution it will update the most recent version
        df = pd.concat([df, pd.DataFrame([result_data])], ignore_index=True)
        df.to_csv(csv_file_path, index=False)
    
if __name__ == '__main__':
    main()
