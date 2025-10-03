from denseCapEvaluator import DenseCapEvaluator

from pycocotools.coco import COCO

from PIL import Image

from pycocotools.coco import COCO
import pycocotools.mask as mask_util
import tqdm
import numpy as np
import torch
import json
import os

import sys
sys.path.append('..')
from src.bbox_utils import draw_bounding_boxes

import h5py
import clip
import argparse

def preprocess(img, ann, crop: bool = False):
    # bounding_boxes (list): A list of bounding boxes, each as [x1, y1, x2, y2].
    if crop:
        assert isinstance(img, Image.Image), "img should be a PIL Image, received: {}".format(type(img))
        x1, y1, x2, y2 = ann['bbox']
        if x1 == x2:
            x1 = x2 - 1 if x2 > 0 else x2 + 1
        if y1 == y2:
            y1 = y2 - 1 if y2 > 0 else y2 + 1
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        img = img.crop((x1, y1, x2, y2))
    else:
        img = draw_bounding_boxes(img, [ann['bbox']])
    return img

def get_img_path(img_path, idx, ann, tmp_path, crop: bool = False):
    out_path = os.path.join(tmp_path, f"{idx}.jpg")
    if os.path.exists(out_path):
        return out_path
    img = preprocess(Image.open(img_path), ann, crop=crop)
    img.save(out_path)
    return out_path

import re
def extract_coco_id(filename):
    numbers = re.findall(r'\d+', filename)

    if len(numbers) > 1:
        second_number = int(numbers[1])  # Get the second number
        return second_number
    assert "Invalid filename"


base_path = "/raid/datasets/coco"
train_path = os.path.join(base_path, "train2014")
val_path = os.path.join(base_path, "val2014")
test_path = os.path.join(base_path, "test2014")
coco_filenames = {extract_coco_id(x): ('train2014/' if 'train2014' in x else 'test2014/' if 'test2014' in x else 'val2014/') + x for x in os.listdir(train_path) + os.listdir(val_path) + os.listdir(test_path)}

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function to parse arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate dense captioning annotations.")
    parser.add_argument(
        "--evaluated_annotations_file_path", 
        required=True, 
        type=str, 
        help="Path to the evaluated annotations file (JSON format)."
    )
    parser.add_argument(
        "--eval_dataset_name", 
        default="vg12", 
        type=str, 
        choices=["vg12", "vg10", "vgcoco", "refcocog"], 
        help="Name of the evaluation dataset."
    )
    parser.add_argument(
        "--store_associated_gt_capts", 
        type=str2bool, default=True,
        help="Flag to store associated ground truth captions."
    )
    parser.add_argument(
        "--store_single_meteor_scores", 
        type=str2bool, default=True, 
        help="Flag to store single meteor scores for each bounding box."
    )
    parser.add_argument(
        "--limit",
        default=-1,
        type=int,
        help="For debug purposes, limits the number of images to be evaluated."
    )
    parser.add_argument(
        "--disable_evaluate",
        action="store_true",
        help="For debug purposes, prevent the .evaluate() method to be called"
    )
    parser.add_argument("--path_to_data_folder", type=str, default="/raid/datasets/densecaptioning-annotations/data", help="path to the folder containing dense captioning annotations")
    parser.add_argument("--bbox_field", type=str, default="bbox", help="field to use for bounding boxes. Either 'bbox' or 'segmentation'")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for RefPAC-S score computation")
    return parser.parse_args()

def get_clipscore_model(device, clip_model_name = "ViT-B/32"):
    model, clip_preprocess = clip.load(clip_model_name, device=device)

    model.to(device)
    model.float()
    model.eval()

    return model, clip_preprocess



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
    text_tokens = clip.tokenize(prompts).to(device)

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


def eval_densecap_compute_scores(evaluated_annotations_file_path, 
                                 eval_dataset_name="vg12", 
                                 store_single_meteor_scores=True, 
                                 store_associated_gt_capts=True, 
                                 limit=-1, 
                                 disable_evaluate=False, 
                                 path_to_data_folder="/raid/datasets/densecaptioning-annotations/data", 
                                 bbox_field="bbox", 
                                 device="cuda" if torch.cuda.is_available() else "cpu",
                                 print_results=True,
                                 tmp_path='tmp'):
    tmp_path = f"{tmp_path}_{eval_dataset_name}"
    tmp_path_cropped = f"{tmp_path}_cropped"
    #args = parse_arguments()
    images_path = '/raid/datasets/vg1.2/VG_100K' if 'coco' not in eval_dataset_name else "/raid/datasets/coco"
    # Create tmp path to store images preprocessed for PAC Score
    os.makedirs(tmp_path, exist_ok=True)
    os.makedirs(tmp_path_cropped, exist_ok=True)

    evaluator = DenseCapEvaluator()

    EVALUATED_ANNOTATIONS_FILE_PATH = evaluated_annotations_file_path

    BBOX_FIELD = bbox_field #"bbox" # "segmentation" -> this latter case calls seg2bbox

    #print(f"{store_associated_gt_capts = }")

    EVALUATED_ANNOTATIONS_FILE_NAME = os.path.basename(EVALUATED_ANNOTATIONS_FILE_PATH)

    

    eval_annotations_file_coco_format = EVALUATED_ANNOTATIONS_FILE_PATH
    EVALUATION_RESULTS_FILE_PATH = os.path.join(os.path.dirname(EVALUATED_ANNOTATIONS_FILE_PATH), f"{''.join(EVALUATED_ANNOTATIONS_FILE_NAME.split('.')[:-1])}-global-scores.json")
    EVALUATION_RESULTS_PER_BBOX_FILE_PATH = os.path.join(os.path.dirname(EVALUATED_ANNOTATIONS_FILE_PATH), f"{''.join(EVALUATED_ANNOTATIONS_FILE_NAME.split('.')[:-1])}-single-scores.json")
    

    if os.path.exists(EVALUATION_RESULTS_FILE_PATH):
        print(f"[W] An evaluation result file output already exists, will be overwritten [W]")

    eval_annotations_file_coco_format#"predictions_refcocog.json"

    print()
    print(f"Going to use field '{BBOX_FIELD}' for bounding boxes")
    print()
    print(f"Going to evaluate the annotations from '{eval_annotations_file_coco_format}'")
    print()

    # prediction
    result = COCO(eval_annotations_file_coco_format)

    metadata_path = '../flickr_metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    coco2vg = {x['coco_id']: x['image_id'] for x in metadata if x['coco_id'] is not None}
    vg2coco = {x['image_id']: x['coco_id'] for x in metadata if x['coco_id'] is not None}

    # ground truth
    gt_dict = {"vg12": f"{path_to_data_folder}/vg/controlcap/vg1.2/test.json",
                "vg10": f"{path_to_data_folder}/vg/controlcap/vg1.0/test.json",
                "vgcoco": f"{path_to_data_folder}/vg/controlcap/vgcoco/test.json",
                "refcocog": f"{path_to_data_folder}/refcoco/controlcap/refcocog_val.json"}
    GROUND_TRUTH_FILE_PATH = gt_dict.get(eval_dataset_name, None)
    gt = COCO(GROUND_TRUTH_FILE_PATH)

    empty_pred_num = 0

    # evaluation

    # used for meteor score computation when store_single_meteor_scores is true
    candidates = []
    references_list = []
    references_images = []
    references_images_cropped = []
    recs = []
    for enum_i, (image_id, _) in tqdm.tqdm(enumerate(list(gt.imgs.items())), total=len(gt.imgs)):
        if limit > 0 and enum_i >= limit:
            print(f"Limit reached, stopping at {limit} images")
            break
        anns = gt.imgToAnns[image_id]
        rec = dict()
        target_boxes = []
        target_text = []
        for ann in anns:
            if BBOX_FIELD == "bbox":
                box = ann['bbox']
            else:
                box = seg2bbox(ann['segmentation'])
            if 'vgcoco' in eval_dataset_name:
                filename = f"{coco_filenames[vg2coco[ann['image_id']]]}"
            elif 'coco' in eval_dataset_name:
                filename = f"{coco2vg[ann['image_id']]}.jpg"
            else:
                filename = f"{ann['image_id']}.jpg"
            references_images.append(get_img_path(os.path.join(images_path, filename), 
                                                  len(references_images),
                                                  ann,
                                                  tmp_path))
            references_images_cropped.append(get_img_path(os.path.join(images_path, filename),
                                                  len(references_images_cropped),
                                                  ann,
                                                  tmp_path_cropped,
                                                  crop=True))
            
            target_boxes.append(box)
            target_text.append(ann['caption'])
        rec['target_boxes'] = target_boxes
        rec['target_text'] = target_text
        
        references_list.extend(target_text) # TODO: or .append ??  (solved below, depends on the evaluation method)
        
        preds = result.imgToAnns.get(image_id, [])
        if len(preds) == 0:
            empty_pred_num += 1
            continue
        scores = []
        boxes = []
        text = []
        for pred in preds:
            if BBOX_FIELD == "bbox":
                box = pred['bbox']
            else:
                box = seg2bbox(pred['segmentation'])

            score = pred.get('score', 1)
            caption = pred.get('caption')
            scores.append(score)
            boxes.append(box)
            text.append(caption)
        rec['scores'] = scores
        rec['boxes'] = boxes
        rec['text'] = text

        candidates.extend(text)

        rec['img_info'] = image_id
        recs.append(rec)

    print(f"LOADED ANNOTATIONS FROM FILE")


    print("Going to load methods' annotations in evaluation format")

    for rec in tqdm.tqdm(recs):
        #assert rec['boxes'] == rec['target_boxes'], f"SONO DIVERSI {rec['boxes'] = } -- {rec['target_boxes'] = }"
        try:
            evaluator.add_result(
                scores=torch.tensor(rec['scores']),
                boxes=torch.tensor(rec['boxes']),
                text=rec['text'],
                target_boxes=torch.tensor(rec['target_boxes']),
                target_text=rec['target_text'],
                img_info=rec['img_info'],
            )
            #print(f"Adding result with scores: {torch.tensor(rec['scores'])}, boxes: {torch.tensor(rec['boxes'])}, text: {rec['text']}")

        except Exception as e:
            print("sample error", e)
            #raise e
    if empty_pred_num != 0:
        print(f":Image numbers with empty prediction ({empty_pred_num}).")

    print(F"Annotations loaded, going to compute scores")

    if not disable_evaluate:
        metrics = evaluator.evaluate()
    else:
        print(f"Skipping call to .evaluate() method")

    if store_single_meteor_scores:
        
        scores_dict = {}

        print(f"Going to compute METEOR score for each bounding box")
        aggregated_meteor, gt_boxes_meteors = evaluator.meteor_score_captions(references_list, candidates)
        scores_dict['METEOR'] = aggregated_meteor
        # add here BLEU, CIDEr, ROUGE, RefPAC-S score computation
        from speaksee.evaluation import Bleu, Rouge, Cider, Spice
        from speaksee.evaluation import PTBTokenizer

        # import RefPAC-S from pacsMetric folder, in the parent folder of this script
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
        from pacsMetric.pac_score import RefPACScore, PACScore
        import clip

        clip_model_name = "ViT-B/32"
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

        clips_model, clips_preprocess = get_clipscore_model(device)


        gts_t = PTBTokenizer.tokenize(references_list)
        gen_t = PTBTokenizer.tokenize(candidates)

        print(f"Going to compute CLIPScores")
        val_clip_score, clip_score_per_instance = get_CLIPScore(
            clips_model, clips_preprocess, references_images, candidates=candidates, device=device, cache_file=f"{tmp_path}.hdf5")
        val_clip_score, clip_score_per_instance = float(val_clip_score), [float(x) for x in clip_score_per_instance]
        scores_dict['CLIP-S'] = val_clip_score
        print(f"CLIP-S: {val_clip_score}")

        # clip-score on cropped images
        print(f"Going to compute CLIPScores on cropped images")
        val_clip_score_cropped, clip_score_per_instance_cropped = get_CLIPScore(
            clips_model, clips_preprocess, references_images_cropped, candidates=candidates, device=device, cache_file=f"{tmp_path_cropped}.hdf5")
        val_clip_score_cropped, clip_score_per_instance_cropped = float(val_clip_score_cropped), [float(x) for x in clip_score_per_instance_cropped]
        scores_dict['CLIP-S_cropped'] = val_clip_score_cropped
        print(f"CLIP-S_cropped: {val_clip_score_cropped}")

        print(f"Going to compute PACScores")
        val_pac_score, pac_score_per_instance, _, len_candidates = PACScore(
            pacS_model, clip_preprocess, references_images, candidates=candidates, device=device, w=2.0)
        val_pac_score, pac_score_per_instance = float(val_pac_score), [float(x) for x in pac_score_per_instance]
        scores_dict['PAC-S'] = val_pac_score
        print(f"PAC-S: {val_pac_score}")

        # PAC-S on cropped images
        print(f"Going to compute PACScores on cropped images")
        val_pac_score_cropped, pac_score_per_instance_cropped, _, len_candidates = PACScore(
            pacS_model, clip_preprocess, references_images_cropped, candidates=candidates, device=device, w=2.0)
        val_pac_score_cropped, pac_score_per_instance_cropped = float(val_pac_score_cropped), [float(x) for x in pac_score_per_instance_cropped]
        scores_dict['PAC-S_cropped'] = val_pac_score_cropped
        print(f"PAC-S_cropped: {val_pac_score_cropped}")

        # Compute BLEU scores
        print(f"Going to compute BLEU scores")
        aggregated_bleu, per_instance_bleu = Bleu(n=4).compute_score(gts_t, gen_t)
        method = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
        for metric, score in zip(method, aggregated_bleu):
            scores_dict[metric] = score

        #print(f"{len(per_instance_bleu) = } ... {per_instance_bleu = } ---- {per_instance_bleu[0] = }")

        ## Compute METEOR score
        #aggregated_meteor, per_instance_meteor = Meteor().compute_score(gts_t, gen_t)
        #scores_dict['METEOR'] = aggregated_meteor

        # Compute ROUGE_L score
        print(f"Going to compute ROUGE_L scores")
        aggregated_rouge,  per_instance_rouge = Rouge().compute_score(gts_t, gen_t)
        scores_dict['ROUGE_L'] = aggregated_rouge

        # Compute CIDEr score
        print(f"Going to compute CIDEr scores")
        aggregated_cider,  per_instance_cider = Cider().compute_score(gts_t, gen_t)
        scores_dict['CIDEr'] = aggregated_cider

        # Compute SPICE score
        print(f"Going to compute SPICE scores")
        aggregated_spice,  per_instance_spice = Spice().compute_score(gts_t, gen_t)
        scores_dict['SPICE'] = aggregated_spice

        len_candidates = [len(c.split()) for c in candidates]
        references_list_of_list = [ [r] for r in references_list]
        val_ref_pac_score, ref_pac_score_per_instance = RefPACScore(pacS_model, references=references_list_of_list, candidates=candidates, device=device, len_candidates=len_candidates)
        # convert from np.float32 to float
        val_ref_pac_score, ref_pac_score_per_instance = float(val_ref_pac_score), [float(s) for s in ref_pac_score_per_instance]
        scores_dict['RefPAC-S'] = val_ref_pac_score

        #print(f"{type(val_ref_pac_score) = } ------- {type(ref_pac_score_per_instance) = } -------- {type(ref_pac_score_per_instance[0]) = }")
        
        #coco_format_predictions = {
        #    "images": [],
        #    "annotations": [],
        #    "categories": [{"id": 1, "name": "dense_caption"}]
        #}

        
        # Standard deviation for METEOR
        scores_dict['METEOR_std'] = float(np.std(gt_boxes_meteors))
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
        # Standard deviation for CLIP-S
        scores_dict['CLIP-S_std'] = float(np.std(clip_score_per_instance))
        # Standard deviation for PAC-S
        scores_dict['PAC-S_std'] = float(np.std(pac_score_per_instance))
        # Standard deviation for RefPAC-S
        scores_dict['RefPAC-S_std'] = float(np.std(ref_pac_score_per_instance))
        # Standard deviation for CLIP-S_cropped
        scores_dict['CLIP-S_cropped_std'] = float(np.std(clip_score_per_instance_cropped))
        # Standard deviation for PAC-S_cropped
        scores_dict['PAC-S_cropped_std'] = float(np.std(pac_score_per_instance_cropped))

        with open(EVALUATED_ANNOTATIONS_FILE_PATH) as annotations_file:
            coco_format_predictions = json.load(annotations_file)

        coco_format_predictions['scores'] = scores_dict
        metrics['global_scores'] = scores_dict

        if limit > 0:
            # should keep only the coco_format_predictions associated to the first limit images
            first_images = list(gt.imgs.keys())[:limit]
            coco_format_predictions['annotations'] = [ann for ann in coco_format_predictions['annotations'] if ann['image_id'] in first_images]

        assert len(gt_boxes_meteors) == len(coco_format_predictions['annotations']), f"[!] {len(gt_boxes_meteors) = } != {len(coco_format_predictions['annotations']) = } [!]"
        assert len(coco_format_predictions['annotations']) == len(evaluator.meteors), f"[!] {len(coco_format_predictions['annotations']) = } != {len(evaluator.meteors) = } [!]"

        print(f"Going to store meteor scores for each bbox")

        prev_img_id = None
        offset = 0
        gt_anns_for_img = None
        for i in tqdm.tqdm(range(len(coco_format_predictions['annotations']))):
            coco_format_predictions['annotations'][i]['meteor'] = evaluator.meteors[i]

            coco_format_predictions['annotations'][i]['meteor-gt-boxes'] = gt_boxes_meteors[i]
            coco_format_predictions['annotations'][i]['bleu_1'] = per_instance_bleu[0][i]
            coco_format_predictions['annotations'][i]['bleu_2'] = per_instance_bleu[1][i]
            coco_format_predictions['annotations'][i]['bleu_3'] = per_instance_bleu[2][i]
            coco_format_predictions['annotations'][i]['bleu_4'] = per_instance_bleu[3][i]
            coco_format_predictions['annotations'][i]['rouge'] = per_instance_rouge[i]
            coco_format_predictions['annotations'][i]['cider'] = per_instance_cider[i]
            coco_format_predictions['annotations'][i]['spice'] = per_instance_spice[i]
            coco_format_predictions['annotations'][i]['refpac'] = float(ref_pac_score_per_instance[i])
            coco_format_predictions['annotations'][i]['clipscore'] = float(clip_score_per_instance[i])
            coco_format_predictions['annotations'][i]['pac'] = float(pac_score_per_instance[i])

            if store_associated_gt_capts is False: continue

            if prev_img_id != coco_format_predictions['annotations'][i]['image_id']:
                prev_img_id = coco_format_predictions['annotations'][i]['image_id']
                offset = 0
                gt_anns_for_img = gt.imgToAnns[coco_format_predictions['annotations'][i]['image_id']]
            # hypothesis that coco_format_predictions and coco_format_gt are in the same order
            coco_format_predictions['annotations'][i]['gt-bbox'] = gt_anns_for_img[offset]['bbox']
            coco_format_predictions['annotations'][i]['gt-capt'] = gt_anns_for_img[offset]['caption']
            offset += 1
        
        with open(EVALUATION_RESULTS_PER_BBOX_FILE_PATH, "w") as output_file:
            json.dump(coco_format_predictions, output_file)


    print(f"[+] score computation done [+]")

    
    metrics["agg_metrics"] = metrics["map"]

    if print_results:
        print()
        print()

        print(metrics)

    with open(EVALUATION_RESULTS_FILE_PATH, "w") as scores_file:
        json.dump(metrics, scores_file)


    print(f"Stored results to file '{EVALUATION_RESULTS_FILE_PATH}'")

    if print_results:
        print(metrics["map"])

    return metrics

if __name__ == "__main__":

    args = parse_arguments()

    images_path = '/raid/datasets/vg1.2/VG_100K' if 'coco' not in args.eval_dataset_name else "/raid/datasets/coco"
    metrics = eval_densecap_compute_scores(eval_dataset_name=args.eval_dataset_name,
        evaluated_annotations_file_path=args.evaluated_annotations_file_path,
        store_single_meteor_scores=args.store_single_meteor_scores,
        store_associated_gt_capts=args.store_associated_gt_capts,
        limit=args.limit,
        disable_evaluate=args.disable_evaluate,
        path_to_data_folder=args.path_to_data_folder,
        bbox_field=args.bbox_field,
        device=args.device)