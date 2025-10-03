from pycocotools.coco import COCO
from tqdm import tqdm
import numpy as np
import torch
import json
import os, sys

import argparse

from PIL import Image

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
    parser = argparse.ArgumentParser(description="Evaluate captioning annotations.")
    parser.add_argument(
        "--evaluated_predictions_path", 
        required=True, 
        type=str, 
        help="Path to the evaluated annotations file (JSON format)."
    )
    parser.add_argument(
        "--eval_dataset_name", 
        default="coco", 
        type=str, 
        choices=["coco", "flickr30k"], 
        help="Name of the evaluation dataset."
    )
    parser.add_argument(
        "--store_associated_gt_capts", 
        type=str2bool, default=True,
        help="Flag to store associated ground truth captions."
    )
    parser.add_argument(
        "--store_single_scores", 
        type=str2bool, default=True, 
        help="Flag to store single scores for each annotation."
    )
    parser.add_argument(
        "--limit",
        default=-1,
        type=int,
        help="For debug purposes, limits the number of images to be evaluated."
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for RefPAC-S score computation")
    return parser.parse_args()

import clip
from speaksee.evaluation import Bleu, Rouge, Cider, Spice, Meteor
from speaksee.evaluation import PTBTokenizer
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pacsMetric.pac_score import RefPACScore, PACScore

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


def get_img_path(image_dataset_path, img):
    if 'coco' in image_dataset_path:
        img_path = os.path.join(image_dataset_path, img['file_name'].split('_')[1], img['file_name'])
    if 'flickr' in image_dataset_path:
        img_path = os.path.join(image_dataset_path, img['file_name'])
    return img_path

def evaluate_captioning(annotations_path, reference_dataset_path, image_dataset_path=None, store_associated_gt_capts=True, store_single_scores=True, limit=-1, device="cuda"):
    
    if limit is not None and limit > 0:
        raise Exception("Limit is not implemented yet.")
    
    # Load the reference dataset
    reference_dataset = COCO(reference_dataset_path)

    # Load the evaluated annotations
    evaluated_annotations = COCO(annotations_path)

    output_file = annotations_path.replace(".json", "_evaluated.json")

    # load score computation models

    clip_model_name = "ViT-B/32"

    _MODELS = {
            "ViT-B/32": "/raid/datasets/models_weights/pacs-metric/clip_ViT-B-32.pth",
            "open_clip_ViT-L/14": "/raid/datasets/models_weights/pacs-metric/openClip_ViT-L-14.pth"
        }

    pacS_model, clip_preprocess = clip.load(clip_model_name, device=device)
    clips_model, clips_preprocess = get_clipscore_model(device)

    pacS_model = pacS_model.to(device)
    pacS_model = pacS_model.float()

    checkpoint = torch.load(_MODELS[clip_model_name], map_location=device)
    pacS_model.load_state_dict(checkpoint['state_dict'])
    pacS_model.eval()

    # here we have to iterate over the annotations and compute the scores for each of them

    candidates_captions = []
    references_captions = [] # list of lists
    reference_images = []

    for img in evaluated_annotations.imgs.values():
        # get the reference captions for the image
        ref_ann_ids = reference_dataset.getAnnIds(imgIds=img['id'])
        ref_anns = reference_dataset.loadAnns(ref_ann_ids)
        ref_capts = [ann['caption'] for ann in ref_anns]

        ann_ids = evaluated_annotations.getAnnIds(imgIds=img['id'])
        anns = evaluated_annotations.loadAnns(ann_ids)
        assert len(anns) == 1 # we assume that there is only one annotation per image
        candidate = anns[0]['caption'] #candidates = [ann['caption'] for ann in anns]
        candidates_captions.append(candidate)
        references_captions.append(ref_capts)

        # Loading the images (only to calculate PacScore)
        if image_dataset_path is not None:
            img_path = get_img_path(image_dataset_path, img)
            reference_images.append(img_path)

    scores_dict = {}

    gts_t = PTBTokenizer.tokenize(references_captions)
    gen_t = PTBTokenizer.tokenize(candidates_captions)

    print(f"Going to compute CLIPScores")
    val_clip_score, clip_score_per_instance = get_CLIPScore(
        clips_model, clips_preprocess, reference_images, candidates=candidates_captions, device=device, cache_file=f"{os.path.basename(reference_dataset_path)}_clips.hdf5")
    val_clip_score, clip_score_per_instance = float(val_clip_score), [float(x) for x in clip_score_per_instance]
    scores_dict['CLIP-S'] = val_clip_score
    print(f"CLIP-S: {val_clip_score}")

    print(f"Going to compute BLEU scores")
    aggregated_bleu, per_instance_bleu = Bleu(n=4).compute_score(gts_t, gen_t)
    method = ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']
    for metric, score in zip(method, aggregated_bleu):
        scores_dict[metric] = score

    print(f"Going to compute METEOR scores")
    val_meteor, per_instance_meteor = Meteor().compute_score(gts_t, gen_t)
    scores_dict['METEOR'] = val_meteor
    
    print(f"Going to compute ROUGE_L scores")
    aggregated_rouge,  per_instance_rouge = Rouge().compute_score(gts_t, gen_t)
    aggregated_rouge,  per_instance_rouge = float(aggregated_rouge), [float(s) for s in per_instance_rouge]
    scores_dict['ROUGE_L'] = aggregated_rouge

    print(f"Going to compute CIDEr scores")
    aggregated_cider,  per_instance_cider = Cider().compute_score(gts_t, gen_t)
    aggregated_cider, per_instance_cider = float(aggregated_cider), [float(s) for s in per_instance_cider]
    scores_dict['CIDEr'] = aggregated_cider

    # Compute SPICE score
    print(f"Going to compute SPICE scores")
    aggregated_spice,  per_instance_spice = Spice().compute_score(gts_t, gen_t)
    scores_dict['SPICE'] = aggregated_spice

    len_candidates = [len(c.split()) for c in candidates_captions]
    
    print(f"Going to compute PACScores")
    val_pac_score, pac_score_per_instance, _, len_candidates = PACScore(
        pacS_model, clip_preprocess, reference_images, candidates=candidates_captions, device=device, w=2.0)
    val_pac_score, pac_score_per_instance = float(val_pac_score), [float(x) for x in pac_score_per_instance]
    scores_dict['PAC-S'] = val_pac_score

    references_list_of_list = references_captions
    print(f"Going to compute RefPACScores")
    val_ref_pac_score, ref_pac_score_per_instance = RefPACScore(pacS_model, references=references_list_of_list, candidates=candidates_captions, device=device, len_candidates=len_candidates)
    # convert from np.float32 to float
    val_ref_pac_score, ref_pac_score_per_instance = float(val_ref_pac_score), [float(s) for s in ref_pac_score_per_instance]
    scores_dict['RefPAC-S'] = val_ref_pac_score

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

    for img in evaluated_annotations.imgs.values():
        ann_ids = evaluated_annotations.getAnnIds(imgIds=img['id'])
        anns = evaluated_annotations.loadAnns(ann_ids)
        assert len(anns) == 1
        anns[0]['scores'] = {}
        anns[0]['scores']['RefPAC-S'] = ref_pac_score_per_instance.pop(0)
        anns[0]['scores']['PAC-S'] = pac_score_per_instance.pop(0)
        anns[0]['scores']['CLIP-S'] = clip_score_per_instance.pop(0)
        anns[0]['scores']['METEOR'] = per_instance_meteor.pop(0)
        anns[0]['scores']['Bleu_1'] = per_instance_bleu[0].pop(0)
        anns[0]['scores']['Bleu_2'] = per_instance_bleu[1].pop(0)
        anns[0]['scores']['Bleu_3'] = per_instance_bleu[2].pop(0)
        anns[0]['scores']['Bleu_4'] = per_instance_bleu[3].pop(0)
        anns[0]['scores']['ROUGE_L'] = per_instance_rouge.pop(0)
        anns[0]['scores']['CIDEr'] = per_instance_cider.pop(0)
        anns[0]['scores']['SPICE'] = per_instance_spice.pop(0)

        if store_associated_gt_capts:
            anns[0]['gt_captions'] = references_list_of_list.pop(0)
    

    # store the scores in the annotations
    if store_single_scores:
        output_dataset = evaluated_annotations.dataset
        # store in output dataset also the global scores
        output_dataset['scores'] = scores_dict
        with open(output_file, 'w') as f:
            json.dump(output_dataset, f)

    return scores_dict

datasets_paths_dict = {
    'coco' : '/raid/datasets/coco/coco_annotations/captions_val2014.json',
    'flickr30k' : '/raid/datasets/flickr30kannotations/flickr30k/flickr30k_test_coco.json',
}

images_paths_dict = {
    'coco' : '/raid/datasets/coco',
    'flickr30k' : '/raid/datasets/flickr30k-images',
}

def main():
    args = parse_arguments()

    gt_dataset_path = datasets_paths_dict[args.eval_dataset_name]
    image_dataset_path = images_paths_dict[args.eval_dataset_name]

    scores_dict = evaluate_captioning(args.evaluated_predictions_path, gt_dataset_path, image_dataset_path=image_dataset_path,
                        store_associated_gt_capts=args.store_associated_gt_capts, store_single_scores=args.store_single_scores, limit=args.limit, device=args.device)
    
    print(json.dumps(scores_dict, indent=4))

if __name__ == '__main__':
    main()