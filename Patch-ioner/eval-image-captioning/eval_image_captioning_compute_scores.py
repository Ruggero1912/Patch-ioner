from pycocotools.coco import COCO
import tqdm
import numpy as np
import torch
import json
import os, sys

import argparse


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
from pacsMetric.pac_score import RefPACScore

def evaluate_captioning(annotations_path, reference_dataset_path, store_associated_gt_capts=True, store_single_scores=True, limit=-1, device="cuda"):
    
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

    pacS_model = pacS_model.to(device)
    pacS_model = pacS_model.float()

    checkpoint = torch.load(_MODELS[clip_model_name], map_location=device)
    pacS_model.load_state_dict(checkpoint['state_dict'])
    pacS_model.eval()

    # here we have to iterate over the annotations and compute the scores for each of them

    candidates_captions = []
    references_captions = [] # list of lists

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

    #for ann in evaluated_annotations.anns.values():


    scores_dict = {}

    gts_t = PTBTokenizer.tokenize(references_captions)
    gen_t = PTBTokenizer.tokenize(candidates_captions)

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
    
    references_list_of_list = references_captions

    val_ref_pac_score, ref_pac_score_per_instance = RefPACScore(pacS_model, references=references_list_of_list, candidates=candidates_captions, device=device, len_candidates=len_candidates)
    # convert from np.float32 to float
    val_ref_pac_score, ref_pac_score_per_instance = float(val_ref_pac_score), [float(s) for s in ref_pac_score_per_instance]
    scores_dict['RefPAC-S'] = val_ref_pac_score

    for img in evaluated_annotations.imgs.values():
        ann_ids = evaluated_annotations.getAnnIds(imgIds=img['id'])
        anns = evaluated_annotations.loadAnns(ann_ids)
        assert len(anns) == 1
        anns[0]['scores'] = {}
        anns[0]['scores']['RefPAC-S'] = ref_pac_score_per_instance.pop(0)
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

def main():
    args = parse_arguments()

    gt_dataset_path = datasets_paths_dict[args.eval_dataset_name]

    scores_dict = evaluate_captioning(args.evaluated_predictions_path, gt_dataset_path, 
                        store_associated_gt_capts=args.store_associated_gt_capts, store_single_scores=args.store_single_scores, limit=args.limit, device=args.device)
    
    print(json.dumps(scores_dict, indent=4))

if __name__ == '__main__':
    main()