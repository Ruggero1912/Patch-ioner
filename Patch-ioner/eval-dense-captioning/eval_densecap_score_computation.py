from denseCapEvaluator import DenseCapEvaluator

from pycocotools.coco import COCO



from pycocotools.coco import COCO
import pycocotools.mask as mask_util
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

def seg2bbox(seg):
    if isinstance(seg, list):
        seq = []
        for seg_ in seg:
            seq.extend(seg_)
        x1, y1 = np.array(seq).reshape(-1, 2).min(0)
        x2, y2 = np.array(seq).reshape(-1, 2).max(0)
        bbox = [x1, y1, x2, y2]
    else:
        if isinstance(seg["counts"], list):
            seg = mask_util.frPyObjects(seg, *seg["size"])
        elif not isinstance(seg["counts"], bytes):
            seg["counts"] = seg["counts"].encode()
        mask = mask_util.decode(seg)
        x1, x2 = np.nonzero(mask.sum(0) != 0)[0][0], np.nonzero(mask.sum(0) != 0)[0][-1]
        y1, y2 = np.nonzero(mask.sum(1) != 0)[0][0], np.nonzero(mask.sum(1) != 0)[0][-1]
        bbox = [x1, y1, x2, y2]
    return bbox

def eval_densecap_compute_scores(evaluated_annotations_file_path, 
                                 eval_dataset_name="vg12", 
                                 store_single_meteor_scores=True, 
                                 store_associated_gt_capts=True, 
                                 limit=-1, 
                                 disable_evaluate=False, 
                                 path_to_data_folder="/raid/datasets/densecaptioning-annotations/data", 
                                 bbox_field="bbox", 
                                 device="cuda" if torch.cuda.is_available() else "cpu",
                                 print_results=True):
    
    #args = parse_arguments()

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

    recs = []
    for enum_i, (image_id, _) in tqdm.tqdm(enumerate(list(gt.imgs.items()))):
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
        from pacsMetric.pac_score import RefPACScore
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

        """
        # CLIP Score
        clip_model_name = "ViT-B/32"
        clip_model, clip_preprocess = clip.load(clip_model_name, device=device)
        clip_model.eval()

        def ref_clip_score(model, captions, ref_captions, device):
            model.eval()
            
            text_features = clip.tokenize(captions).to(device)
            ref_text_features = clip.tokenize(ref_captions).to(device)
            
            with torch.no_grad():
                text_features = model.encode_text(text_features)
                ref_text_features = model.encode_text(ref_text_features)
                
                text_features /= text_features.norm(dim=-1, keepdim=True)
                ref_text_features /= ref_text_features.norm(dim=-1, keepdim=True)
                
                similarities = (text_features @ ref_text_features.T).diag().tolist()
            
            return similarities
        """

        gts_t = PTBTokenizer.tokenize(references_list)
        gen_t = PTBTokenizer.tokenize(candidates)

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


    metrics = eval_densecap_compute_scores(eval_dataset_name=args.eval_dataset_name,
        evaluated_annotations_file_path=args.evaluated_annotations_file_path,
        store_single_meteor_scores=args.store_single_meteor_scores,
        store_associated_gt_capts=args.store_associated_gt_capts,
        limit=args.limit,
        disable_evaluate=args.disable_evaluate,
        path_to_data_folder=args.path_to_data_folder,
        bbox_field=args.bbox_field,
        device=args.device)