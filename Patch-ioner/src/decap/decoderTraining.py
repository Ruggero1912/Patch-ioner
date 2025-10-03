import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from im2txtprojection.im2txtprojection import Im2TxtProjector, ProjectionType
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
from decap import get_decap_model
import os
import sys
import argparse
import json
from typing import Union
import sys
import clip
import json

import csv


from src.dataset import ClipCocoDataset
from src.datasetMix import ClipCocoDatasetMix
from src.model import DeCap, ProjectionLayer

DECAP_DECODER_CONFIG_PATH = os.path.join("./decoder_config.pkl")

def save_config(args: argparse.Namespace):
    config = {}
    for key, item in args._get_kwargs():
        config[key] = item
    out_path = os.path.join(args.out_dir, f"{args.prefix}.json")
    with open(out_path, 'w') as outfile:
        json.dump(config, outfile)


def load_model(config_path: str, epoch_or_latest: Union[str, int] = '_latest'):
    with open(config_path) as f:
        config = json.load(f)
    parser = argparse.ArgumentParser()
    parser.set_defaults(**config)
    args = parser.parse_args()
    if type(epoch_or_latest) is int:
        epoch_or_latest = f"-{epoch_or_latest:03d}"
    model_path = os.path.join(args.out_dir, f"{args.prefix}{epoch_or_latest}.pt")
    if args.only_prefix:
        model = ClipCaptionPrefix(args.prefix_length)
    else:
        model = ClipCaptionModel(args.prefix_length)
    if os.path.isfile(model_path):
        print(f"loading model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    else:
        print(f"{model_path} is not exist")
    return model, parser


    

def train_decoder(args,
          lr: float = 1e-5, warmup_steps: int = 1000, output_dir: str = ".", output_prefix: str = ""):

    # device = torch.device('cuda:1')
    batch_size = args.bs
    epochs = args.epochs
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    args.is_master = ( args.local_rank == 0 or args.not_distributed != False)

    # set the device
    #torch.cuda.set_device(args.local_rank)
    #device = torch.device('cuda:'+str(args.local_rank))
    if args.not_distributed == False:
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda:'+str(args.local_rank))
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        device = torch.device('cuda:'+str(args.local_rank))
        print(f"NOT DISTRIBUTED")
    print(f"Using device {device}")
    SEED=42
    torch.cuda.manual_seed_all(SEED)
    
    if args.use_regionclip:
        # RegionCLIP typically uses 1024 dimensions for ResNet-50 or 512 for ViT
        # We'll determine this from the loaded model
        prefix_size = 1024  # Default for RegionCLIP ResNet-50, but will be adjusted if needed
    elif args.denseclip_config is not None:
        # DenseClip typically uses 512 dimensions (similar to CLIP ViT-B)
        from src.denseclip.loader import load_denseclip_config
        denseclip_config_dict = load_denseclip_config(args.denseclip_config)
        prefix_size = denseclip_config_dict.get('model', {}).get('text', {}).get('embed_dim', None)
        if prefix_size is None:
            print(f"Warning: Could not determine prefix_size from DenseClip config {args.denseclip_config}. Defaulting to 512.")
            prefix_size = 512  # Fallback to a common size)

    elif 'H' in args.clip_model or args.use_dinotxt:
        prefix_size = 1024
    elif args.talk2dino_weights is not None or args.use_dino_feats:
        prefix_size = 768
    else:
        prefix_size = 512
    
    if args.im_proj:
        memory_bank_path = os.path.abspath(args.dataset)
        print(f"Using Im2TxtProjector with {memory_bank_path = }")
        im_proj = Im2TxtProjector(
            type=memory_bank_path,
            use_talk2dino=True,
            linear_talk2dino=False,
            memory_bank_name='coco_karpathy',
            device_str=device)

    if args.use_regionclip:
        from src.regionclip.loader import load_regionclip_from_checkpoint
        from src.regionclip.datasets.clip_prompt_utils import tokenize as regionclip_tokenize
        
        print("Using RegionCLIP for text encoding.")
        if args.regionclip_checkpoint is None:
            raise ValueError("RegionCLIP checkpoint path must be provided when using --use-regionclip")
        
        clip_model = load_regionclip_from_checkpoint(
            args.regionclip_checkpoint, 
            device=device, 
            config=args.regionclip_config
        )
        tokenizer = regionclip_tokenize
        preprocess = None  # RegionCLIP doesn't need preprocessing for text-only training
        
        # Determine the actual embedding dimension from the loaded model
        if hasattr(clip_model, 'text_projection'):
            actual_prefix_size = clip_model.text_projection.shape[1]
            print(f"RegionCLIP text embedding dimension: {actual_prefix_size}")
            if actual_prefix_size != prefix_size:
                print(f"Updating prefix_size from {prefix_size} to {actual_prefix_size}")
                prefix_size = actual_prefix_size
        
        # Test RegionCLIP text encoding to ensure it works
        try:
            test_text = ["A test sentence"]
            test_tokens = tokenizer(test_text)
            test_features = clip_model.encode_text(test_tokens.to(device))
            print(f"RegionCLIP test encoding successful. Output shape: {test_features.shape}")
        except Exception as e:
            print(f"Warning: RegionCLIP test encoding failed: {e}")
            print("This might cause issues during training.")
        
    elif args.denseclip_config is not None:
        from src.denseclip.loader import load_denseclip
        
        print(f"Using DenseClip for text encoding with config: {args.denseclip_config}")
        
        try:
            clip_model = load_denseclip(
                config_name=args.denseclip_config,
                device=device
            )
            
            # Try to use DenseClip's tokenizer first
            try:
                from src.denseclip.loader import DenseCLIP_tokenize
                tokenizer = DenseCLIP_tokenize
                print("Using DenseClip tokenizer")
            except ImportError:
                # Fallback to CLIP tokenizer if DenseClip tokenizer is not available
                import clip
                tokenizer = clip.tokenize
                print("Warning: DenseClip tokenizer not available, using CLIP tokenizer")
            
            preprocess = None  # DenseClip doesn't need preprocessing for text-only training
            
            # Determine the actual embedding dimension from the loaded model
            if hasattr(clip_model, 'text_encoder') and hasattr(clip_model.text_encoder, 'embed_dim'):
                actual_prefix_size = clip_model.text_encoder.embed_dim
                print(f"DenseClip text embedding dimension: {actual_prefix_size}")
                if actual_prefix_size != prefix_size:
                    print(f"Updating prefix_size from {prefix_size} to {actual_prefix_size}")
                    prefix_size = actual_prefix_size
            
            # Test DenseClip text encoding to ensure it works
            test_text = ["A test sentence"]
            test_tokens = tokenizer(test_text)
            if hasattr(test_tokens, 'to'):
                test_tokens = test_tokens.to(device)
            test_features = clip_model.encode_text(test_tokens)
            print(f"DenseClip test encoding successful. Output shape: {test_features.shape}")
            
        except Exception as e:
            print(f"Error loading DenseClip model: {e}")
            raise e
        
    elif args.use_open_clip:
        from open_clip import create_model_and_transforms, tokenize
        print("Using open_clip for model loading.")
        clip_model, preprocess_train, preprocess_val = create_model_and_transforms(model_name=args.clip_model, pretrained="laion2b_s32b_b79k", device=device)
        preprocess = preprocess_train
        tokenizer = tokenize

    elif args.use_dinotxt:
        from src.dinotxt_utils import get_tokenizer
        clip_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l')
        tokenizer = get_tokenizer().tokenize
    else:
        clip_model, preprocess = clip.load(args.clip_model, device=device, jit=False)
        tokenizer = clip.tokenize
    clip_model.eval()
    clip_model.to(device)
    
    # Create model after determining the correct prefix_size
    if args.decap_weights is None:
        model = DeCap(prefix_size)
    else:
        model = get_decap_model(device, args.decap_weights, prefix_size)
    
    if args.talk2dino_weights is not None:
        # loading Talk2DINO
        print(f"Loading Talk2DINO weights from {args.talk2dino_weights}")
        talk2dino = ProjectionLayer.from_config(args.talk2dino_config)
        talk2dino.load_state_dict(torch.load(args.talk2dino_weights, device))
        talk2dino.to(device)
        talk2dino.eval()

    else:
        talk2dino = None
        
        
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
    model.to(device)

    if args.not_distributed == False:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    if not args.pre_extract_features:
        print("Features pre-extraction de-activated")
        if args.mix_captions:
            print("Using mix captions")
            dataset = ClipCocoDatasetMix(args.dataset, use_precomputed_feats=args.use_dino_feats, tokenizer=tokenizer)
        else:
            dataset = ClipCocoDataset(args.dataset, use_dino_feats=args.use_dino_feats, tokenizer=tokenizer)
    else:
        if args.mix_captions:
            print("Using mix captions")
            dataset = ClipCocoDatasetMix(args.dataset, clip_model=clip_model, talk2dino=talk2dino, tokenizer=tokenizer)
        else:
            dataset = ClipCocoDataset(args.dataset, clip_model=clip_model, talk2dino=talk2dino, tokenizer=tokenizer)
        
    
    optimizer = AdamW(model.parameters(),lr=lr)
    
    print(f"Going to construct DataLoader with {len(dataset)} samples")
    if args.not_distributed == False:
        sampler = DistributedSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    else:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print("DataLoader constructed")
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    
    
    for epoch in range(epochs):
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        
        loss_token_save,ac_save= 0,0
        sys.stdout.flush()
        if args.is_master:
            print(f">>> Training epoch {epoch}")
            progress = tqdm(total=int(len(train_dataloader)/10), desc=output_prefix, dynamic_ncols=True)
        
        if args.not_distributed == False:
            dist.barrier()
        
        for idx,(clip_tokens, pipeline_input) in enumerate(train_dataloader):
            

            clip_tokens, pipeline_input = clip_tokens.to(device), pipeline_input.to(device)
            
            with torch.no_grad():
                if not args.pre_extract_features and not args.use_dino_feats:
                    if args.use_regionclip:
                        # RegionCLIP text encoding
                        feature_text = clip_model.encode_text(pipeline_input)
                    elif args.denseclip_config is not None:
                        # DenseClip text encoding
                        feature_text = clip_model.encode_text(pipeline_input)
                    else:
                        # Standard CLIP or OpenCLIP text encoding
                        feature_text = clip_model.encode_text(pipeline_input)
                    
                    if args.use_dinotxt:
                        feature_text = feature_text[:, 1024:] # patch-aligned text embedding

                    if args.talk2dino_weights is not None:
                        feature_text = talk2dino.project_clip_txt(feature_text)
                else:
                    feature_text = pipeline_input
                    if args.im_proj:
                        feature_text = im_proj.project(feature_text, normalize=True)
                
                feature_text /= feature_text.norm(dim=-1, keepdim=True)
                
                if args.gaussian_noise != 0:
                    feature_text += args.gaussian_noise * torch.randn(feature_text.shape).to(device)
                    feature_text /= feature_text.norm(dim=-1, keepdim=True)
                    

            outputs = model(feature_text.float(),clip_tokens)
            logits = outputs
            
            logits = logits.logits

            logits = logits[:,: -1]
            clip_tokens = clip_tokens.flatten()
            logits = logits.reshape(-1, logits.shape[-1])
            
            loss_token = loss_ce(logits, clip_tokens)
            ac=((logits.argmax(1)==clip_tokens)*(clip_tokens>0)).sum()/(clip_tokens>0).sum()
            optimizer.zero_grad()
            loss_all = loss_token
            loss_all.backward()
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss_token.item()
            epoch_acc += ac.item()
            num_batches += 1
            
            if args.is_master:
                
                if(idx+1) %10 == 0:
                    progress.set_postfix({"loss_token": loss_token_save/10.0,"acc_token":ac_save/10.0})
                    progress.update()
                    loss_token_save,ac_save= 0,0
                else:
                    loss_token_save += loss_token.item()
                    ac_save += ac.item()

        if args.is_master:
            log_dir = os.path.join('./log', f"{args.dataset}.txt")#'./log/'+args.dataset+'.txt'
            with open(log_dir,'w') as f:
                f.writelines('epoch ' +str(epoch) +': '+ progress.postfix+'\r\n')
            progress.close()
            if epoch % args.save_every == 0 or epoch == epochs - 1:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}-{epoch:03d}.pt"),
                )
        
        # after the epoch, we need to synchronize the loss and accuracy across all processes
        loss_tensor = torch.tensor(epoch_loss, device=device)
        acc_tensor = torch.tensor(epoch_acc, device=device)
        count_tensor = torch.tensor(num_batches, device=device)

        if args.not_distributed == False:
            # sum on all processes
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(acc_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)

        # compute global mean
        avg_loss = loss_tensor.item() / count_tensor.item()
        avg_acc = acc_tensor.item() / count_tensor.item()

        if args.is_master:
            epoch_loss_current = {'epoch': epoch, 'loss': avg_loss, 'accuracy': avg_acc}
            #epoch_losses.append(epoch_loss_current)
            print(f"Epoch {epoch} loss: {avg_loss}, accuracy: {avg_acc}")

            loss_csv_path = os.path.join(output_dir, f"{output_prefix}_epoch_losses.csv")
            with open(loss_csv_path, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'loss', 'accuracy'])
                # Write the header only if the file is empty
                if os.stat(loss_csv_path).st_size == 0:
                    writer.writeheader()
                writer.writerow(epoch_loss_current)
    return model

# DeCap CLIP B16 karpathy train split:
#python decapTraining.py --out_dir weights_clip_b16_karpathy --not-distributed 1 --local-rank 0 --dataset coco_train_karpathy.json --prefix coco_karpathy
# DECAP with proj -> ma in realtà non serve.
#python decapTraining.py --out_dir weights_clip_b16_proj_karpathy --not-distributed 1 --local-rank 0 --dataset coco_train_karpathy.json --prefix coco_karpathy --im_proj

# Patchioner DINOv2 karpathy train split with proj:
#python decapTraining.py --out_dir weights_dino_b14_proj_karpathy --not-distributed 1 --local-rank 1 --dataset coco_train_karpathy.json --prefix coco_karpathy --talk2dino_weights weights_talk2dino/vitb_mlp_infonce.pth --talk2dino_config configs_talk2dino/vitb_mlp_infonce.yaml --pre_extract_features --im_proj
# Patchioner DINOv2 karpathy train split
#python decapTraining.py --out_dir weights_dino_b14_karpathy --not-distributed 1 --local-rank 1 --dataset coco_train_karpathy.json --prefix coco_karpathy --talk2dino_weights weights_talk2dino/vitb_mlp_infonce.pth --talk2dino_config configs_talk2dino/vitb_mlp_infonce.yaml
#python decapTraining.py --out_dir weights_dino_b14_karpathy --not-distributed 1 --local-rank 1 --dataset coco_train_karpathy.json --prefix coco_karpathy --talk2dino_weights weights_talk2dino/vitb_mlp_infonce.pth --talk2dino_config configs_talk2dino/vitb_mlp_infonce.yaml --use_dino_feats --pre_extract_features

# DeCap CLIP B32 karpathy train split:
#python decapTraining.py --out_dir weights_clip_b32_karpathy --not-distributed 1 --local-rank 0 --dataset coco_train_karpathy.json --prefix coco_karpathy --clip_model ViT-B/32

# DeCap with RegionCLIP text encoder:
#python decoderTraining.py --out_dir weights_regionclip_karpathy --not-distributed 1 --local-rank 0 --dataset coco_train_karpathy.json --prefix coco_karpathy --use-regionclip

# DeCap with DenseClip text encoder:
#python decoderTraining.py --out_dir weights_denseclip_segmentation_vitb16_karpathy --not-distributed 1 --local-rank 0 --dataset coco_train_karpathy.json --prefix coco_karpathy --denseclip-config denseclip_segmentation_vitb16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decap_weights', type=str, default=None, help="If setted the Decap initialization is not random")
    parser.add_argument('--clip_model', type=str, default='ViT-B/16', help="CLIP configuration")
    parser.add_argument('--use_dinotxt', default=None, action='store_true', help="CLIP configuration")
    parser.add_argument('--gaussian_noise', type=float, default=0, help="Standard deviation of the Gaussian noise to apply to the text input")
    parser.add_argument('--out_dir', default='./coco_model')
    parser.add_argument('--prefix', default='./coco_prefix', help='prefix for saved filenames')
    parser.add_argument('--dataset', default='coco', help='coco or cc3m or bookcorpus')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--prefix_length', type=int, default=1)
    parser.add_argument('--prefix_length_clip', type=int, default=1)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--talk2dino_weights', type=str, default=None, help="Talk2DINO weights. If None, the training will be performed without Talk2DINO.")
    parser.add_argument('--talk2dino_config', type=str, default=None, help="Talk2DINO configs. Valid only if the weights are setted.")
    parser.add_argument('--use_dino_feats', action="store_true", default=False, help="If setted, we use the pre-extracted features of DINOv2")
    parser.add_argument('--im_proj', action="store_true", default=False, help="If setted, we use the projection on the input features")
    parser.add_argument('--pre_extract_features', action="store_true", default=False, help="If setted, the features will be extracted during the dataloading")
    parser.add_argument('--only_prefix', dest='only_prefix', action='store_true')
    parser.add_argument('--mapping_type', type=str, default='mlp', help='mlp/transformer')
    parser.add_argument('--num_layers', type=int, default=8)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--normalize_prefix', dest='normalize_prefix', action='store_true')
    parser.add_argument('--local-rank', type=int, default=-1, metavar='N', help='Local process rank.') 
    parser.add_argument('--not-distributed', type=int, default=False, metavar='N', help='Not Distributed toggle.')
    parser.add_argument('--use-open-clip', action='store_true', default=False, help='Use OpenCLIP instead of CLIP')
    parser.add_argument('--mix-captions', action='store_true', default=False, help='Mix captions from the same image')
    parser.add_argument('--use-regionclip', action='store_true', default=False, help='Use RegionCLIP for text encoding')
    parser.add_argument('--regionclip-checkpoint', type=str, default='/raid/datasets/models_weights/regionclip/regionclip_pretrained-cc_rn50x4.pth', help='Path to RegionCLIP checkpoint file')
    parser.add_argument('--regionclip-config', type=str, default='pretrain/RegionCLIP_RN50x4.yaml', help='Path to RegionCLIP config file or config name')
    parser.add_argument('--denseclip-config', type=str, default=None, help='Path to DenseClip config file or config name')
    args = parser.parse_args()
    
    # Validate RegionCLIP arguments
    if args.use_regionclip and args.regionclip_checkpoint is None:
        parser.error("--regionclip-checkpoint is required when using --use-regionclip")
    
    if args.use_regionclip and args.use_open_clip:
        parser.error("Cannot use both --use-regionclip and --use-open-clip at the same time")
    
    # Validate DenseClip arguments
    if args.denseclip_config is not None and args.use_regionclip:
        parser.error("Cannot use both --denseclip-config and --use-regionclip at the same time")
    
    if args.denseclip_config is not None and args.use_open_clip:
        parser.error("Cannot use both --denseclip-config and --use-open-clip at the same time")
    

    train_decoder(args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()