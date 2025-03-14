import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from im2txtprojection.im2txtprojection import Im2TxtProjector, ProjectionType
from transformers import AdamW, get_linear_schedule_with_warmup
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

from src.dataset import ClipCocoDataset
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
    args.is_master = args.local_rank == 0

    # set the device
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda:'+str(args.local_rank))
    if args.not_distributed == False:
        dist.init_process_group(backend='nccl', init_method='env://')
    SEED=42
    torch.cuda.manual_seed_all(SEED)
    
    if args.talk2dino_weights is not None or args.use_dino_feats:
        prefix_size = 768
    else:
        prefix_size = 512
    
    if args.decap_weights is None:
        model = DeCap(prefix_size)
    else:
        model = get_decap_model(device, args.decap_weights, prefix_size)
    
    if args.im_proj:
        memory_bank_path = os.path.abspath(args.dataset)
        print(f"Using Im2TxtProjector with {memory_bank_path = }")
        im_proj = Im2TxtProjector(
            type=memory_bank_path,
            use_talk2dino=True,
            linear_talk2dino=False,
            memory_bank_name='coco_karpathy',
            device_str=device)

    clip_model, preprocess = clip.load(args.clip_model, device=device, jit=False)
    clip_model.eval()
    
    if args.talk2dino_weights is not None:
        # loading Talk2DINO
        talk2dino = ProjectionLayer.from_config(args.talk2dino_config)
        talk2dino.load_state_dict(torch.load(args.talk2dino_weights, device))
        talk2dino.to(device)
        talk2dino.eval()

        
    loss_ce = torch.nn.CrossEntropyLoss(ignore_index=0,label_smoothing=0.1)
    model.to(device)

    if not args.not_distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )
    
    if not args.pre_extract_features:
        print("Features pre-extraction de-activated")
        dataset = ClipCocoDataset(args.dataset, use_dino_feats=args.use_dino_feats)
    else:
        dataset = ClipCocoDataset(args.dataset, clip_model=clip_model, talk2dino=talk2dino)
        
    
    optimizer = AdamW(model.parameters(),lr=lr)
    
    print(f"Going to construct DataLoader with {len(dataset)} samples")
    if not args.not_distributed:
        sampler = DistributedSampler(dataset)
        train_dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size, drop_last=True)
    else:
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )
    
    
    for epoch in range(epochs):
        loss_token_save,ac_save= 0,0
        sys.stdout.flush()
        if args.is_master:
            print(f">>> Training epoch {epoch}")
            progress = tqdm(total=int(len(train_dataloader)/10), desc=output_prefix)
        
        if not args.not_distributed:
            dist.barrier()
        
        for idx,(clip_tokens, pipeline_input) in enumerate(train_dataloader):
            clip_tokens, pipeline_input = clip_tokens.to(device), pipeline_input.to(device)
            
            with torch.no_grad():
                if not args.pre_extract_features and not args.use_dino_feats:
                    feature_text = clip_model.encode_text(pipeline_input)
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
            if args.is_master:
                
                if(idx+1) %10 ==0:
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
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--decap_weights', type=str, default=None, help="If setted the Decap initialization is not random")
    parser.add_argument('--clip_model', type=str, default='ViT-B/16', help="CLIP configuration")
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
    args = parser.parse_args()
    

    train_decoder(args, output_dir=args.out_dir, output_prefix=args.prefix)


if __name__ == '__main__':
    main()