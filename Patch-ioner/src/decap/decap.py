import os
from torch import nn
import numpy as np
import torch
import torch.nn.functional as nnf
import sys
from typing import Tuple, List, Union, Optional
from tqdm import tqdm, trange
import pickle
import PIL.Image as Image
import json
import random
import sys
import PIL
import random

from torch.utils.data import Dataset, DataLoader
from enum import Enum
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup

# Import HuggingFace Hub utilities for model loading
from ..hf_utils import load_model_with_hf_fallback

from tqdm import tqdm
import os
import pickle
import sys
import argparse
import json
from typing import Tuple, Optional, Union

import os
from dotenv import load_dotenv

load_dotenv()


DECAP_DECODER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "decoder_config.pkl")
DECAP_COCO_WEIGHTS_PATH = None#'../../thesis-data/decap/coco_model/coco_prefix-009.pt'
        
class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'

    
class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)
        

class DeCap(nn.Module):

    def __init__(self,prefix_size: int = 512):
        super(DeCap, self).__init__()
        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open(DECAP_DECODER_CONFIG_PATH,'rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size,self.embedding_size))
        
    def forward(self, clip_features,tokens):
        embedding_text = self.decoder.transformer.wte(tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out

from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_Tokenizer = _Tokenizer()

def Decoding(model,clip_features):
    model.eval()
    embedding_cat = model.clip_project(clip_features).reshape(1,1,-1)
    entry_length = 30
    temperature = 1
    tokens = None
    for i in range(entry_length):
        # print(location_token.shape)
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits_max = logits.max()
        logits = torch.nn.functional.softmax(logits, -1)
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = model.decoder.transformer.wte(next_token)

        if tokens is None:
            tokens = next_token

        else:
            tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item()==49407:
            break
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
    try:
        output_list = list(tokens.squeeze().cpu().numpy())
        output = _Tokenizer.decode(output_list)
    except:
        output = 'None'
    return output

def decoding_batched(model, clip_features, compute_scores : bool = False, decoding_method : callable = None, return_start_end_tokens : bool = False):
    """
    Returns the generated sequences for a batch of clip features.
    - if compute_scores is True, also returns the scores of the generated sequences.
    - returns a list of strings if compute_scores is False, otherwise a tuple of a list of strings and a list of floats.
    """

    model.eval()
    embedding_cat = model.clip_project(clip_features).view(clip_features.shape[0], 1, -1)
    entry_length = 30
    temperature = 1
    tokens = None
    sequence_log_probs = None

    for i in range(entry_length):
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits[:, -1, :]
        logits = logits / (temperature if temperature > 0 else 1.0)

        probs = torch.nn.functional.softmax(logits, -1)

        if compute_scores:
            log_probs = torch.log(probs)  # Convert to log-probabilities

        next_token = torch.argmax(probs, -1).unsqueeze(1)
        next_token_embed = model.decoder.transformer.wte(next_token)

        if tokens is None:
            tokens = next_token
            if compute_scores:
                sequence_log_probs = log_probs.gather(1, next_token)  # Store log-prob of first token
        else:
            tokens = torch.cat((tokens, next_token), dim=1)
            if compute_scores:
                token_log_probs = log_probs.gather(1, next_token)  # Get log-prob of chosen token
                sequence_log_probs = torch.cat((sequence_log_probs, token_log_probs), dim=1)  # Append
        
        # Append new token embedding to input
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)

    if compute_scores:
        # Compute total sequence scores
        sequence_scores = sequence_log_probs.sum(dim=-1)  # Sum log-probs over sequence
        final_scores = torch.exp(sequence_scores)  # Convert log-sum-prob to probability-like score
    
    try:
        outputs = []
        for tokens_elem in tokens:
            output_list = list(tokens_elem.squeeze().cpu().numpy())
            if decoding_method is not None:
                output = decoding_method(output_list)
            else:
                output = _Tokenizer.decode(output_list)
            


            output = output.split('<|endoftext|>')[0]
            if not return_start_end_tokens:
                output = output.replace('<|startoftext|>', '')
            else:
                output += '<|endoftext|>'

            outputs.append(output)
    except:
        outputs = None
    
    return (outputs, final_scores.cpu().numpy().tolist()) if compute_scores else outputs


decap_model = None

def get_decap_model(device, weights_path = DECAP_COCO_WEIGHTS_PATH, prefix_size=512, hf_repo_id=None):
    """
    Load a DeCap model from local checkpoint or HuggingFace Hub.
    
    Args:
        device: Device to load the model on
        weights_path: Path to local checkpoint file
        prefix_size: Size of the prefix for the model
        hf_repo_id: HuggingFace repository ID for fallback download
        
    Returns:
        Loaded DeCap model
    """
    #global decap_model
    #if decap_model is not None:
    #    return decap_model
    decap_model = DeCap(prefix_size)
    
    # Try to load with HuggingFace Hub fallback
    try:
        
        state_dict = load_model_with_hf_fallback(
            local_path=weights_path,
            hf_repo_id=hf_repo_id,
            map_location=torch.device('cpu')
        )
        decap_model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"Warning: Failed to load with HF fallback: {e}")
        # Fallback to original loading method
        decap_model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')), strict=False)
    
    decap_model = decap_model.to(device)
    decap_model = decap_model.eval()
    return decap_model
