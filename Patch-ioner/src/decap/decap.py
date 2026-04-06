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
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup, AutoConfig, AutoModelForCausalLM
# AdamW optimizer
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
DEFAULT_QWEN_DECODER_ID = "Qwen/Qwen3-0.6B"
DEFAULT_LLAMA_DECODER_ID = "meta-llama/Llama-3.2-1B"
DEFAULT_OPENELM_DECODER_ID = "apple/OpenELM-270M"
DEFAULT_GEMMA3_DECODER_ID = "google/gemma-3-270m"

CLIP_EOS_TOKEN_ID = 49407
        
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

    def __init__(
        self,
        prefix_size: int = 512,
        decoder_family: str = "gpt2",
        decoder_model_id: Optional[str] = None,
        decoder_config_path: Optional[str] = None,
        decoder_max_position_embeddings: Optional[int] = None,
        decoder_trust_remote_code: bool = True,
        decoder_torch_dtype: Optional[torch.dtype] = None,
        decoder_random_init: bool = False,
    ):
        super(DeCap, self).__init__()
        decoder_family = decoder_family.lower()

        if decoder_family == 'gpt2':
            with open(decoder_config_path or DECAP_DECODER_CONFIG_PATH,'rb') as f:
                config = pickle.load(f)
            if not hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
            if not hasattr(config, "_output_attentions"):
                if hasattr(config, "output_attentions"):
                    val = config.output_attentions
                else:
                    val = False
                    print(f"DEBUG: Setting config._output_attentions to False by default")
                config._output_attentions = val
            self.decoder = GPT2LMHeadModel(config)
        elif decoder_family in ['qwen3', 'openelm', 'gemma3', 'llama']:
            defaults = {
                'qwen3': DEFAULT_QWEN_DECODER_ID,
                'openelm': DEFAULT_OPENELM_DECODER_ID,
                'gemma3': DEFAULT_GEMMA3_DECODER_ID,
                'llama': DEFAULT_LLAMA_DECODER_ID
            }
            model_id = decoder_model_id or defaults.get(decoder_family)
            config_source = decoder_config_path or model_id
            config = AutoConfig.from_pretrained(
                config_source,
                trust_remote_code=decoder_trust_remote_code,
            )
            if decoder_max_position_embeddings is not None:
                config.max_position_embeddings = decoder_max_position_embeddings
            if decoder_torch_dtype is not None:
                config.dtype = decoder_torch_dtype

            if decoder_random_init:
                self.decoder = AutoModelForCausalLM.from_config(
                    config,
                    trust_remote_code=decoder_trust_remote_code,
                    dtype=decoder_torch_dtype
                )
            else:
                from_pretrained_kwargs = dict(trust_remote_code=decoder_trust_remote_code)
                if decoder_torch_dtype is not None:
                    from_pretrained_kwargs['dtype'] = decoder_torch_dtype
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    config=config,
                    **from_pretrained_kwargs,
                )
        else:
            raise ValueError(f"Unsupported decoder family: {decoder_family}")

        input_embeddings = self.decoder.get_input_embeddings()
        self.embedding_size = input_embeddings.weight.shape[1]
        self.clip_project = MLP((prefix_size,self.embedding_size))
        self.decoder_family = decoder_family
        
    def forward(self, clip_features,tokens):
        embedding_layer = self.decoder.get_input_embeddings()
        embedding_text = embedding_layer(tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out

from ..clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_Tokenizer = _Tokenizer()


def _decode_token_list(tokenizer, token_list, skip_special_tokens: bool = True):
    if tokenizer is None:
        return _Tokenizer.decode(token_list)
    if hasattr(tokenizer, 'decode'):
        try:
            return tokenizer.decode(token_list, skip_special_tokens=skip_special_tokens)
        except TypeError:
            return tokenizer.decode(token_list)
    if callable(tokenizer):
        return tokenizer(token_list)
    return _Tokenizer.decode(token_list)


def _get_default_eos_token_id(tokenizer, decoder_family: Optional[str]):
    if tokenizer is not None and hasattr(tokenizer, 'eos_token_id') and tokenizer.eos_token_id is not None:
        return tokenizer.eos_token_id
    if decoder_family == 'gpt2':
        return 50256
    if decoder_family == 'qwen3':
        # Qwen3 defaults (151645) if tokenizer not provided
        return 151645
    if decoder_family == 'openelm':
        return 2 # LLama / OpenELM EOS token ID
    if decoder_family == 'gemma3':
        return 1  # Gemma3 EOS token ID
    if decoder_family == 'llama':
        return 2  # LLaMA EOS token ID
    return CLIP_EOS_TOKEN_ID

full_stop_lookup_table = {}

def _get_default_full_stop_token_id(tokenizer, decoder_family: Optional[str]):
    if decoder_family == 'gpt2':
        return 13  # GPT-2 full stop token ID
    if decoder_family == 'qwen3':
        return 13  # Qwen3 full stop token ID
    if decoder_family in ['openelm', 'llama']:
        return 13  # LLama / OpenELM full stop token ID
    if decoder_family == 'gemma3':
        return 236761
    
    global full_stop_lookup_table
    if tokenizer in full_stop_lookup_table:
        return full_stop_lookup_table[tokenizer]
    if tokenizer is not None: 
        if hasattr(tokenizer, 'forward'):
            full_stop_token_id = tokenizer(["."])
        
            if not isinstance(full_stop_token_id, int) and not isinstance(full_stop_token_id, list) and not isinstance(full_stop_token_id, torch.Tensor):
                full_stop_lookup_table[tokenizer] = full_stop_token_id['input_ids'][0][-1]
                return full_stop_token_id['input_ids'][0][-1]
            else:
                full_stop_lookup_table[tokenizer] = full_stop_token_id[0][-1]
                return full_stop_token_id[0][-1]
        elif hasattr(tokenizer, 'encode'):
            full_stop_token_id = tokenizer.encode(".")[0]
            if isinstance(full_stop_token_id, list) or isinstance(full_stop_token_id, torch.Tensor):
                full_stop_token_id = full_stop_token_id[-1]
            full_stop_lookup_table[tokenizer] = full_stop_token_id
            return full_stop_token_id
    

def Decoding(model,clip_features, tokenizer=None, decoder_family: Optional[str] = None, eos_token_id: Optional[int] = None, return_start_end_tokens: bool = False):
    model.eval()
    embedding_cat = model.clip_project(clip_features).reshape(1,1,-1)
    embedding_layer = model.decoder.get_input_embeddings()
    entry_length = 30
    temperature = 1
    tokens = None
    decoder_family = decoder_family.lower() if decoder_family is not None else None
    effective_eos = eos_token_id if eos_token_id is not None else _get_default_eos_token_id(tokenizer, decoder_family)
    for i in range(entry_length):
        # print(location_token.shape)
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits
        logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
        logits_max = logits.max()
        logits = torch.nn.functional.softmax(logits, -1)
        next_token = torch.argmax(logits, -1).unsqueeze(0)
        next_token_embed = embedding_layer(next_token)

        if tokens is None:
            tokens = next_token

        else:
            tokens = torch.cat((tokens, next_token), dim=1)
        if next_token.item()==effective_eos:
            break
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)
    try:
        output_list = list(tokens.squeeze().cpu().numpy())
        output = _decode_token_list(tokenizer, output_list)
        if tokenizer is None:
            output = output.split('<|endoftext|>')[0]
            if return_start_end_tokens:
                output += '<|endoftext|>'
            else:
                output = output.replace('<|startoftext|>', '')
    except:
        output = 'None'
    return output

def decoding_batched(model, clip_features, compute_scores : bool = False, decoding_method : callable = None, return_start_end_tokens : bool = False, tokenizer=None, decoder_family: Optional[str] = None, eos_token_id: Optional[int] = None, full_stop_token_id = None):
    """
    Returns the generated sequences for a batch of clip features.
    - if compute_scores is True, also returns the scores of the generated sequences.
    - returns a list of strings if compute_scores is False, otherwise a tuple of a list of strings and a list of floats.
    """

    model.eval()
    model_dtype = next(model.parameters()).dtype
    embedding_cat = model.clip_project(clip_features).view(clip_features.shape[0], 1, -1).to(model_dtype)
    embedding_layer = model.decoder.get_input_embeddings()
    entry_length = 30
    temperature = 1
    tokens = None
    sequence_log_probs = None

    decoder_family = decoder_family.lower() if decoder_family is not None else None
    effective_eos = eos_token_id if eos_token_id is not None else _get_default_eos_token_id(tokenizer, decoder_family)

    if full_stop_token_id is None:
        full_stop_token_id = _get_default_full_stop_token_id(tokenizer, decoder_family)
        
    for i in range(entry_length):
        outputs = model.decoder(inputs_embeds=embedding_cat)

        logits = outputs.logits[:, -1, :]

        if i > 0:
            # Force finished sequences to stay finished
            logits[finished] = float('-inf')
            logits[finished, effective_eos] = 0

        #if i == 0:
        #    # For the first token, prevent EOS prediction if it is the highest logit
        #    logits[:, effective_eos] = float('-inf')
        #    logits[:, full_stop_token_id] = float('-inf')

        logits = logits / (temperature if temperature > 0 else 1.0)

        probs = torch.nn.functional.softmax(logits, -1)

        if compute_scores:
            log_probs = torch.log(probs)  # Convert to log-probabilities

        next_token = torch.argmax(probs, -1).unsqueeze(1)
        next_token_embed = embedding_layer(next_token)

        if tokens is None:
            tokens = next_token
            if compute_scores:
                sequence_log_probs = log_probs.gather(1, next_token)  # Store log-prob of first token
        else:
            tokens = torch.cat((tokens, next_token), dim=1)
            if compute_scores:
                token_log_probs = log_probs.gather(1, next_token)  # Get log-prob of chosen token
                sequence_log_probs = torch.cat((sequence_log_probs, token_log_probs), dim=1)  # Append
        
        # detect stopping for each sample
        stop_mask = (next_token.squeeze(1) == effective_eos) | (
            next_token.squeeze(1) == full_stop_token_id
        )

        # init finished mask if first iteration
        if i == 0:
            finished = stop_mask.clone()
        else:
            finished = finished | stop_mask

        # force EOS token for all finished sequences
        next_token[finished] = effective_eos
        next_token_embed = embedding_layer(next_token)

        # append embeddings
        embedding_cat = torch.cat((embedding_cat, next_token_embed), dim=1)

        # if all sequences are done, exit early
        if finished.all():
            break

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
                output = _decode_token_list(tokenizer, output_list)

            if '<|startoftext|>' in output or '<|endoftext|>' in output: #if tokenizer is None:
                output = output.split('<|endoftext|>')[0]
                if not return_start_end_tokens:
                    output = output.replace('<|startoftext|>', '')
                else:
                    output += '<|endoftext|>'

            outputs.append(output)
    except Exception as e:
        raise e
        outputs = None
    
    return (outputs, final_scores.cpu().float().numpy().tolist()) if compute_scores else outputs

import copy


@torch.no_grad()
def _decoding_diverse_autoregressive(
    model,
    full_inputs_embeds,
    entry_length: int,
    num_candidates: int,
    eos_token_id: Optional[int],
    full_stop_token_id: Optional[int],
    top_p: float,
    temperature: float,
    tokenizer,
    decoding_method: Optional[callable],
    return_start_end_tokens: bool,
    compute_scores: bool,
    prevent_eos_at_start: bool,
):
    device = full_inputs_embeds.device
    embedding_layer = model.decoder.get_input_embeddings()

    batch_size = full_inputs_embeds.shape[0]
    expanded_embeds = full_inputs_embeds.repeat_interleave(num_candidates, dim=0)

    finished = torch.zeros(batch_size * num_candidates, dtype=torch.bool, device=device)
    tokens_steps = []
    log_prob_steps = []

    safe_temperature = max(float(temperature), 1e-5)
    do_top_p = 0.0 < float(top_p) < 1.0

    for step in range(entry_length):
        logits = model.decoder(inputs_embeds=expanded_embeds).logits[:, -1, :].float()

        if step == 0 and prevent_eos_at_start:
            if eos_token_id is not None and 0 <= eos_token_id < logits.shape[-1]:
                logits[:, eos_token_id] = float("-inf")
            if full_stop_token_id is not None and 0 <= full_stop_token_id < logits.shape[-1]:
                logits[:, full_stop_token_id] = float("-inf")

        if finished.any():
            logits[finished] = float("-inf")
            if eos_token_id is not None and 0 <= eos_token_id < logits.shape[-1]:
                logits[finished, eos_token_id] = 0.0

        logits = logits / safe_temperature
        probs = torch.softmax(logits, dim=-1)

        if do_top_p:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            to_remove = cumulative_probs > float(top_p)
            to_remove[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(to_remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-12)

            sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_idx.gather(dim=-1, index=sampled_sorted)
            chosen_prob = sorted_probs.gather(dim=-1, index=sampled_sorted).squeeze(-1)
        else:
            next_token = torch.multinomial(probs, num_samples=1)
            chosen_prob = probs.gather(dim=-1, index=next_token).squeeze(-1)

        if finished.any() and eos_token_id is not None and 0 <= eos_token_id < probs.shape[-1]:
            next_token[finished] = eos_token_id
            chosen_prob = torch.where(finished, torch.ones_like(chosen_prob), chosen_prob)

        tokens_steps.append(next_token)
        log_prob_steps.append(torch.log(chosen_prob.clamp(min=1e-12)))

        next_token_embed = embedding_layer(next_token)
        expanded_embeds = torch.cat((expanded_embeds, next_token_embed), dim=1)

        stop_mask = torch.zeros_like(finished)
        if eos_token_id is not None and 0 <= eos_token_id < probs.shape[-1]:
            stop_mask = stop_mask | (next_token.squeeze(1) == eos_token_id)
        if full_stop_token_id is not None and 0 <= full_stop_token_id < probs.shape[-1]:
            stop_mask = stop_mask | (next_token.squeeze(1) == full_stop_token_id)

        finished = finished | stop_mask
        if finished.all():
            break

    if len(tokens_steps) == 0:
        sequences = torch.empty((batch_size * num_candidates, 0), dtype=torch.long, device=device)
        token_log_probs = torch.empty((batch_size * num_candidates, 0), dtype=torch.float32, device=device)
    else:
        sequences = torch.cat(tokens_steps, dim=1)
        token_log_probs = torch.stack(log_prob_steps, dim=1)

    captions = []
    for seq in sequences:
        output_list = seq.tolist()
        if decoding_method is not None:
            output = decoding_method(output_list)
        else:
            output = _decode_token_list(tokenizer, output_list)

        for stop in ['<|endoftext|>', '<|im_end|>', '<|end|>', '<eos>', '</s>']:
            output = output.split(stop)[0]
        if not return_start_end_tokens:
            for start in ['<|startoftext|>', '<|im_start|>', '<bos>', '<s>']:
                output = output.replace(start, '')
        captions.append(output.strip())

    if not compute_scores:
        return captions

    if sequences.shape[1] == 0:
        perplexities = [float('inf')] * (batch_size * num_candidates)
        return captions, perplexities

    if eos_token_id is not None:
        pos = torch.arange(sequences.shape[1], device=device).unsqueeze(0)
        eos_seen = sequences.eq(eos_token_id)
        eos_at = torch.where(
            eos_seen.any(dim=1),
            eos_seen.float().argmax(dim=1),
            torch.tensor(sequences.shape[1] - 1, device=device)
        )
        valid_mask = pos <= eos_at.unsqueeze(1)
    else:
        valid_mask = torch.ones_like(token_log_probs, dtype=torch.bool)

    mean_nll = -(token_log_probs * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
    perplexities = torch.exp(mean_nll).detach().cpu().numpy().tolist()
    return captions, perplexities

@torch.no_grad()
def decoding_diverse_batched(
    model,
    clip_features,
    compute_scores: bool = False,
    decoding_method: callable = None,
    return_start_end_tokens: bool = False,
    num_candidates: int = 32,
    diverse_mode: str = "strict",
    entry_length: int = 30,
    num_beam_groups: int = 8,
    diversity_penalty: float = 0.5,
    top_p: float = 0.9,
    temperature: float = 1.0,
    prevent_eos_at_start: bool = True,
    min_new_tokens: int = 1,
    tokenizer=None,
    decoder_family: Optional[str] = None,
):
    model.eval()
    device = clip_features.device
    model_dtype = next(model.decoder.parameters()).dtype
    decoder_family = (decoder_family or getattr(model, "decoder_family", "gpt2")).lower()

    # 1. PREPARE PREFIX
    prefix_embeds = model.clip_project(clip_features).view(clip_features.shape[0], 1, -1)
    prefix_embeds = prefix_embeds.to(model_dtype)
    batch_size = prefix_embeds.shape[0]

    # 2. RESOLVE TOKENS
    eos_token_id = _get_default_eos_token_id(tokenizer, decoder_family)
    bos_token_id = getattr(model.decoder.config, "bos_token_id", None)
    full_stop_token_id = _get_default_full_stop_token_id(tokenizer, decoder_family)

    ## 3. ARCHITECTURE BRANCHING (The Fix)
    ## GPT-2 does NOT want a BOS. Gemma/Qwen/Llama DO.
    #if decoder_family != 'gpt2' and bos_token_id is not None:
    #    embedding_layer = model.decoder.get_input_embeddings()
    #    bos_tokens = torch.full((batch_size, 1), bos_token_id, device=device, dtype=torch.long)
    #    bos_embeds = embedding_layer(bos_tokens)
    #    full_inputs_embeds = torch.cat([bos_embeds, prefix_embeds], dim=1)
    #else:
    #    # For GPT-2 or models without BOS
    #    full_inputs_embeds = prefix_embeds

    full_inputs_embeds = prefix_embeds

    # For non-GPT2 decoder families, avoid generate(inputs_embeds=...) because it
    # can collapse to EOS/special-token outputs. Use explicit autoregressive sampling.
    if decoder_family in ['qwen3', 'gemma3', 'llama', 'openelm']:
        return _decoding_diverse_autoregressive(
            model=model,
            full_inputs_embeds=full_inputs_embeds,
            entry_length=entry_length,
            num_candidates=num_candidates,
            eos_token_id=eos_token_id,
            full_stop_token_id=full_stop_token_id,
            top_p=top_p,
            temperature=temperature,
            tokenizer=tokenizer,
            decoding_method=decoding_method,
            return_start_end_tokens=return_start_end_tokens,
            compute_scores=compute_scores,
            prevent_eos_at_start=prevent_eos_at_start,
        )

    # 4. ATTENTION MASK
    attention_mask = torch.ones(batch_size, full_inputs_embeds.shape[1], device=device, dtype=torch.long)

    # 5. GENERATION CONFIG
    # Diverse Beam Search fallback logic
    actual_beam_groups = num_beam_groups
    if diverse_mode == "strict":
        actual_beam_groups = max(1, min(num_beam_groups, num_candidates))
        while num_candidates % actual_beam_groups != 0 and actual_beam_groups > 1:
            actual_beam_groups -= 1
    else:
        actual_beam_groups = 1

    generation_kwargs = {
        "inputs_embeds": full_inputs_embeds,
        "attention_mask": attention_mask,
        "max_new_tokens": entry_length,
        "num_beams": num_candidates,
        "num_return_sequences": num_candidates,
        "eos_token_id": eos_token_id,
        "pad_token_id": eos_token_id,
        "return_dict_in_generate": True,
        "use_cache": True,
    }

    # Prevent immediate EOS collapse, which can produce empty candidates.
    if prevent_eos_at_start and min_new_tokens > 0:
        generation_kwargs["min_new_tokens"] = min(min_new_tokens, entry_length)

    if diverse_mode == "strict" and actual_beam_groups > 1:
        generation_kwargs.update({
            "do_sample": False,
            "num_beam_groups": actual_beam_groups,
            "diversity_penalty": diversity_penalty,
        })
    else:
        generation_kwargs.update({
            "do_sample": True,
            "top_p": top_p,
            "temperature": temperature,
        })

    # Deep clean config
    if hasattr(model.decoder, "generation_config"):
        gen_config = copy.deepcopy(model.decoder.generation_config)
        gen_config.custom_generate = False
        generation_kwargs["generation_config"] = gen_config

    # 6. GENERATE WITH EXPLICIT ERROR HANDLING
    try:
        generated = model.decoder.generate(**generation_kwargs)
    except Exception as e:
        # If grouped beam search is not supported (common on some decoder families),
        # fall back to multinomial sampling to retain candidate diversity.
        if "group" in str(e).lower() or "attribute" in str(e).lower():
            generation_kwargs.pop("num_beam_groups", None)
            generation_kwargs.pop("diversity_penalty", None)
            generation_kwargs["do_sample"] = True
            generation_kwargs["num_beams"] = 1
            generation_kwargs["top_p"] = top_p
            generation_kwargs["temperature"] = temperature
            generated = model.decoder.generate(**generation_kwargs)
        else:
            raise e

    sequences = generated.sequences

    # 7. DECODE (Restoring GPT2/CLIP compatibility)
    captions = []
    for seq in sequences:
        output_list = seq.tolist()
        if decoding_method is not None:
            output = decoding_method(output_list)
        else:
            output = _decode_token_list(tokenizer, output_list)
        
        # Cleanup
        for stop in ['<|endoftext|>', '<|im_end|>', '<|end|>', '<eos>', '</s>']:
            output = output.split(stop)[0]
        if not return_start_end_tokens:
            for start in ['<|startoftext|>', '<|im_start|>', '<bos>', '<s>']:
                output = output.replace(start, '')
        captions.append(output.strip())

    if not compute_scores:
        return captions

    # 8. PERPLEXITY
    current_prefix_len = full_inputs_embeds.shape[1]
    expanded_prefix = full_inputs_embeds.repeat_interleave(num_candidates, dim=0)
    token_embeds = model.decoder.get_input_embeddings()(sequences)
    combined_embeds = torch.cat([expanded_prefix, token_embeds], dim=1)
    
    full_mask = torch.ones(combined_embeds.shape[:2], device=device, dtype=torch.long)
    logits = model.decoder(inputs_embeds=combined_embeds, attention_mask=full_mask).logits
    
    # Align: Logit at index T predicts Token at index T+1
    shift_logits = logits[:, current_prefix_len-1:-1, :]
    log_probs = torch.nn.functional.log_softmax(shift_logits.float(), dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=sequences.unsqueeze(-1)).squeeze(-1)

    # Valid mask (ignore padding)
    pos = torch.arange(sequences.shape[1], device=device).unsqueeze(0)
    eos_at = torch.where(sequences.eq(eos_token_id).any(1), sequences.eq(eos_token_id).float().argmax(1), torch.tensor(sequences.shape[1] - 1, device=device))
    valid_mask = pos <= eos_at.unsqueeze(1)

    mean_nll = -(token_log_probs * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
    perplexities = torch.exp(mean_nll).detach().cpu().numpy().tolist()

    return captions, perplexities

decap_model = None

def get_decap_model(device, weights_path = DECAP_COCO_WEIGHTS_PATH, prefix_size=512, hf_repo_id=None, decoder_kwargs=None):
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
    decoder_kwargs = decoder_kwargs or {}
    decap_model = DeCap(prefix_size, **decoder_kwargs)
    
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
