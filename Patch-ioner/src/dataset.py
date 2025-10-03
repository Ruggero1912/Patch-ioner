import torch
from torch.utils.data import Dataset

from tqdm import tqdm
import json
from typing import Tuple
import clip
import random
import json
import random
from tqdm import tqdm

class ClipCocoDataset(Dataset):

    def __len__(self) -> int:
        return len(self.captions_tokens)
    
    def pad_tokens(self, item: int):
        tokens = self.captions_tokens[item]
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # tokens = self.captions_tokens[item]
        
        clip_tokens = self.pad_tokens(item)
        if self.feats is None:
            clip_tokens_77 = self.captions_tokens[item]
            return clip_tokens, clip_tokens_77
        else:
            return clip_tokens, self.feats[item]

    def __init__(self, data_path: str,  clip_model=None, talk2dino=None, use_dino_feats=False, tokenizer=None):
        if tokenizer is not None:
            self.clip_tokenizer = tokenizer
        else:
            print(f"Using default tokenizer")
            self.clip_tokenizer = clip.tokenize
        self.prefix_length = 10
        self.max_seq_len = 20
        self.feats = None
        
        if clip_model is not None:
            device = next(clip_model.parameters()).device
            print("Pre-extracting features...")

        if not use_dino_feats:
            with open(data_path, 'r') as f:
                self.captions = [ann['caption'] for ann in json.load(f)['annotations']]
        else:
            data = torch.load(data_path)
            self.captions = [ann['caption'] for ann in data['annotations']]
            self.feats = [ann['features'] for ann in data['annotations']]

                    
        random.shuffle(self.captions)
        self.captions_tokens = []
        
        batch_size = 64
        batched_captions = [self.captions[i:i + batch_size] for i in range(0, len(self.captions), batch_size)]

        for batch in tqdm(batched_captions):
            try:
                # Tokenize the batch of captions
                batch_tokens = [torch.tensor(self.clip_tokenizer(caption)[0], dtype=torch.int64) for caption in batch]
                
                # Pad tokens to the same length for batching
                batch_tokens_padded = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True)
                self.captions_tokens.extend(batch_tokens)

                if clip_model is not None:
                    with torch.no_grad():
                        # Encode the text batch
                        feats = clip_model.encode_text(batch_tokens_padded.to(device))
                        
                        if talk2dino is not None:
                            # Project to desired feature space
                            feats = talk2dino.project_clip_txt(feats).to('cpu')

                        # Concatenate features
                        if self.feats is None:
                            self.feats = feats
                        else:
                            self.feats = torch.cat((self.feats, feats))
            except Exception as e:
                print(f"Error processing batch: {e}")
        print(len(self.captions_tokens))

    
