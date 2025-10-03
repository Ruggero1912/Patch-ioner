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

from pycocotools.coco import COCO

class ClipCocoDatasetMix(Dataset):

    def __len__(self) -> int:
        return len(self.image_index_list)

    def _pad_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            tokens = tokens[:self.max_seq_len]
        return tokens


    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        
        # get the image index for the item
        img_idx = self.image_index_list[item]
        # get the caption index for that image
        first_caption_idx = self.image_index_list.index(img_idx)

        # the caption index is the item - the first caption index
        caption_idx = item - first_caption_idx

        # how many captions are there for that image?
        num_captions = len(self.captions_list_of_lists[img_idx])
        try:
            tokens = self.captions_tokens_list_of_lists[img_idx][caption_idx] #self.captions_list_of_lists[img_idx][caption_idx]
        except IndexError:
            print(f"{len(self.captions_tokens_list_of_lists)= } - {len(self.captions_tokens_list_of_lists[img_idx])= }")
            print(f"IndexError: {img_idx}, {caption_idx}, {num_captions}")
            raise
        padded_tokens = self._pad_tokens(tokens)

        feats_same_img = self.feats[img_idx][random.choice(range(num_captions))]

        if self.feats is None or len(self.feats) == 0:
            raise Exception("Precomputed features required")
        else:
            return padded_tokens, feats_same_img

    def __init__(self, data_path: str,  clip_model=None, talk2dino=None, use_precomputed_feats=False, tokenizer=None):
        
        batch_size = 64
        self.max_seq_len = 20

        if use_precomputed_feats:
            raise Exception("Precomputed features not supported")

        if tokenizer is not None:
            self.clip_tokenizer = tokenizer
        else:
            print(f"Using default tokenizer")
            self.clip_tokenizer = clip.tokenize

        coco_data = COCO(data_path)
        # I want to load the captions from the json file in a list of lists, 
        # where each list contains the captions for a single image

        self.captions_list_of_lists = []
        
        self.image_index_list = []

        max_seq_len = 20

        for img_idx, (img_id, image) in enumerate(list(coco_data.imgs.items())):
            # get the captions for that image
            captions = coco_data.imgToAnns[img_id]
            # get the texts of the captions
            captions = [cap['caption'] for cap in captions] #[coco_data.anns[cap]['caption'] for cap in captions]
            self.captions_list_of_lists.append(captions)
            self.image_index_list.append([img_idx] * len(captions))

            #max_seq_len = max(max_seq_len, max([len(caption) for caption in captions]))
        
        self.max_seq_len = max_seq_len
        print(f"Computed Max seq len: {max_seq_len}")

        if clip_model is not None:
            device = next(clip_model.parameters()).device
            print("Pre-extracting features...")
                    
        #random.shuffle(self.captions_list_of_lists)
        # should shuffle in the same way self.image_index_list and self.captions_list_of_lists
        # Combine captions and image indices into a list of pairs
        combined = list(zip(self.captions_list_of_lists, self.image_index_list, range(len(self.captions_list_of_lists))))

        # Shuffle them together
        random.shuffle(combined)

        # Unzip the shuffled pairs back into two separate lists
        self.captions_list_of_lists, self.image_index_list, img_idxes_shuffled = zip(*combined)
        # Convert back to lists (zip returns tuples)
        self.captions_list_of_lists = list(self.captions_list_of_lists)
        self.image_index_list = list(self.image_index_list)
        img_idxes_shuffled = list(img_idxes_shuffled)

        # self.image_index_list is a list of lists, where each list contains the image index for each caption,
        # so we need to flatten it
        self.image_index_list = [img_idxes_shuffled.index(item) for sublist in self.image_index_list for item in sublist]


        self.captions_tokens_list_of_lists = []
        self.feats = [] # feats will be a list of tensors, each tensor will be (num_captions, embedding_dimension)
        #ignore. # feats shape will be (num_images, num_captions, embedding_dimension)
        
        #batched_captions = [self.captions[i:i + batch_size] for i in range(0, len(self.captions), batch_size)]

        for captions_list in tqdm(self.captions_list_of_lists, dynamic_ncols=True):
            try:
                # Tokenize the batch of captions
                batch_tokens = [torch.tensor(self.clip_tokenizer(caption)[0], dtype=torch.int64) for caption in captions_list]
                
                # Pad tokens to the same length for batching
                batch_tokens_padded = torch.nn.utils.rnn.pad_sequence(batch_tokens, batch_first=True)
                self.captions_tokens_list_of_lists.append(batch_tokens)

                # alternative:
                # tokens = self.clip_tokenizer(captions_list, truncate=True).to(device)  # shape: (num_captions, context_length)


                if clip_model is not None:
                    with torch.no_grad():
                        # Encode the text batch
                        feats = clip_model.encode_text(batch_tokens_padded.to(device))
                        
                        if talk2dino is not None:
                            # Project to desired feature space
                            feats = talk2dino.project_clip_txt(feats).to('cpu')

                        self.feats.append(feats.cpu())  # store (num_captions, embed_dim) for each image

            except Exception as e:
                print(f"Error processing batch: {e}")
        
        print(f"Dataset loaded with {len(self.captions_list_of_lists)} images")
        print(f"Max seq len: {max_seq_len}")
        print(f"Number of captions: {len(self.image_index_list)}")
    
