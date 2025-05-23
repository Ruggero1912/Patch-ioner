import torch
from torch.nn.utils.rnn import pad_sequence
from .ClipCap import ClipCaptionModel
from transformers import AutoTokenizer
from .utils import compose_discrete_prompts
from .load_annotations import load_entities_text
from .search import greedy_search, beam_search, opt_search
from .retrieval_categories import clip_texts_embeddings, image_text_simiarlity, top_k_categories
import os 
from typing import List
from argparse import Namespace


class VieCap(torch.nn.Module):

    def __init__(self, args, device, clip_name):
        super(VieCap, self).__init__()
        self.args = args = self.load_config(args)
        self.device = device
        
        self.clip_hidden_size = 640 if 'RN' in clip_name else 512

        self.entities_text, self.texts_embeddings = self.get_viecap_texts_embeddings(args, clip_name)

        self.tokenizer = AutoTokenizer.from_pretrained(args.language_model)
        self.model = ClipCaptionModel(args.continuous_prompt_length, args.clip_project_length, self.clip_hidden_size, gpt_type = args.language_model)
        self.model.load_state_dict(torch.load(args.weight_path, map_location = device), strict = False)
        self.model.to(device)

        self.eval()

    defaults = {
            #"clip_model": "ViT-B/32",
            "language_model": "gpt2",
            "continuous_prompt_length": 10,
            "clip_project_length": 10,
            "temperature": 0.01,
            "top_k": 3,
            "threshold": 0.2,
            "disable_all_entities": False,
            "name_of_entities_text": 'vinvl_vgoi_entities',
            'prompt_ensemble' : False,
            "weight_path" : '/raid/datasets/viecap_files/checkpoints/train_coco/coco_prefix-0014.pt',
            'files_path' : '/raid/datasets/viecap_files/',
            "using_hard_prompt": False,
            "soft_prompt_first": False,
            "only_hard_prompt": False,
            "using_greedy_search": False,
            "beam_width": 5,
            "text_prompt": None,
        }
    
    def load_config(self, args_dict : dict) -> Namespace:
        
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d
        # namespace should be loaded recursively
        for key, value in self.defaults.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    args_dict.setdefault(key, {}).setdefault(sub_key, sub_value)
            else:
                args_dict.setdefault(key, value)
        args = dict_to_namespace(args_dict)
        return args

    def forward(self, image_features, compute_scores : bool = False) -> List[str]:
        """
        Image Features: (batch_size, clip_hidden_size)
        - returns: List[str]
        """
        #args = self.args
        #model = self.model
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        

        image_features /= image_features.norm(2, dim = -1, keepdim = True)

        continuous_embeddings = self.model.mapping_network(image_features).view(-1, self.args.continuous_prompt_length, self.model.gpt_hidden_size)
        
        if self.args.using_hard_prompt:
            
            #logits = image_text_simiarlity(self.texts_embeddings, temperature = self.args.temperature, images_features = image_features)
            #detected_objects, _ = top_k_categories(self.entities_text, logits, self.args.top_k, self.args.threshold) # List[List[]], [[category1, category2, ...], [], ...]
            #detected_objects = detected_objects[0] # infering single image -> List[category1, category2, ...]
            #discrete_tokens = compose_discrete_prompts(self.tokenizer, detected_objects).unsqueeze(dim = 0).to(self.args.device)
            logits = image_text_simiarlity(self.texts_embeddings, temperature=self.args.temperature, images_features=image_features)
            all_discrete_tokens = []
            for i in range(image_features.shape[0]):
                detected_objects, _ = top_k_categories(self.entities_text, logits[i:i+1], self.args.top_k, self.args.threshold)
                discrete_tokens = compose_discrete_prompts(self.tokenizer, detected_objects[0])
                all_discrete_tokens.append(discrete_tokens)

            all_discrete_tokens = [t.to(self.device) for t in all_discrete_tokens]
            discrete_tokens = pad_sequence(all_discrete_tokens, batch_first=True, padding_value=pad_id)
            #discrete_tokens = torch.stack(all_discrete_tokens).to(self.device)

            discrete_embeddings = self.model.word_embed(discrete_tokens)
            if self.args.only_hard_prompt:
                embeddings = discrete_embeddings
            elif self.args.soft_prompt_first:
                embeddings = torch.cat((continuous_embeddings, discrete_embeddings), dim = 1)
            else:
                embeddings = torch.cat((discrete_embeddings, continuous_embeddings), dim = 1)
        else:
            embeddings = continuous_embeddings
        
        if 'gpt' in self.args.language_model:
            if not self.args.using_greedy_search:
                
                #sentences = beam_search(embeddings = embeddings, tokenizer = self.tokenizer, beam_width = self.args.beam_width, model = self.model.gpt) # List[str]
                # make one beam_search call for each element in the batch
                sentences = []
                for i in range(embeddings.shape[0]):
                    sentence = beam_search(embeddings = embeddings[i:i+1], tokenizer = self.tokenizer, beam_width = self.args.beam_width, model = self.model.gpt)
                    sentences.append(sentence[0])
            else:
                sentences = greedy_search(embeddings = embeddings, tokenizer = self.tokenizer, model = self.model.gpt)
        else:
            sentences = opt_search(prompts=self.args.text_prompt, embeddings = embeddings, tokenizer = self.tokenizer, beam_width = self.args.beam_width, model = self.model.gpt)
        
        if compute_scores:
            perplexities = self.compute_perplexity(
                sentences,
                tokenizer=self.tokenizer,
                model=self.model.gpt,
                device=self.device,
            )
            return sentences, perplexities
        else:
            return sentences
    
    def compute_perplexity(self, sentences, tokenizer, model, device):
        perplexities = []
        model.eval()
        with torch.no_grad():
            for sentence in sentences:
                encodings = tokenizer(sentence, return_tensors="pt").to(device)
                input_ids = encodings.input_ids
                attention_mask = encodings.attention_mask

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                perplexity = torch.exp(loss).item()
                perplexities.append(perplexity)
        return perplexities


    def get_viecap_texts_embeddings(self, args, clip_name : str):
        clip_name = clip_name.replace('/', '')
        

        # loading categories vocabulary for objects
        if args.name_of_entities_text == 'visual_genome_entities':
            entities_text = load_entities_text(args.name_of_entities_text, os.path.join(args.files_path, 'annotations/vocabulary/all_objects_attributes_relationships.pickle'), not args.disable_all_entities)
            if args.prompt_ensemble: # loading ensemble embeddings
                texts_embeddings = clip_texts_embeddings(entities_text, os.path.join(args.files_path, f'annotations/vocabulary/visual_genome_embedding_{clip_name}_with_ensemble.pickle'))
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, os.path.join(args.files_path, f'annotations/vocabulary/visual_genome_embedding_{clip_name}.pickle'))
        elif args.name_of_entities_text == 'coco_entities':
            entities_text = load_entities_text(args.name_of_entities_text, os.path.join(args.files_path, 'annotations/vocabulary/coco_categories.json'), not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, os.path.join(args.files_path, f'annotations/vocabulary/coco_embeddings_{clip_name}_with_ensemble.pickle'))
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, os.path.join(args.files_path, f'annotations/vocabulary/coco_embeddings_{clip_name}.pickle'))
        elif args.name_of_entities_text == 'open_image_entities':
            entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/oidv7-class-descriptions-boxable.csv', not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}_with_ensemble.pickle')
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/open_image_embeddings_{clip_name}.pickle')
        elif args.name_of_entities_text == 'vinvl_vg_entities':
            entities_text = load_entities_text(args.name_of_entities_text, './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json', not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}_with_ensemble.pickle')
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, f'./annotations/vocabulary/vg_embeddings_{clip_name}.pickle')
        elif args.name_of_entities_text == 'vinvl_vgoi_entities':
            entities_text = load_entities_text(args.name_of_entities_text, os.path.join(args.files_path, 'annotations/vocabulary/vgcocooiobjects_v1_class2ind.json'), not args.disable_all_entities)
            if args.prompt_ensemble:
                texts_embeddings = clip_texts_embeddings(entities_text, os.path.join(args.files_path, f'annotations/vocabulary/vgoi_embeddings_{clip_name}_with_ensemble.pickle'))
            else:
                texts_embeddings = clip_texts_embeddings(entities_text, os.path.join(args.files_path, f'annotations/vocabulary/vgoi_embeddings_{clip_name}.pickle'))
        else:
            print('The entities text should be input correctly!')
            return None
        return entities_text, texts_embeddings

