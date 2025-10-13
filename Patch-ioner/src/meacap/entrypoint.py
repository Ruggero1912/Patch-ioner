import os, sys


if os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) not in sys.path:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ..hf_utils import get_model_path_with_hf_fallback

from ..viecap.entrypoint import VieCap
from ..viecap.utils import compose_discrete_prompts
from ..viecap.search import greedy_search, beam_search, opt_search

from .models.clip_utils import CLIP

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch, json
from torch.nn.utils.rnn import pad_sequence

from typing import List

from .utils.detect_utils import retrieve_concepts

class MeaCap(VieCap):

    retrieve_on_CPU = False

    def __init__(self, args, device, clip_name):
        super(MeaCap, self).__init__(args, device, clip_name)

        args = self.args

        self.vl_model = CLIP(args.meacap.vl_model)
        self.vl_model = self.vl_model.to(self.device)
        print('[MeaCap] Loaded CLIP vl_model from the checkpoint {}.'.format(args.meacap.vl_model))

        self.wte_model = SentenceTransformer(args.meacap.wte_model_path, device=self.device)
        print('[MeaCap] Load sentenceBERT from the checkpoint {}.'.format(args.meacap.wte_model_path))

        # --- Load the Textual Scene Graph parser ---
        if torch.cuda.is_available() and "cuda" in str(self.device):
            # Load parser directly on the selected CUDA device
            with torch.cuda.device(self.device):
                self.parser_tokenizer = AutoTokenizer.from_pretrained(args.meacap.parser_checkpoint)
                self.parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.meacap.parser_checkpoint)
        else:
            # Fallback to CPU (avoid using torch.cuda.device)
            self.parser_tokenizer = AutoTokenizer.from_pretrained(args.meacap.parser_checkpoint)
            self.parser_model = AutoModelForSeq2SeqLM.from_pretrained(args.meacap.parser_checkpoint)
        self.parser_model.eval()
        self.parser_model.to(self.device)
        print('[MeaCap] Load Textual Scene Graph parser from the checkpoint {}.'.format(args.meacap.parser_checkpoint))

        memory_id = args.meacap.memory_id
        memory_base_path = args.meacap.memory_base_path
        
        memory_caption_relative_path = f"memory/{memory_id}/memory_captions.json"
        memory_caption_local_path = os.path.join(memory_base_path, memory_caption_relative_path)
        memory_caption_path = get_model_path_with_hf_fallback(
            local_path=memory_caption_local_path,
            hf_repo_id=args.meacap.hf_repo_id,
            filename=memory_caption_relative_path
        )
        memory_clip_embedding_relative_path = f"memory/{memory_id}/memory_clip_embeddings.pt"
        memory_clip_embedding_local_file = os.path.join(memory_base_path, memory_clip_embedding_relative_path)
        memory_clip_embedding_file = get_model_path_with_hf_fallback(
            local_path=memory_clip_embedding_local_file,
            hf_repo_id=args.meacap.hf_repo_id,
            filename=memory_clip_embedding_relative_path
        )
        memory_wte_embedding_relative_path = f"memory/{memory_id}/memory_wte_embeddings.pt"
        memory_wte_embedding_local_file = os.path.join(memory_base_path, memory_wte_embedding_relative_path)
        memory_wte_embedding_file = get_model_path_with_hf_fallback(
            local_path=memory_wte_embedding_local_file,
            hf_repo_id=args.meacap.hf_repo_id,
            filename=memory_wte_embedding_relative_path
        )
        
        self.memory_clip_embeddings = torch.load(memory_clip_embedding_file, map_location=self.device).to(self.device)
        self.memory_wte_embeddings = torch.load(memory_wte_embedding_file, map_location=self.device).to(self.device)
        with open(memory_caption_path, 'r') as f:
            self.memory_captions = json.load(f)
        print('[MeaCap] Loaded memory bank for memory_id {}.'.format(memory_id))

        self.vl_model_retrieve = self.vl_model

        self.eval()
    
    def get_viecap_texts_embeddings(self, args, clip_name):
        return None, None
    
    def load_config(self, args_dict):
        default = {
            "meacap" : {
                "memory_caption_num" : 5,
                "vl_model" : "openai/clip-vit-base-patch32",
                "wte_model_path" : "sentence-transformers/all-MiniLM-L6-v2",
                "parser_checkpoint" : "lizhuang144/flan-t5-base-VG-factual-sg",
                "memory_id" : "coco",
                "memory_base_path" : "/raid/datasets/meacap_files/"
            }
        }

        def deep_merge(dict1, dict2):
            """
            Recursively merges the contents of dict2 into dict1.
            - the value from dict2 overwrites the value in dict1
            For each key in dict2:
                - If the key exists in dict1 and both values are dictionaries, merge them recursively.
                - Otherwise, the value from dict2 overwrites the value in dict1.
            Parameters:
                dict1 (dict): The dictionary to be updated in place.
                dict2 (dict): The dictionary whose values will be merged into dict1.
            """
            for key, value in dict2.items():
                if (
                    key in dict1 and isinstance(dict1[key], dict)
                    and isinstance(value, dict)
                ):
                    deep_merge(dict1[key], value)
                else:
                    dict1[key] = value
            return dict1

        deep_merge(default, args_dict) # the priority of the default config is lower than the user input
        args_dict = default

        return super().load_config(args_dict)
    
    def forward(self, image_features, compute_scores : bool = False, eval_mode : bool = True) -> List[str]:

        if eval_mode:
            self.eval()

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        image_features /= image_features.norm(2, dim = -1, keepdim = True)
        
        continuous_embeddings = self.model.mapping_network(image_features).view(-1, self.args.continuous_prompt_length, self.model.gpt_hidden_size)
        
        if self.args.using_hard_prompt:
            
            #batch_image_embeds = self.vl_model.compute_image_representation_from_image_path(self.args.image_path)
            batch_image_embeds = image_features

            if self.retrieve_on_CPU != True:
                #batch _size = batch_image_ embeds.sha pe[0]
                #memory_clip_em beds_batched = self.memory_c lip_embeddings.unsq ueeze(0).repeat(batch_size, 1, 1)
                #clip_sc ore, cli p_ref = self.vl_model_r etrieve.compute_image _text_similarity_via_embeddings(
                #    batch_image_e mbeds, memory_clip _embeds_batched)
                clip_score, clip_ref = self.vl_model_retrieve.compute_image_text_similarity_via_embeddings_new(
                    batch_image_embeds, self.memory_clip_embeddings)
            else:

                raise Exception("retrieve_on_CPU is not supported in this version.")
                #batch_image_embeds_cpu = batch_image_embeds.to(cpu_device)
                #clip_score_cpu, clip_ref_cpu = vl_model_retrieve.compute_image_text_similarity_via_embeddings(
                #    batch_image_embeds_cpu,
                #    memory_clip_embeddings)
                #clip_score = clip_score_cpu.to(device)
                #clip_ref = clip_ref_cpu.to(device)

            select_memory_ids_batch = clip_score.topk(self.args.meacap.memory_caption_num, dim=-1)[1]#.squeeze(0)

            all_discrete_tokens = []

            for select_memory_ids in select_memory_ids_batch:
                select_memory_captions = [self.memory_captions[id] for id in select_memory_ids]
                select_memory_wte_embeddings = self.memory_wte_embeddings[select_memory_ids]
                detected_objects = retrieve_concepts(parser_model=self.parser_model, parser_tokenizer=self.parser_tokenizer,
                                                    wte_model=self.wte_model,
                                                    select_memory_captions=select_memory_captions,
                                                    image_embeds=batch_image_embeds,
                                                    device=self.device)

                #print("memory concepts:", detected_objects)
                discrete_tokens = compose_discrete_prompts(self.tokenizer, detected_objects).to(self.device) #.unsqueeze(dim = 0)
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