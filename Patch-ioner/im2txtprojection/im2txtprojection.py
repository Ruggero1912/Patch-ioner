from enum import Enum
import numpy as np
import math
import json
import random
import torch
from tqdm import tqdm
import os
import h5py
from typing import Tuple
from dotenv import load_dotenv


load_dotenv()

class ProjectionType(Enum):
    COCO_CAPTIONS       = 'coco_captions'
    MS_MARCO_QUERIES_A  = 'ms_marco_queries_a'
    CC3M_BLIP           = 'cc3m_blip_captions'
    VISUAL_GENOME       = 'vg_captions'
    VISUAL_GENOME_TEST  = "vg_dense_captions_test"
    ONLINE_TEXTS        = "online_texts"

class Im2TxtProjector:

    SUPPORT_MEMORY_SIZE = 500000

    __IM2TXT_MEMORY_PATH = os.getenv("IM2TXT_MEMORY_PATH")
    __DECAP_FOLDER = os.path.join(os.path.dirname(__file__), "../")
    __TALK2DINO_CONFIG_WEIGHTS_PATH = __DECAP_FOLDER

    captions_dataType = 'train2017'
    ANNOTATIONS_CAPTION_FILE_PATH = os.path.join(__DECAP_FOLDER, 'captions_{}.json'.format(captions_dataType))
    VG_ANNOTATIONS_DENSE_CAPTIONS_FILE_PATH = '/raid/datasets/densecaptioning-annotations/data/vg/controlcap/vg1.2/train.json'
    VG_ANNOTATIONS_DENSE_CAPTIONS_TEST_FILE_PATH = '/raid/datasets/densecaptioning-annotations/data/vg/controlcap/vg1.2/test.json'

    CC3M_BLIP_FILE_PATH = os.path.join(__DECAP_FOLDER, "blipv2_captions.txt")

    def __init__(self, type = ProjectionType.COCO_CAPTIONS, verbose : bool = True, device_str = "cpu", use_talk2dino : bool = True, 
                 support_memory_size : int = SUPPORT_MEMORY_SIZE, batch_size=1000, 
                 clip_modelname = None, linear_talk2dino : bool = False, 
                 normalize_memory_embs : bool = False, talk2dino_attn_type='qkv', online_texts=None,
                 memory_bank_name = None) -> None:
        """
        - normalize_memory_embs -> normalizes the embeddings memory (required for projection in CLIP space)
        - type : ProjectionType -> the type of the support memory to be built . Can either be the path to the file containing the captions or the type of the support memory to be built
        """
        # check if hdf5 already exists, otherwhise builds the support memory for that kind
        
        #if type not in ProjectionType.mro()

        self.type = type
        self.device_str = device_str
        self.device = torch.device(self.device_str)
        self.use_talk2dino = use_talk2dino
        self.linear_talk2dino = linear_talk2dino
        self.talk2dino_attn_type = talk2dino_attn_type
        self.online_texts = online_texts
        if clip_modelname is None:
            clip_modelname = "ViT-B/16" if self.use_talk2dino else "ViT-B/32"
        self.clip_modelname = clip_modelname

        self.SUPPORT_MEMORY_SIZE = support_memory_size
        if use_talk2dino:
            prefix = ""
            postfix = '-B16' if use_talk2dino is True else use_talk2dino
            if linear_talk2dino:
                postfix += "-linear"
        else:
            prefix = "clip-"
            postfix = ""
        if talk2dino_attn_type != 'qkv':
            self.talk2dino_attn_type_str = f"_{talk2dino_attn_type}"
        else:
            self.talk2dino_attn_type_str = ''
        if isinstance(type, ProjectionType):
            dataset_name = type.value 
        elif memory_bank_name is not None:
            dataset_name = memory_bank_name
        else:
            dataset_name = 'coco'

        self.H5PY_FILE_PATH = os.path.join( self.__IM2TXT_MEMORY_PATH, prefix + f'{dataset_name}{self.talk2dino_attn_type_str}_text_embeddings{postfix}-{clip_modelname.replace("/", ".")}-{self.SUPPORT_MEMORY_SIZE}.h5' )
        self.H5PY_EMBEDDINGS_DATASET_NAME = '{}-embeddings'.format(dataset_name)
        self.H5PY_TEXT_DATASET_NAME = '{}-text'.format(dataset_name)

        embs_dataset, text_dataset = self._load_support_memory()

        if text_dataset is None:
            if verbose: print(f"[+] Going to build support memory for the given data type: {type} [+]")
            embs_dataset, text_dataset = self._build_support_memory(batch_size)
            if verbose: print(f"[+] Done [+]")
        
        if self.type != ProjectionType.ONLINE_TEXTS:
            embs_dataset, text_dataset = self._load_support_memory()
        
        print(f"[-] loaded memory from {os.path.abspath( self.H5PY_FILE_PATH )} [-]")

        self.text_dataset = text_dataset
        self.embs_dataset = torch.tensor(embs_dataset[:]).to(self.device)
        self.embs_dataset = self.embs_dataset[self.embs_dataset.norm(dim=-1) != 0]


        if normalize_memory_embs:
            self.embs_dataset /= self.embs_dataset.norm(dim=-1,keepdim=True).float()
        


    def project(self, image_embedding, temperature : float = 0.01, normalize : bool = False, return_argmax_text : bool = False, return_n_best_sims=None) -> torch.TensorType:
        if not isinstance(image_embedding, torch.Tensor):
            print(f"the type of image_embedding is '{type(image_embedding)}' converting it to torch tensor")
            image_embedding = torch.tensor(image_embedding, dtype=torch.float).to(self.device)
        
        orig_device = image_embedding.device
        
        if image_embedding.device != self.device:
            image_embedding = image_embedding.to(self.device)
        
        embs_dataset = self.embs_dataset / self.embs_dataset.norm(dim=-1, keepdim=True)
        image_embedding /= image_embedding.norm(dim=-1,keepdim=True)
        
        sim = image_embedding@embs_dataset.T.float()
        if return_argmax_text:
            argmax_texts = [self.text_dataset[idx].decode() for idx in sim.argmax(dim=-1)]
            if return_n_best_sims:
                return argmax_texts, sim.sort(dim=-1, descending=True).values[:, :return_n_best_sims].tolist()
            return argmax_texts
        softmax_sim = (sim / temperature).softmax(dim=-1)
        prefix_embedding = softmax_sim@self.embs_dataset.float()
    
        if normalize:
            prefix_embedding /= prefix_embedding.norm(dim=-1,keepdim=True)
        
        if return_n_best_sims:
            return prefix_embedding.to(orig_device), sim.sort(dim=-1, descending=True).values[:, :return_n_best_sims].tolist()

        return prefix_embedding.to(orig_device)

    def _load_support_memory(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.type == ProjectionType.ONLINE_TEXTS:
            print(f"[-] _load_support_memory: support memory for provided texts will be constructed [-]")
            return None, None          
        if not os.path.exists(self.H5PY_FILE_PATH):
            print(f"[-] _load_support_memory: the path '{self.H5PY_FILE_PATH}' does not exist [-]")
            return None, None

        with h5py.File(self.H5PY_FILE_PATH, 'a') as hf:

            if self.H5PY_EMBEDDINGS_DATASET_NAME in hf:
                embeddings_dataset = hf[self.H5PY_EMBEDDINGS_DATASET_NAME][:]
                text_dataset = hf[self.H5PY_TEXT_DATASET_NAME][:]
            else:
                embeddings_dataset = None
                text_dataset = None
        
        return embeddings_dataset, text_dataset


        
    def _build_support_memory(self, batch_size = 1000) -> Tuple[np.ndarray, np.ndarray]:
        ## construct the support memory

        self._load_models()

        if self.type == ProjectionType.COCO_CAPTIONS:
            from pycocotools.coco import COCO
            coco_obj = COCO(Im2TxtProjector.ANNOTATIONS_CAPTION_FILE_PATH)
            data = random.sample(list(coco_obj.anns.values()), k=self.SUPPORT_MEMORY_SIZE)
            data = [ d['caption'] for d in data ]
        elif self.type == ProjectionType.VISUAL_GENOME:
            from pycocotools.coco import COCO
            coco_obj = COCO(Im2TxtProjector.VG_ANNOTATIONS_DENSE_CAPTIONS_FILE_PATH)
            # data = random.sample(list(coco_obj.anns.values()), k=self.SUPPORT_MEMORY_SIZE)
            data = list(coco_obj.anns.values())[:self.SUPPORT_MEMORY_SIZE]
            data = [ d['caption'] for d in data ]
        elif self.type == ProjectionType.VISUAL_GENOME_TEST:
            from pycocotools.coco import COCO
            coco_obj = COCO(Im2TxtProjector.VG_ANNOTATIONS_DENSE_CAPTIONS_TEST_FILE_PATH)
            # data = random.sample(list(coco_obj.anns.values()), k=self.SUPPORT_MEMORY_SIZE)
            data = list(coco_obj.anns.values())[:self.SUPPORT_MEMORY_SIZE]
            data = [ d['caption'] for d in data ]
        elif self.type == ProjectionType.MS_MARCO_QUERIES_A:
            print(f"Loading MSMarco queries from file ", Im2TxtProjector.MS_MARCO_QUERIES_FILE_PATH)
            with open(Im2TxtProjector.MS_MARCO_QUERIES_FILE_PATH, "r") as input_file:
                lines = input_file.readlines()
            data = random.sample(lines, k=self.SUPPORT_MEMORY_SIZE)
            data = [ d.split("\t")[1].replace("\n", "") for d in data ]
            print(f"Loaded from file '{self.SUPPORT_MEMORY_SIZE}' lines, example of line: '{data[0]}'")
        elif self.type == ProjectionType.CC3M_BLIP:
            print(f"Loading cc3m captions txt file ", Im2TxtProjector.CC3M_BLIP_FILE_PATH)
            with open(Im2TxtProjector.CC3M_BLIP_FILE_PATH, "r") as input_file:
                lines = input_file.readlines()
            data = random.sample(lines, k=self.SUPPORT_MEMORY_SIZE)
            data = [ d.replace("\n", "") for d in data ]
            print(f"Loaded from file '{len(data)}' lines, example of line: '{data[0]}'")
        elif self.type == ProjectionType.CC3M_BLIP:
            print(f"Loading cc3m captions txt file ", Im2TxtProjector.CC3M_BLIP_FILE_PATH)
            with open(Im2TxtProjector.CC3M_BLIP_FILE_PATH, "r") as input_file:
                lines = input_file.readlines()
            data = random.sample(lines, k=self.SUPPORT_MEMORY_SIZE)
            data = [ d.replace("\n", "") for d in data ]
            print(f"Loaded from file '{len(data)}' lines, example of line: '{data[0]}'") 
        elif self.type == ProjectionType.ONLINE_TEXTS:
            data = self.online_texts
            print(f"Loaded online_texts '{len(data)}' lines, example of line: '{data[0]}'") 
        elif type(self.type) == str:
            if os.path.exists(self.type):
                path = self.type
                from pycocotools.coco import COCO
                coco_obj = COCO(path)
                data = random.sample(list(coco_obj.anns.values()), k=min(self.SUPPORT_MEMORY_SIZE, len(coco_obj.anns)))
                data = [ d['caption'] for d in data ]
        else:
            #data = random.sample(data,500000)
            print(f"[!] Unimplemented data type '{self.type}'[!]")
            return None, None
        
        text_features = []
        captions = []
        
        self.clip_model.eval()
        
        n_txts = len(data)
        n_batch = math.ceil(n_txts / batch_size)
        for i in tqdm(range(n_batch)):
            start = i * batch_size
            end = start + batch_size if i < n_batch - 1 else n_txts
            
            texts = data[start:end]
            with torch.no_grad():
                texts_token = self.tokenizer(texts).to(self.device)
                text_feature = self.clip_model.encode_text(texts_token)
                if self.use_talk2dino:
                    text_feature = self.talk2dino.project_clip_txt(text_feature)
                text_features.append(text_feature)
                captions.extend(texts)

        text_features = torch.cat(text_features,dim=0)

        #text_features /= text_features.norm(dim=-1,keepdim=True).float()
        
        # store captions and text features in hdf5 dataset

        text_features_ndarray = text_features.cpu().numpy()

        assert len(text_features_ndarray) == len(captions), f"len(text_features_ndarray) = {len(text_features_ndarray)} != len(captions) = {len(captions)}"

        #if not os.path.exists(self.H5PY_FILE_PATH):
        #    print(f"os.path '{self.H5PY_FILE_PATH}' does not exists")

        EMBEDDINGS_DIMENSION = text_features_ndarray.shape[1]
            
        if self.type != ProjectionType.ONLINE_TEXTS:
            with h5py.File(self.H5PY_FILE_PATH, 'w') as hf:

                if self.H5PY_EMBEDDINGS_DATASET_NAME in hf:
                    embeddings_dataset = hf[self.H5PY_EMBEDDINGS_DATASET_NAME]
                    text_dataset = hf[self.H5PY_TEXT_DATASET_NAME]
                    print(f"[!] Dataset '{self.H5PY_EMBEDDINGS_DATASET_NAME}' already exists! Going to overwrite [!]")
                else:
                    embeddings_dataset = hf.create_dataset(self.H5PY_EMBEDDINGS_DATASET_NAME, shape=(self.SUPPORT_MEMORY_SIZE, EMBEDDINGS_DIMENSION), dtype='float32')
                    text_dataset = hf.create_dataset(self.H5PY_TEXT_DATASET_NAME, shape=(self.SUPPORT_MEMORY_SIZE, ), dtype=h5py.string_dtype(encoding='utf-8'))    #, dtype='str'
            
                for num_row in range(len(text_features_ndarray)):
                    embeddings_dataset[num_row] = text_features_ndarray[num_row]
                    text_dataset[num_row] = captions[num_row]
        else:
            embeddings_dataset = text_features_ndarray
            text_dataset = [x.encode() for x in captions]

        return embeddings_dataset, text_dataset

    clip_model = None
    def _load_models(self):

        if self.clip_model is not None:
            # case already done
            return

        import clip
        if self.clip_modelname is None:
            clip_model_name = "ViT-B/16" if self.use_talk2dino else "ViT-B/32"
        else:
            clip_model_name = self.clip_modelname
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=self.device, jit=False)
        self.tokenizer = clip.tokenize
        if self.use_talk2dino:
            # loading Talk2DINO
            if type(self.use_talk2dino) == str:
                proj_name = self.use_talk2dino
            elif self.linear_talk2dino is False:
                proj_name = 'vitb_mlp_infonce'
            else:
                proj_name = 'vitb_linear_infonce'
            
            
            config = os.path.join(self.__TALK2DINO_CONFIG_WEIGHTS_PATH, "configs_talk2dino", proj_name + '.yaml')
            weights = os.path.join(self.__TALK2DINO_CONFIG_WEIGHTS_PATH, "weights_talk2dino", proj_name + self.talk2dino_attn_type_str + '.pth')
            #import sys
            #import os
            #add_path = os.path.abspath( os.path.dirname("../"))
            ##print(add_path)
            #sys.path.insert(1, add_path )
            from src.model import ProjectionLayer
            self.talk2dino = ProjectionLayer.from_config(config)
            self.talk2dino.load_state_dict(torch.load((weights), self.device))
            self.talk2dino.to(self.device)

