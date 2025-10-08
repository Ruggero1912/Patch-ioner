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
from ...dinotxt_utils import get_tokenizer
# Import HuggingFace Hub utilities for memory bank downloading
from ...hf_utils import get_model_path_with_hf_fallback

load_dotenv()

class ProjectionType(Enum):
    COCO_CAPTIONS       = 'coco_captions'
    MS_MARCO_QUERIES_A  = 'ms_marco_queries_a'
    CC3M_BLIP           = 'cc3m_blip_captions'
    VISUAL_GENOME       = 'vg_captions'
    VISUAL_GENOME_TEST  = "vg_dense_captions_test"
    ONLINE_TEXTS        = "online_texts"

class Im2TxtProjector:
    """
    Im2TxtProjector creates and manages text embedding memory banks for different models:
    - Standard CLIP models
    - OpenCLIP models  
    - RegionCLIP models
    - DenseClip models
    - Talk2DINO projected embeddings
    
    For RegionCLIP usage, pass regionclip_config as a dict with:
    {
        'checkpoint': '/path/to/regionclip_checkpoint.pth',
        'config_name': 'RegionCLIP_RN50.yaml'  # optional
    }
    
    For DenseClip usage, pass denseclip_config as a string with the config file name:
    'denseclip_vitb16'  # or other valid DenseClip config name
    """

    SUPPORT_MEMORY_SIZE = 500000

    __IM2TXT_MEMORY_PATH = os.getenv("IM2TXT_MEMORY_PATH")
    
    if __IM2TXT_MEMORY_PATH is None:
        default_path = "/im2txtmemories" #os.path.join(os.path.dirname(__file__), "../../../im2txtmemories")
        print(f"[!] Warning: IM2TXT_MEMORY_PATH not set in environment variables, using '{default_path}' [!]")
        __IM2TXT_MEMORY_PATH = default_path
    
    __DECAP_FOLDER = os.path.join(os.path.dirname(__file__), "../")
    __TALK2DINO_CONFIG_WEIGHTS_PATH = __DECAP_FOLDER

    captions_dataType = 'train2017'
    ANNOTATIONS_CAPTION_FILE_PATH = os.path.join(__DECAP_FOLDER, 'captions_{}.json'.format(captions_dataType))
    VG_ANNOTATIONS_DENSE_CAPTIONS_FILE_PATH = '/raid/datasets/densecaptioning-annotations/data/vg/controlcap/vg1.2/train.json'
    VG_ANNOTATIONS_DENSE_CAPTIONS_TEST_FILE_PATH = '/raid/datasets/densecaptioning-annotations/data/vg/controlcap/vg1.2/test.json'

    CC3M_BLIP_FILE_PATH = os.path.join(__DECAP_FOLDER, "blipv2_captions.txt")
    MS_MARCO_QUERIES_FILE_PATH = '/raid/datasets/MSMarco/queries/queries.train.tsv'

    @staticmethod
    def create_regionclip_config(checkpoint_path: str, config_name: str = None):
        """
        Helper method to create RegionCLIP configuration dictionary.
        
        Args:
            checkpoint_path (str): Path to RegionCLIP checkpoint file
            config_name (str, optional): RegionCLIP config name (e.g., 'RegionCLIP_RN50.yaml')
        
        Returns:
            dict: Configuration dictionary for RegionCLIP
        """
        return {
            'checkpoint': checkpoint_path,
            'config_name': config_name
        }

    @staticmethod
    def _build_filename_components(
        projection_type, 
        clip_modelname: str,
        support_memory_size: int,
        use_talk2dino: bool = True,
        linear_talk2dino: bool = False,
        talk2dino_attn_type: str = 'qkv',
        memory_bank_name: str = None,
        use_open_clip: bool = False,
        regionclip_config=None,
        invite_config=None,
        denseclip_config=None
    ) -> tuple:
        """
        Build filename components for HDF5 memory bank file.
        
        Returns:
            tuple: (prefix, dataset_name, talk2dino_attn_type_str, postfix)
        """
        
        # Determine prefix based on model configuration
        if use_talk2dino:
            prefix = ""
            postfix = '-B16' if use_talk2dino is True else use_talk2dino
            if linear_talk2dino:
                postfix += "-linear"
        elif invite_config is not None:
            prefix = "invite-"
            postfix = ""
        elif regionclip_config is not None:
            prefix = "regionclip-"
            postfix = ""
        elif denseclip_config is not None:
            prefix = "denseclip-"
            postfix = ""
        else:
            prefix = "clip-"
            postfix = ""
            
        # Handle attention type
        if talk2dino_attn_type != 'qkv':
            talk2dino_attn_type_str = f"_{talk2dino_attn_type}"
        else:
            talk2dino_attn_type_str = ''
        
        # Determine dataset name
        if isinstance(projection_type, ProjectionType):
            dataset_name = projection_type.value 
        elif isinstance(projection_type, str):
            # Try to match string to ProjectionType
            try:
                proj_type = ProjectionType(projection_type)
                dataset_name = proj_type.value
            except ValueError:
                # If not a ProjectionType, treat as file path or custom name
                if os.path.exists(projection_type):
                    # It's a file path, try to infer dataset name
                    basename = os.path.basename(projection_type)
                    if 'karpathy' in basename.lower():
                        dataset_name = 'coco_captions'
                    elif 'coco' in basename.lower():
                        dataset_name = 'coco_captions' 
                    elif 'vg' in basename.lower():
                        dataset_name = 'vg_captions'
                    elif 'marco' in basename.lower():
                        dataset_name = 'ms_marco_queries_a'
                    else:
                        dataset_name = 'coco_captions'  # default
                else:
                    dataset_name = projection_type  # use as is
        elif memory_bank_name is not None:
            dataset_name = memory_bank_name
        else:
            dataset_name = 'coco_captions' if use_talk2dino else 'coco'  # different defaults for legacy compatibility

        # Handle postfix modifications            
        if use_open_clip:
            postfix += "-open_clip"
        elif regionclip_config is not None:
            postfix += "-regionclip"
            # Add checkpoint identifier to make filename unique
            checkpoint_path = regionclip_config.get('checkpoint', '')
            checkpoint_name = os.path.basename(checkpoint_path).replace('.pth', '').replace('.pt', '')
            if checkpoint_name:
                postfix += f"-{checkpoint_name}"
        elif denseclip_config is not None:
            postfix += "-denseclip"
            # Add config identifier to make filename unique
            config_name = os.path.basename(denseclip_config).replace('.yaml', '').replace('.yml', '')
            if config_name:
                postfix += f"-{config_name}"

        return prefix, dataset_name, talk2dino_attn_type_str, postfix

    @staticmethod
    def get_memory_bank_filename_and_path(
        projection_type, 
        clip_modelname: str = 'ViT-B/32',
        support_memory_size: int = SUPPORT_MEMORY_SIZE,
        use_talk2dino: bool = True,
        linear_talk2dino: bool = False,
        talk2dino_attn_type: str = 'qkv',
        memory_bank_name: str = None,
        use_open_clip: bool = False,
        regionclip_config=None,
        invite_config=None,
        denseclip_config=None,
        im2txt_memory_path: str = None
    ) -> tuple:
        """
        Get the HDF5 memory bank filename and full path for a given configuration.
        
        Args:
            projection_type: ProjectionType enum or string
            clip_modelname: Name of the CLIP model
            support_memory_size: Size of the memory bank
            use_talk2dino: Whether to use Talk2DINO projection
            linear_talk2dino: Whether to use linear Talk2DINO
            talk2dino_attn_type: Attention type for Talk2DINO
            memory_bank_name: Custom memory bank name
            use_open_clip: Whether to use OpenCLIP
            regionclip_config: RegionCLIP configuration
            invite_config: INViTE configuration
            denseclip_config: DenseClip configuration
            im2txt_memory_path: Path to im2txt memories (if None, uses env var or default)
            
        Returns:
            tuple: (filename, full_path) where filename is just the .h5 file name
                   and full_path is the complete path to the file
        """
        
        # Get memory path
        if im2txt_memory_path is None:
            im2txt_memory_path = os.getenv("IM2TXT_MEMORY_PATH", "/im2txtmemories")
        
        # Use the shared filename construction logic
        prefix, dataset_name, talk2dino_attn_type_str, postfix = Im2TxtProjector._build_filename_components(
            projection_type=projection_type,
            clip_modelname=clip_modelname,
            support_memory_size=support_memory_size,
            use_talk2dino=use_talk2dino,
            linear_talk2dino=linear_talk2dino,
            talk2dino_attn_type=talk2dino_attn_type,
            memory_bank_name=memory_bank_name,
            use_open_clip=use_open_clip,
            regionclip_config=regionclip_config,
            invite_config=invite_config,
            denseclip_config=denseclip_config
        )

        # Construct filename
        filename = prefix + f'{dataset_name}_text_embeddings{talk2dino_attn_type_str}{postfix}-{clip_modelname.replace("/", ".")}-{support_memory_size}.h5'
        full_path = os.path.join(im2txt_memory_path, filename)
        
        return filename, full_path

    def __init__(self, type = ProjectionType.COCO_CAPTIONS, verbose : bool = True, device_str = "cpu", use_talk2dino : bool = True, 
                 support_memory_size : int = SUPPORT_MEMORY_SIZE, batch_size=1000, 
                 clip_modelname = None, linear_talk2dino : bool = False, 
                 normalize_memory_embs : bool = False, talk2dino_attn_type='qkv', online_texts=None,
                 memory_bank_name = None, use_open_clip = False, regionclip_config=None, invite_config=None, denseclip_config=None,
                 hf_repo_id=None, memory_bank_hf_repo_id=None) -> None:
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
        self.hf_repo_id = hf_repo_id
        self.memory_bank_hf_repo_id = memory_bank_hf_repo_id or hf_repo_id
        self.use_open_clip = use_open_clip
        self.regionclip_config = regionclip_config
        self.invite_config = invite_config
        self.denseclip_config = denseclip_config
        
        if use_open_clip:
            assert use_talk2dino is False, "use_open_clip and use_talk2dino cannot be used together"
        
        if regionclip_config is not None:
            assert use_talk2dino is False, "regionclip_config and use_talk2dino cannot be used together"
            assert use_open_clip is False, "regionclip_config and use_open_clip cannot be used together"

        if invite_config is not None:
            # overwrite clip_modelname with invite_config['name'] if provided
            clip_modelname = invite_config.get('name', clip_modelname)
            assert use_talk2dino is False, "invite_config and use_talk2dino cannot be used together"
            
        if denseclip_config is not None:
            assert use_talk2dino is False, "denseclip_config and use_talk2dino cannot be used together"
            assert use_open_clip is False, "denseclip_config and use_open_clip cannot be used together"
            assert regionclip_config is None, "denseclip_config and regionclip_config cannot be used together"
            

        if clip_modelname is None:
            if self.use_talk2dino:
                clip_modelname = "ViT-B/16"
            elif regionclip_config is not None:
                # For RegionCLIP, we'll use a generic identifier since the model type is in the config
                clip_modelname = "RegionCLIP"
            elif denseclip_config is not None:
                # For DenseClip, we'll use a generic identifier since the model type is in the config
                clip_modelname = "DenseClip"
            else:
                clip_modelname = "ViT-B/32"
        self.clip_modelname = clip_modelname

        self.SUPPORT_MEMORY_SIZE = support_memory_size
        
        # Use the shared filename construction logic
        prefix, dataset_name, talk2dino_attn_type_str, postfix = self._build_filename_components(
            projection_type=type,
            clip_modelname=clip_modelname,
            support_memory_size=support_memory_size,
            use_talk2dino=use_talk2dino,
            linear_talk2dino=linear_talk2dino,
            talk2dino_attn_type=talk2dino_attn_type,
            memory_bank_name=memory_bank_name,
            use_open_clip=use_open_clip,
            regionclip_config=regionclip_config,
            invite_config=invite_config,
            denseclip_config=denseclip_config
        )
        
        # Store for later use
        self.talk2dino_attn_type_str = talk2dino_attn_type_str
        
        # Construct full file path
        filename = prefix + f'{dataset_name}_text_embeddings{talk2dino_attn_type_str}{postfix}-{clip_modelname.replace("/", ".")}-{self.SUPPORT_MEMORY_SIZE}.h5'
        self.H5PY_FILE_PATH = os.path.join(self.__IM2TXT_MEMORY_PATH, filename)
        self.H5PY_EMBEDDINGS_DATASET_NAME = '{}-embeddings'.format(dataset_name)
        self.H5PY_TEXT_DATASET_NAME = '{}-text'.format(dataset_name)

        embs_dataset, text_dataset = self._load_support_memory()

        if text_dataset is None:
            if verbose: 
                model_type = "RegionCLIP" if regionclip_config is not None else ("DenseClip" if denseclip_config is not None else ("OpenCLIP" if use_open_clip else "CLIP"))
                print(f"[+] Going to build support memory for the given data type: {type} using {model_type} [+]")
            embs_dataset, text_dataset = self._build_support_memory(batch_size)
            if verbose: print(f"[+] Done [+]")
        
        if self.type != ProjectionType.ONLINE_TEXTS:
            embs_dataset, text_dataset = self._load_support_memory()
        
        print(f"[-] loaded memory from {os.path.abspath( self.H5PY_FILE_PATH )} [-]")
        if regionclip_config is not None:
            print(f"[-] Using RegionCLIP text embeddings from checkpoint: {regionclip_config.get('checkpoint', 'Unknown')} [-]")
        elif denseclip_config is not None:
            print(f"[-] Using DenseClip text embeddings from config: {denseclip_config} [-]")

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
        
        if image_embedding.dtype != float:
            #print(f"[-] image_embedding.dtype is {image_embedding.dtype}, converting it to float [-]")
            image_embedding = image_embedding.float()
        
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
        
        # Try to resolve the HDF5 file path with HuggingFace Hub fallback
        resolved_h5py_path = self._resolve_memory_bank_path()
        if resolved_h5py_path is None:
            return None, None

        with h5py.File(resolved_h5py_path, 'r') as hf:

            if self.H5PY_EMBEDDINGS_DATASET_NAME in hf:
                embeddings_dataset = hf[self.H5PY_EMBEDDINGS_DATASET_NAME][:]
                text_dataset = hf[self.H5PY_TEXT_DATASET_NAME][:]
            else:
                embeddings_dataset = None
                text_dataset = None
        if 'DINO.txt' in self.clip_modelname:
            embeddings_dataset = embeddings_dataset[:, 1024:] # Get patch-aligned text embeddings
        return embeddings_dataset, text_dataset

    def _resolve_memory_bank_path(self) -> str:
        """
        Resolve the memory bank HDF5 file path with HuggingFace Hub fallback.
        
        Returns:
            Path to the memory bank file (local or cached from HF Hub) or None if not found
        """
        # If local file exists, use it
        if os.path.exists(self.H5PY_FILE_PATH):
            print(f"[-] Using local memory bank: {self.H5PY_FILE_PATH} [-]")
            return self.H5PY_FILE_PATH
        
        # Try HuggingFace Hub fallback if available and configured
        if get_model_path_with_hf_fallback is not None and self.memory_bank_hf_repo_id is not None:
            try:
                # Extract filename from the full path for HF Hub lookup
                h5py_filename = os.path.basename(self.H5PY_FILE_PATH)
                
                print(f"[-] Local memory bank not found, trying HuggingFace Hub: {self.memory_bank_hf_repo_id}/{h5py_filename} [-]")
                
                resolved_path = get_model_path_with_hf_fallback(
                    local_path=self.H5PY_FILE_PATH,
                    hf_repo_id=self.memory_bank_hf_repo_id,
                    filename=h5py_filename
                )
                
                print(f"[-] Successfully resolved memory bank from HF Hub: {resolved_path} [-]")
                return resolved_path
                
            except Exception as e:
                print(f"[-] Failed to download memory bank from HuggingFace Hub: {e} [-]")
        
        # If no HF Hub configuration or download failed
        if self.memory_bank_hf_repo_id is None:
            print(f"[-] _load_support_memory: the path '{self.H5PY_FILE_PATH}' does not exist and no HF repository configured [-]")
        
        return None

        
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
        
        if self.use_open_clip:
            print("[-] loading open_clip model [-]")
            assert self.clip_modelname is not None, "clip_modelname must be provided when using open_clip"
            from open_clip import create_model_and_transforms, tokenize
            self.clip_model, preprocess_train, preprocess_val = create_model_and_transforms(self.clip_modelname, pretrained="laion2b_s32b_b79k", device=self.device)
            self.preprocess = preprocess_train
            self.tokenizer = tokenize
            return
        
        if self.regionclip_config is not None:
            print("[-] loading RegionCLIP model [-]")
            from src.regionclip.loader import load_regionclip_from_checkpoint
            from src.regionclip.datasets.clip_prompt_utils import tokenize as regionclip_tokenize
            
            regionclip_checkpoint = self.regionclip_config.get('checkpoint', None)
            if regionclip_checkpoint is None:
                raise ValueError("RegionCLIP checkpoint not specified in the configuration")
            regionclip_config_name = self.regionclip_config.get('config_name', None)
            
            print(f"[-] Loading RegionCLIP from checkpoint: {regionclip_checkpoint} [-]")
            if regionclip_config_name:
                print(f"[-] Using RegionCLIP config: {regionclip_config_name} [-]")
            
            self.clip_model = load_regionclip_from_checkpoint(
                regionclip_checkpoint, 
                device=self.device, 
                config=regionclip_config_name
            )
            self.tokenizer = regionclip_tokenize
            self.preprocess = None  # RegionCLIP doesn't need preprocessing for text encoding
            
            # Test RegionCLIP text encoding to ensure it works
            try:
                test_text = ["A test sentence for RegionCLIP"]
                test_tokens = self.tokenizer(test_text)
                test_features = self.clip_model.encode_text(test_tokens.to(self.device))
                print(f"[-] RegionCLIP text encoding test successful. Output shape: {test_features.shape} [-]")
            except Exception as e:
                print(f"[!] Warning: RegionCLIP text encoding test failed: {e} [!]")
                raise e
            
            return

        if self.denseclip_config is not None:
            print("[-] loading DenseClip model [-]")
            from src.denseclip.loader import load_denseclip, DenseCLIP_tokenize
            
            print(f"[-] Loading DenseClip from config: {self.denseclip_config} [-]")
            
            # Load DenseClip model
            self.clip_model = load_denseclip(
                config_name=self.denseclip_config,
                device=self.device                
            )
            
            # DenseClip should have encode_text method and a tokenizer
            # We need to check if DenseClip has a tokenizer method
            if DenseCLIP_tokenize is not None:
                self.tokenizer = DenseCLIP_tokenize
            else:
                # Fallback to CLIP tokenizer if DenseClip doesn't provide one
                import clip
                self.tokenizer = clip.tokenize
                print("[!] Warning: DenseClip model doesn't have tokenizer, using CLIP tokenizer [!]")
            
            self.preprocess = None  # DenseClip doesn't need preprocessing for text encoding
            
            # Test DenseClip text encoding to ensure it works
            try:
                test_text = ["A test sentence for DenseClip"]
                test_tokens = self.tokenizer(test_text)
                if hasattr(test_tokens, 'to'):
                    test_tokens = test_tokens.to(self.device)
                test_features = self.clip_model.encode_text(test_tokens)
                print(f"[-] DenseClip text encoding test successful. Output shape: {test_features.shape} [-]")
            except Exception as e:
                print(f"[!] Warning: DenseClip text encoding test failed: {e} [!]")
                raise e
            
            return

        import clip
        if self.clip_modelname is None:
            clip_model_name = "ViT-B/16" if self.use_talk2dino else "ViT-B/32"
        else:
            clip_model_name = self.clip_modelname
        if 'DINO.txt' not in clip_model_name:
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
        else:
            self.clip_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l').to(self.device)
            self.tokenizer = get_tokenizer().tokenize
