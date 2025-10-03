import os
import clip
import torch
import pickle
from tqdm import tqdm
from typing import List
from load_annotations import load_entities_text

from talk2dino.talk2dino import ProjectionLayer

@torch.no_grad()
def generate_ensemble_prompt_embeddings(
    device: str,
    clip_type: str,
    entities: List[str],
    prompt_templates: List[str],
    outpath: str,
    talk2dino_weights_path: str = None,
    talk2dino_config : str = None
):
    if os.path.exists(outpath):
        with open(outpath, 'rb') as infile:
            embeddings = pickle.load(infile)
            return embeddings

    model, _ = clip.load(clip_type, device)
    model.eval()

    if talk2dino_weights_path is not None:
        # loading Talk2DINO
        print(f"Loading Talk2DINO weights from {talk2dino_weights_path}")
        talk2dino = ProjectionLayer.from_config(talk2dino_config)
        talk2dino.load_state_dict(torch.load(talk2dino_weights_path, device))
        talk2dino.to(device)
        talk2dino.eval()

    embeddings = []
    for entity in tqdm(entities):
        texts = [template.format(entity) for template in prompt_templates] # ['a picture of dog', 'photo of a dog', ...]
        tokens = clip.tokenize(texts).to(device)               # (len_of_template, 77)
        class_embeddings = model.encode_text(tokens) # (len_of_templates, clip_hidden_size)
        if talk2dino_weights_path is not None:
            class_embeddings = talk2dino.project_clip_txt(class_embeddings) # (len_of_templates, dino_embed_dim)
        class_embeddings = class_embeddings.to('cpu')
        class_embeddings /= class_embeddings.norm(dim = -1, keepdim = True) # (len_of_templates, clip_hidden_size)
        class_embedding = class_embeddings.mean(dim = 0)       # (clip_hidden_size, ) 
        class_embedding /= class_embedding.norm()              # (clip_hidden_size, ) 
        embeddings.append(class_embedding)                     # [(clip_hidden_size, ), (clip_hidden_size, ), ...]
    embeddings = torch.stack(embeddings, dim = 0).to('cpu')
   
    with open(outpath, 'wb') as outfile:
        pickle.dump(embeddings, outfile)
    return embeddings

if __name__ == '__main__':

    # prompts from CLIP
    prompt_templates = [
        'itap of a {}.',
        'a bad photo of the {}.',
        'a origami {}.',
        'a photo of the large {}.',
        'a {} in a video game.',
        'art of the {}.',
        'a photo of the small {}.'
    ]

    #entities = load_entities_text('vinvl_vgoi_entities', './annotations/vocabulary/vgcocooiobjects_v1_class2ind.json')
    entities = load_entities_text('coco_entities', '/raid/datasets/viecap_files/annotations/vocabulary/coco_categories.json')
    # entities = load_entities_text('vinvl_vg_entities', './annotations/vocabulary/VG-SGG-dicts-vgoi6-clipped.json')
    # entities = load_entities_text('visual_genome_entities', './annotations/vocabulary/all_objects_attributes_relationships.pickle', all_entities = False)
    # entities = load_entities_text('open_image_entities', './annotations/vocabulary/oidv7-class-descriptions-boxable.csv')

    device = 'cuda:5'
    clip_type = 'ViT-B/16'

    talk2dino_weights_path = 'talk2dino/weights/vitb_mlp_infonce.pth' #None
    talk2dino_config = 'talk2dino/configs/vitb_mlp_infonce.yaml'

    if talk2dino_weights_path is not None:
        suffix = '_t2d_'
    else:
        suffix = ''

    clip_name = clip_type.replace('/', '')
    #outpath = f'./annotations/vocabulary/vgoi_embeddings_{clip_name}_with_ensemble.pickle'
    outpath = f'/raid/datasets/viecap_files/annotations/vocabulary/coco_embeddings_{clip_name}{suffix}_with_ensemble.pickle'
    # outpath = f'./annotations/vocabulary/vg_embeddings_{clip_name}_with_ensemble.pickle'
    # outpath = f'./annotations/vocabulary/visual_genome_embedding_{clip_name}_with_ensemble.pickle'
    # outpath = f'./annotations/vocabulary/open_image_embeddings_{clip_name}_with_ensemble.pickle'

    

    embeddings = generate_ensemble_prompt_embeddings(device, clip_type, entities, prompt_templates, outpath, talk2dino_weights_path, talk2dino_config)

    print(entities[:10], len(entities))
    print(embeddings.size(), embeddings.dtype)