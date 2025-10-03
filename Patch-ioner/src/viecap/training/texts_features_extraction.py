import os
import clip
import pickle
import torch
from tqdm import tqdm

from talk2dino.talk2dino import ProjectionLayer

@torch.no_grad()
def main(device: str, clip_type: str, inpath: str, outpath: str, talk2dino_weights_path: str = None, talk2dino_config: str = None):

    device = device
    encoder, _ = clip.load(clip_type, device)
    encoder.eval()

    if talk2dino_weights_path is not None:
        # loading Talk2DINO
        print(f"Loading Talk2DINO weights from {talk2dino_weights_path}")
        talk2dino = ProjectionLayer.from_config(talk2dino_config)
        talk2dino.load_state_dict(torch.load(talk2dino_weights_path, device))
        talk2dino.to(device)
        talk2dino.eval()

    with open(inpath, 'rb') as infile:
        captions_with_entities = pickle.load(infile) # [[[entity1, entity2, ...], caption], ...]

    for idx in tqdm(range(len(captions_with_entities)), dynamic_ncols=True):
        caption = captions_with_entities[idx][1]
        tokens = clip.tokenize(caption, truncate = True).to(device)
        embeddings = encoder.encode_text(tokens).squeeze(dim = 0)
        if talk2dino_weights_path is not None:
            embeddings = talk2dino.project_clip_txt(embeddings)
        embeddings = embeddings.to('cpu')
        captions_with_entities[idx].append(embeddings)
    
    with open(outpath, 'wb') as outfile:
        pickle.dump(captions_with_entities, outfile)
    
    return captions_with_entities

if __name__ == '__main__':
    
    idx = 0 # change here! 0 -> coco training data, 1 -> flickr30k training data
    device = 'cuda:5'
    clip_type = 'ViT-B/16' # change here for different clip backbone (ViT-B/32, RN50x4)
    clip_name = clip_type.replace('/', '')

    talk2dino_weights_path = 'talk2dino/weights/vitb_mlp_infonce.pth' # None to disable talk2dino
    talk2dino_config = 'talk2dino/configs/vitb_mlp_infonce.yaml'

    if talk2dino_weights_path is not None:
        suffix = '_t2d_'
    else:
        suffix = ''

    inpath = [
    '/raid/datasets/viecap_files/annotations/coco/coco_with_entities.pickle',
    '/raid/datasets/viecap_files/annotations/flickr30k/flickr30k_with_entities.pickle']
    outpath = [
    f'/raid/datasets/viecap_files/annotations/coco/coco_texts_features_{clip_name}{suffix}.pickle',
    f'/raid/datasets/viecap_files/annotations/flickr30k/flickr30k_texts_features_{clip_name}{suffix}.pickle']

    if os.path.exists(outpath[idx]):
        with open(outpath[idx], 'rb') as infile:
            captions_with_features = pickle.load(infile)
    else:
        captions_with_features = main(device, clip_type, inpath[idx], outpath[idx], talk2dino_weights_path, talk2dino_config)

    import random
    print(f'datasets for {inpath[idx]}')
    print(f'The length of datasets: {len(captions_with_features)}')
    caption_with_features = captions_with_features[random.randint(0, len(captions_with_features) - 1)]
    detected_entities, caption, caption_features = caption_with_features
    print(detected_entities, caption, caption_features.size(), caption_features.dtype)

    encoder, _ = clip.load(clip_type, device)
    encoder.eval()
    if talk2dino_weights_path is not None:
        talk2dino = ProjectionLayer.from_config(talk2dino_config)
        talk2dino.load_state_dict(torch.load(talk2dino_weights_path, device))
        talk2dino.to(device)
        talk2dino.eval()

        
    with torch.no_grad():
        if talk2dino_weights_path is not None:
            embeddings = encoder.encode_text(clip.tokenize(caption, truncate = True).to(device)).squeeze(dim = 0)
            embeddings = talk2dino.project_clip_txt(embeddings)
            embeddings = embeddings.to('cpu')
        else:
            embeddings = encoder.encode_text(clip.tokenize(caption, truncate = True).to(device)).squeeze(dim = 0).to('cpu')
    print(abs(embeddings - caption_features).mean())