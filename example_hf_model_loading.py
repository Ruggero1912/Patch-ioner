from patchioner import Patchioner
# convert the list to a batch tensor
import torchvision.transforms as T
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

config_name = "Ruggero1912/Patch-ioner_talk2dino_viecap_COCO_Captions"
# "./configs_public/talk2dino_decap_COCO_Captions.yaml"
model = Patchioner.from_config(config_name, device=device)

# caption one image 
img_dir = "/raid/homes/giacomo.pacini/decap-dino/decap/test-images"

import sys, os
from PIL import Image

# take all images in img_dir
image_files = [Image.open(os.path.join(img_dir, f)).convert('RGB') for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]




batch_imgs = torch.stack([model.image_transforms(img) for img in image_files]).to(device)

outs = model.forward(batch_imgs, get_cls_capt=True)

print(outs.keys())

print(outs['cls_capt'])