from torch import nn
import torch
from .open_clip_proxy import create_model, tokenizer
from torchvision import transforms as T

class ProxyCLIP(nn.Module):
    def __init__(self, clip_type, model_type, vfm_model, device=torch.device('cuda'), beta=1.2, gamma=3.0, slide_crop=336):

        super().__init__()

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.vfm_model = vfm_model

        if vfm_model == 'dino':
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        elif vfm_model == 'dinov2':
            # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, img):
        if type(img) == list:
            img = img[0]

        clip_token_size = img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[1]

        # imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        # imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = img

        imgs_norm = imgs_norm.half()
        if self.vfm_model == 'dino':
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]

            nb_im = feat.shape[0]  # Batch size

            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size
            ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)

        elif self.vfm_model == 'dinov2':
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            ex_feats = self.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]

        else:
            I, J = clip_token_size
            ex_feats = None

        image_features = self.clip.encode_image(img.half(),
                                               external_feats=ex_feats,
                                               beta=self.beta,
                                               gamma=self.gamma)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        
        
        return {
            'x_norm_patchtokens': image_features.float()
        }