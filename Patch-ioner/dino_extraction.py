import argparse
import torch
import torchvision.transforms as T

from PIL import Image

feats = {}
def get_self_attention(module, input, output):
    feats['self_attn'] = output

def get_layer_n_output(module, input, output):
    feats['intermediate_output'] = output

def transform_to_standard_dino_out(x, model):
    x_norm = model.norm(x)
    return {
        "x_norm_clstoken": x_norm[:, 0],
        "x_norm_regtokens": x_norm[:, 1 : 4 + 1],
        "x_norm_patchtokens": x_norm[:, 4 + 1 :],
        "x_prenorm": x,
        # "masks": masks,
    }   

def process_self_attention(output, batch_size, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False):
    qkv = output.reshape(batch_size, num_tokens, 3, num_attn_heads, embed_dim // num_attn_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)
    self_attn_maps = attn[:, : , 0, num_global_tokens:]
    self_attn = self_attn_maps.mean(dim=1)
    self_attn = self_attn.softmax(dim=-1)
    if ret_self_attn_maps:
        return self_attn, self_attn_maps
    else:
        return self_attn
    

def run_dinov2_extraction(model_name, resize_dim=518, crop_dim=518, img_path='cat.jpeg'):
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    
    num_global_tokens = 1 if "reg" not in model_name else 5
    num_patch_tokens = crop_dim // 14 * crop_dim // 14
    num_tokens = num_global_tokens + num_patch_tokens
    if 'vitl' in model_name or 'vit_large' in model_name or 'ViT-L' in model_name:
        embed_dim = 1024
    elif 'vitb' in model_name or 'vit_base' in model_name or 'ViT-B' in model_name:
        embed_dim = 768
    elif 'vits' in model_name or 'vit_small' in model_name:
        embed_dim = 384
    else:
        raise Exception("Unknown ViT model")
    
    num_attn_heads = 16 if not 'vits' in model_name else 6
    scale = 0.125
    
    # loading the model
    if 'dinov2' in model_name:
        model_family = 'facebookresearch/dinov2'
        model = torch.hub.load(model_family, model_name)
        image_transforms = T.Compose([
            T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(crop_dim),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
    model.eval()
    model.to(device)
    model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
    model.blocks[-1].register_forward_hook(get_layer_n_output)
    
    pil_img = Image.open(img_path)
        
    if pil_img.mode != 'RGB':
        pil_img = pil_img.convert('RGB')

    batch_imgs = image_transforms(pil_img).unsqueeze(0).to(device)
            
    with torch.no_grad():
        outs = model(batch_imgs, is_training=True)
        outs_layer_n = transform_to_standard_dino_out(feats['intermediate_output'], model)
        
        self_attn = process_self_attention(feats['self_attn'], 1, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False)
        avg_self_attn_token = (self_attn.unsqueeze(-1) * outs['x_norm_patchtokens']).mean(dim=1)
    
    print(avg_self_attn_token)
    print(avg_self_attn_token.shape)
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="dinov2_vitb14_reg", help="Model configuration to extract features from")
    parser.add_argument('--resize_dim', type=int, default=518, help="Resize dimension")
    parser.add_argument('--crop_dim', type=int, default=518, help="Crop dimension")
    args = parser.parse_args()
    
    run_dinov2_extraction(args.model, args.resize_dim, args.crop_dim)
if __name__ == '__main__':
    main()