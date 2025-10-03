
import torch
import torch.nn as nn
import yaml

class ProjectionLayer(nn.Module):
    """
    Creates a projection layer on top of the CLIP-text encoder.
    The forward method calculate the similarity between the DINO CLS token and the projected CLIP textual CLS token. 
    """
    def __init__(self, act=nn.Tanh(), hidden_layer=False, cosine=True, dino_embed_dim=1024, clip_embed_dim=512, num_attn_head=16, weight_attn_heads=None,
                 alignment_strategy='max_score', alpha=0.6, keep_cls=False, keep_end_seq=False):
        # mlp_dims list of mlp dimensions
        super().__init__()
        self.num_attn_head = num_attn_head      
        
        self.linear_layer = nn.Linear(clip_embed_dim, dino_embed_dim)
        if hidden_layer:
            hidden_layer = 1 if hidden_layer is True else hidden_layer # ensuring compatibility with old code
            # self.linear_layer2 = nn.Linear(dino_embed_dim, dino_embed_dim) 
            self.hidden_layers = nn.ModuleList([nn.Linear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_layer)])
        self.act = act
        self.cosine = cosine
        
        self.weight_attn_heads = weight_attn_heads
        if weight_attn_heads == 'static':
            self.attn_weights = nn.Parameter(torch.rand(self.num_attn_head))
        elif weight_attn_heads == 'conditioned':
            self.weight_layer1 = nn.Linear(dino_embed_dim, dino_embed_dim)
            self.weight_layer2 = nn.Linear(dino_embed_dim, self.num_attn_head)
            
        self.alignment_strategy = alignment_strategy # relevant only if we use disentangled_self_attn
        self.keep_cls = keep_cls # relevant only if we use clip_txt_tokens_out
        self.keep_end_seq = keep_end_seq # relevant only if we use clip_txt_tokens_out
        self.alpha = alpha
    
    @classmethod
    def from_config(cls, config):
        if type(config) is str:
            # if the configuration is a string, we treat it as a file path
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['model']
        
        # loading the activation function
        act = config.get('act', None)
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act is not None:
            raise Exception("Unknown activation function")
        
        model = cls(
            act=act,
            hidden_layer=config.get('hidden_layer', False),
            cosine=config.get('cosine', True),
            dino_embed_dim=config.get('dino_embed_dim', 1024),
            num_attn_head=config.get('num_attn_head', 16),
            clip_embed_dim=config.get('clip_embed_dim', 512),
            weight_attn_heads=config.get('weight_attn_heads', None),
            alignment_strategy=config.get('alignment_strategy', 'max_score'),
            alpha=config.get('alpha', 0.6),
            keep_cls=config.get('keep_cls', None),
            keep_end_seq=config.get('keep_end_seq', None),
        )
        if config.get('starting_checkpoint', None) is not None:
            model.load_state_dict(torch.load(config['starting_checkpoint'], 'cpu'))
        
        return model
    
    def project_clip_txt(self, textual_embedding):
        textual_embedding = textual_embedding.float()
        x = self.linear_layer(textual_embedding)
        
        if hasattr(self, 'hidden_layers'):
            for hidden_layer in self.hidden_layers:
                if self.act:
                    x = self.act(x)
                x = hidden_layer(x)
            
        return x
    def load_state_dict(self, state_dict, strict=True):
        # compatibility with old code
        if 'linear_layer2.weight' in state_dict:
            state_dict['hidden_layers.0.weight'] = state_dict.pop('linear_layer2.weight')
            state_dict['hidden_layers.0.bias'] = state_dict.pop('linear_layer2.bias')
        # Call the parent class's load_state_dict with the modified state_dict
        super(ProjectionLayer, self).load_state_dict(state_dict, strict)
    
    def set_alignment_strategy(self, alignment_strategy):
        self.alignment_strategy = alignment_strategy
        return
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())  