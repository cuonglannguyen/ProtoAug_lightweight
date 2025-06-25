import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel
from typing import Optional, Union

class LoRAMultiHeadAttention(nn.Module):
    """
    Implementation of MultiHeadAttention with LoRA for UNet2DConditionModel and CLIPTextModel.
    """

    def __init__(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        q_lora: bool = True,
        k_lora: bool = False,
        v_lora: bool = True,
        is_clip: bool = False,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        
        self.upcast_attention = upcast_attention
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.is_clip = is_clip
        
        if is_clip:
            # CLIP uses a single matrix for QKV projections
            self.qkv = lora.MergedLinear(
                query_dim,
                3 * inner_dim,
                bias=bias,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=[q_lora, k_lora, v_lora],
            )
        else:
            # UNet uses separate matrices for Q, K, V
            self.to_q = lora.Linear(
                query_dim,
                inner_dim,
                bias=bias,
                r=lora_r if q_lora else 0,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            
            self.to_k = lora.Linear(
                cross_attention_dim,
                inner_dim,
                bias=bias,
                r=lora_r if k_lora else 0,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            
            self.to_v = lora.Linear(
                cross_attention_dim,
                inner_dim,
                bias=bias,
                r=lora_r if v_lora else 0,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        
        self.to_out = nn.ModuleList([
            nn.Linear(inner_dim, query_dim, bias=bias),
            nn.Dropout(dropout)
        ])

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        
        if self.is_clip:
            # CLIP-style attention
            qkv = self.qkv(hidden_states)
            qkv = qkv.unflatten(-1, (3, -1)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
            query, key, value = qkv[0], qkv[1], qkv[2]
            
            # Handle causal attention mask for CLIP
            attention_mask = causal_attention_mask if causal_attention_mask is not None else attention_mask
        else:
            # UNet-style attention
            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            query = self.to_q(hidden_states)
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

        query = self.reshape_heads_to_batch_dim(query)
        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=causal_attention_mask is not None
        )

        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        
        return hidden_states

    def set_parameters(self, original_module):
        """Copy parameters from original attention module"""
        if self.is_clip:
            # For CLIP text encoder
            if hasattr(original_module, 'in_proj_weight'):
                self.qkv.weight.data = original_module.in_proj_weight.data
                if hasattr(original_module, 'in_proj_bias') and original_module.in_proj_bias is not None:
                    self.qkv.bias.data = original_module.in_proj_bias.data
        else:
            # For UNet
            self.to_q.weight.data = original_module.to_q.weight.data
            self.to_k.weight.data = original_module.to_k.weight.data
            self.to_v.weight.data = original_module.to_v.weight.data
            
            if hasattr(original_module.to_q, 'bias') and original_module.to_q.bias is not None:
                self.to_q.bias.data = original_module.to_q.bias.data
            if hasattr(original_module.to_k, 'bias') and original_module.to_k.bias is not None:
                self.to_k.bias.data = original_module.to_k.bias.data
            if hasattr(original_module.to_v, 'bias') and original_module.to_v.bias is not None:
                self.to_v.bias.data = original_module.to_v.bias.data
        
        if hasattr(original_module, 'out_proj'):
            # CLIP style
            self.to_out[0].weight.data = original_module.out_proj.weight.data
            if original_module.out_proj.bias is not None:
                self.to_out[0].bias.data = original_module.out_proj.bias.data
        else:
            # UNet style
            self.to_out[0].weight.data = original_module.to_out[0].weight.data
            if original_module.to_out[0].bias is not None:
                self.to_out[0].bias.data = original_module.to_out[0].bias.data

def lora_replace_sd_attention_layers(
    model: Union[UNet2DConditionModel, CLIPTextModel],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_replace_module: Optional[str] = None,
):
    """
    Replace attention layers with LoRA attention layers.
    
    Args:
        model: Either UNet2DConditionModel or CLIPTextModel
        lora_r (int): LoRA rank
        lora_alpha (int): LoRA alpha scaling factor
        lora_dropout (float): Dropout probability for LoRA layers
        target_replace_module (str, optional): Name of the module to replace. 
            If None, automatically detects based on model type.
    """
    is_clip = isinstance(model, CLIPTextModel)
    if target_replace_module is None:
        target_replace_module = "MultiheadAttention" if is_clip else "CrossAttention"
    
    def walk_and_replace(module, name=""):
        for child_name, child in module.named_children():
            curr_name = f"{name}.{child_name}" if name else child_name
            
            if child.__class__.__name__ == target_replace_module:
                if is_clip:
                    lora_attn = LoRAMultiHeadAttention(
                        query_dim=child.embed_dim,
                        heads=child.num_heads,
                        dim_head=child.head_dim,
                        dropout=child.dropout,
                        bias=True,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        is_clip=True
                    )
                else:
                    lora_attn = LoRAMultiHeadAttention(
                        query_dim=child.to_q.in_features,
                        cross_attention_dim=child.to_k.in_features,
                        heads=child.heads,
                        dim_head=child.to_q.out_features // child.heads,
                        dropout=0.0,
                        bias=child.to_q.bias is not None,
                        lora_r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        upcast_attention=getattr(child, 'upcast_attention', False),
                        is_clip=False
                    )
                lora_attn.set_parameters(child)
                setattr(module, child_name, lora_attn)
            else:
                walk_and_replace(child, curr_name)
    
    walk_and_replace(model)
    return model

# Example usage:
if __name__ == "__main__":
    from diffusers import UNet2DConditionModel
    from transformers import CLIPTextModel
    
    # Load models
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
    )
    
    text_encoder = CLIPTextModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="text_encoder",
    )
    
    # Replace attention layers with LoRA
    unet = lora_replace_sd_attention_layers(
        unet,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    text_encoder = lora_replace_sd_attention_layers(
        text_encoder,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )