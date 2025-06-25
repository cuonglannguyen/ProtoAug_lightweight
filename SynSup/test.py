import torch
import torch.nn as nn
import torch.nn.functional as F
import loralib as lora
from diffusers import UNet2DConditionModel
from transformers import CLIPTextModel
from typing import Optional, Union

def lora_replace_attention_layers(
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
    
    # For UNet, we need to replace the attention modules in several places
    if not is_clip:
        # Replace in down blocks
        for down_block in model.down_blocks:
            if hasattr(down_block, 'attentions'):
                for attn in down_block.attentions:
                    # Replace the attention layers
                    attn.to_q = lora.Linear(
                        attn.q.in_features,
                        attn.q.out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=attn.to_q.bias is not None
                    )
                    attn.to_k = lora.Linear(
                        attn.to_k.in_features,
                        attn.to_k.out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=attn.to_k.bias is not None
                    )
                    attn.to_v = lora.Linear(
                        attn.to_v.in_features,
                        attn.to_v.out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=attn.to_v.bias is not None
                    )
                    
                    # Copy original parameters
                    if hasattr(attn.to_q, 'bias') and attn.to_q.bias is not None:
                        attn.to_q.bias.data = attn.to_q.bias.data
                    if hasattr(attn.to_k, 'bias') and attn.to_k.bias is not None:
                        attn.to_k.bias.data = attn.to_k.bias.data
                    if hasattr(attn.to_v, 'bias') and attn.to_v.bias is not None:
                        attn.to_v.bias.data = attn.to_v.bias.data

        # Replace in up blocks
        for up_block in model.up_blocks:
            if hasattr(up_block, 'attentions'):
                for attn in up_block.attentions:
                    # Replace the attention layers
                    attn.to_q = lora.Linear(
                        attn.to_q.in_features,
                        attn.to_q.out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=attn.to_q.bias is not None
                    )
                    attn.to_k = lora.Linear(
                        attn.to_k.in_features,
                        attn.to_k.out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=attn.to_k.bias is not None
                    )
                    attn.to_v = lora.Linear(
                        attn.to_v.in_features,
                        attn.to_v.out_features,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        bias=attn.to_v.bias is not None
                    )
                    
                    # Copy original parameters
                    if hasattr(attn.to_q, 'bias') and attn.to_q.bias is not None:
                        attn.to_q.bias.data = attn.to_q.bias.data
                    if hasattr(attn.to_k, 'bias') and attn.to_k.bias is not None:
                        attn.to_k.bias.data = attn.to_k.bias.data
                    if hasattr(attn.to_v, 'bias') and attn.to_v.bias is not None:
                        attn.to_v.bias.data = attn.to_v.bias.data

        # Replace in mid block
        if hasattr(model.mid_block, 'attentions'):
            for attn in model.mid_block.attentions:
                # Replace the attention layers
                attn.to_q = lora.Linear(
                    attn.to_q.in_features,
                    attn.to_q.out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=attn.to_q.bias is not None
                )
                attn.to_k = lora.Linear(
                    attn.to_k.in_features,
                    attn.to_k.out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=attn.to_k.bias is not None
                )
                attn.to_v = lora.Linear(
                    attn.to_v.in_features,
                    attn.to_v.out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=attn.to_v.bias is not None
                )
                
                # Copy original parameters
                if hasattr(attn.to_q, 'bias') and attn.to_q.bias is not None:
                    attn.to_q.bias.data = attn.to_q.bias.data
                if hasattr(attn.to_k, 'bias') and attn.to_k.bias is not None:
                    attn.to_k.bias.data = attn.to_k.bias.data
                if hasattr(attn.to_v, 'bias') and attn.to_v.bias is not None:
                    attn.to_v.bias.data = attn.to_v.bias.data
    
    else:
        # For CLIP text encoder
        for name, module in model.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                # Replace with LoRA version
                in_features = module.embed_dim
                out_features = module.embed_dim * 3  # For q, k, v combined
                
                new_module = lora.MergedLinear(
                    in_features,
                    out_features,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    enable_lora=[True, False, True],  # Enable for q and v
                    bias=module.in_proj_bias is not None,
                )
                
                # Copy original parameters
                new_module.weight.data = module.in_proj_weight.data
                if module.in_proj_bias is not None:
                    new_module.bias.data = module.in_proj_bias.data
                
                # Replace the module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent_module = model.get_submodule(parent_name)
                setattr(parent_module, child_name, new_module)
    
    return model

def prepare_model_for_training(model):
    """
    Prepare the model for training by freezing all parameters except LoRA
    """
    # First freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze LoRA parameters
    for name, module in model.named_modules():
        if isinstance(module, (lora.Linear, lora.MergedLinear)):
            if hasattr(module, 'lora_A'):
                module.lora_A.requires_grad = True
            if hasattr(module, 'lora_B'):
                module.lora_B.requires_grad = True
    
    # Print trainable parameters
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print("\nTrainable parameters:")
    for param in trainable_params:
        print(f"  {param}")
    
    return model

# Example usage:
if __name__ == "__main__":
    # Load model
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/stable-diffusion-2-1",
        subfolder="unet",
    )
    
    # Add LoRA layers
    unet = lora_replace_attention_layers(
        unet,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    
    # Prepare for training
    unet = prepare_model_for_training(unet)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")