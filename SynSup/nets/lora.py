"""
Code from https://github.com/vturrisi/disef/blob/main/fine-tune/src/adaptation/lora.py
"""

import loralib as lora
import torch.nn as nn
import torch.nn.functional as F


class LoRAMultiHeadAttention(nn.Module):
    """
    Re-implementation of MultiHeadAttention with added LoRA.
    Since CLIP's implementation uses the MultiHeadAttention from Pytorch, modifying it with LoRA
    is not possible.
    This doesn't support different dimensions for q, k and v.
    """

    def __init__(
        self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        embed_dim: int = 1024,
        num_heads: int = 16,
        dropout: float = 0.0,
        bias: bool = True,
        kdim: int = None,
        vdim: int = None,
        batch_first: bool = False,
        q_lora: bool = True,
        k_lora: bool = False,
        v_lora: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            assert NotImplementedError
        else:
            self.qkv = lora.MergedLinear(
                embed_dim,
                3 * embed_dim,
                bias=bias,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                enable_lora=[q_lora, k_lora, v_lora],
            )

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    ):
        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        E = query.size(-1)
        qkv = self.qkv(query)
        qkv = qkv.unflatten(-1, (3, E)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape},"
                        f" but should be {correct_2d_size}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape},"
                        f" but should be {correct_3d_size}."
                    )
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.0

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None

    def set_parameters(self, torch_tgt_module):
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim
        self.qkv.weight.data = torch_tgt_module.in_proj_weight.data
        self.qkv.bias.data = torch_tgt_module.in_proj_bias.data
        self.proj.weight.data = torch_tgt_module.out_proj.weight.data
        self.proj.bias.data = torch_tgt_module.out_proj.bias.data


def lora_replace_attention_layers(
    transformer: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    start_block: int = 0,
):
    """
    Utility function to replace attention layers of a CLIP transformer model
    with LoRAMultiHeadAttention. It expects a pre-defined structure (following OpenAI's CLIP).

    Args:
        transformer (nn.Module): transformer to replace attention layers.
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
        start_block (int): first block to start replacing the attention layers.
    """

    for block in transformer.resblocks[start_block:]:
        attn = block.attn
        embed_dim = attn.embed_dim
        num_heads = attn.num_heads
        dropout = attn.dropout
        lora_attn = LoRAMultiHeadAttention(
            embed_dim=embed_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_heads=num_heads,
            dropout=dropout,
        )
        lora_attn.set_parameters(attn)
        block.attn = lora_attn

    return transformer

def lora_replace_attention_layers_clip(
    transformer: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    start_block: int = 0,
):
    """
    Replace attention layers in Hugging Face's CLIP encoder with LoRA-enhanced layers.

    Args:
        transformer (nn.Module): The encoder (e.g., `CLIPTextModel.text_model.encoder`).
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
        start_block (int): First block to start replacing the attention layers.
    """
    for idx, layer in enumerate(transformer.layers[start_block:], start=start_block):
        attn = layer.self_attn
        embed_dim = attn.q_proj.weight.size(1)  # Dimension of the query projection input
        num_heads = attn.num_heads
        dropout = attn.dropout

        # Create LoRA-enhanced attention
        lora_attn = LoRAMultiHeadAttention(
            embed_dim=embed_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Copy weights to the LoRA-enhanced module
        lora_attn.qkv.weight.data[:embed_dim] = attn.q_proj.weight.data
        lora_attn.qkv.weight.data[embed_dim : 2 * embed_dim] = attn.k_proj.weight.data
        lora_attn.qkv.weight.data[2 * embed_dim :] = attn.v_proj.weight.data

        if attn.q_proj.bias is not None:
            lora_attn.qkv.bias.data[:embed_dim] = attn.q_proj.bias.data
            lora_attn.qkv.bias.data[embed_dim : 2 * embed_dim] = attn.k_proj.bias.data
            lora_attn.qkv.bias.data[2 * embed_dim :] = attn.v_proj.bias.data

        lora_attn.proj.weight.data = attn.out_proj.weight.data
        if attn.out_proj.bias is not None:
            lora_attn.proj.bias.data = attn.out_proj.bias.data

        # Replace the original attention module with LoRA-enhanced attention
        layer.self_attn = lora_attn

    return transformer
def lora_replace_unet_attention_layers(
    unet: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    start_block: int = 0,
):
    """
    Replace attention layers in Hugging Face's UNet2DConditionModel with LoRA-enhanced layers.

    Args:
        unet (nn.Module): The UNet model (e.g., `UNet2DConditionModel`).
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
        start_block (int): First block to start replacing the attention layers.
    """
    for module in unet.modules():
        if hasattr(module, "transformer_blocks"):
            for block in module.transformer_blocks[start_block:]:
                # Replace the CrossAttention layer
                if hasattr(block, "attn1"):  # Self-Attention
                    block.attn1 = replace_with_lora_attention(
                        block.attn1, lora_r, lora_alpha, lora_dropout
                    )
                if hasattr(block, "attn2"):  # Cross-Attention
                    block.attn2 = replace_with_lora_attention(
                        block.attn2, lora_r, lora_alpha, lora_dropout
                    )

    return unet


def replace_with_lora_attention(
    cross_attention: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
):
    """
    Replace a CrossAttention module with a LoRA-enhanced attention module.

    Args:
        cross_attention (nn.Module): The original CrossAttention module.
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
    """
    embed_dim = cross_attention.to_q.weight.size(1)
    num_heads = cross_attention.heads

    # Create LoRA-enhanced attention
    lora_attn = LoRAMultiHeadAttention(
        embed_dim=embed_dim,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_heads=num_heads,
        dropout=0.0,  # No dropout at the attention level
    )
    print(f"lora_attn.qkv.weight.data shape: {lora_attn.qkv.weight.data.shape}")
    print(f"cross_attention.to_k.weight.data shape: {cross_attention.to_k.weight.data.shape}")
    print(f"embed_dim: {embed_dim}")
    # Map weights from the original CrossAttention to the LoRA-enhanced attention
# Replace LoRA weights for cross-attention
    lora_attn.qkv.weight.data[:embed_dim, :] = cross_attention.to_q.weight.data  # Query

    # Handle Key
    key_start = embed_dim
    key_end = 2 * embed_dim

    # Adjust key shape with projection if needed
    if cross_attention.to_k.weight.shape[1] != embed_dim:
        # Project to match LoRA dimensions
        projected_to_k = cross_attention.to_k.weight.data[:, :embed_dim]
        lora_attn.qkv.weight.data[key_start:key_end, :] = projected_to_k
    else:
        lora_attn.qkv.weight.data[key_start:key_end, :] = cross_attention.to_k.weight.data

    # Value
    lora_attn.qkv.weight.data[key_end:, :] = cross_attention.to_v.weight.data  # Value

    # Output projection
    lora_attn.proj.weight.data = cross_attention.to_out[0].weight.data
    lora_attn.proj.bias.data = cross_attention.to_out[0].bias.data

    return lora_attn

def lora_replace_attention_layers_sd_CLIP(
    transformer: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    start_block: int = 0,
):
    """
    Replace attention layers in Hugging Face's CLIP encoder with LoRA-enhanced layers.

    Args:
        transformer (nn.Module): The encoder (e.g., `CLIPTextModel.text_model.encoder`).
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
        start_block (int): First block to start replacing the attention layers.
    """
    for idx, layer in enumerate(transformer.layers[start_block:], start=start_block):
        attn = layer.self_attn
        embed_dim = attn.q_proj.weight.size(1)  # Dimension of the query projection input
        num_heads = attn.num_heads
        dropout = attn.dropout

        # Create LoRA-enhanced attention
        lora_attn = LoRAMultiHeadAttention(
            embed_dim=embed_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_heads=num_heads,
            dropout=dropout,
        )

        # Copy weights to the LoRA-enhanced module
        lora_attn.qkv.weight.data[:embed_dim] = attn.q_proj.weight.data
        lora_attn.qkv.weight.data[embed_dim : 2 * embed_dim] = attn.k_proj.weight.data
        lora_attn.qkv.weight.data[2 * embed_dim :] = attn.v_proj.weight.data

        if attn.q_proj.bias is not None:
            lora_attn.qkv.bias.data[:embed_dim] = attn.q_proj.bias.data
            lora_attn.qkv.bias.data[embed_dim : 2 * embed_dim] = attn.k_proj.bias.data
            lora_attn.qkv.bias.data[2 * embed_dim :] = attn.v_proj.bias.data

        lora_attn.proj.weight.data = attn.out_proj.weight.data
        if attn.out_proj.bias is not None:
            lora_attn.proj.bias.data = attn.out_proj.bias.data

        # Replace the original attention module with LoRA-enhanced attention
        layer.self_attn = lora_attn

    return transformer
def lora_replace_attention_layers_clip_text_encoder(
    text_encoder: nn.Module,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    start_block: int = 0,
):
    """
    Replace attention layers in Hugging Face's CLIPTextEncoder with LoRAMultiHeadAttention.

    Args:
        text_encoder (nn.Module): The CLIPTextEncoder model.
        lora_r (int): LoRA's rank.
        lora_alpha (int): LoRA's alpha for scaling the output.
        lora_dropout (float): LoRA's dropout rate.
        start_block (int): First block to start replacing the attention layers.
    """
    # Access the encoder's transformer layers
    encoder_layers = text_encoder.transformer.encoder.layers

    # Replace attention layers from the specified starting block
    for layer in encoder_layers[start_block:]:
        attn = layer.self_attn
        embed_dim = attn.embed_dim
        num_heads = attn.num_heads
        dropout = attn.dropout
        lora_attn = LoRAMultiHeadAttention(
            embed_dim=embed_dim,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            num_heads=num_heads,
            dropout=dropout,
        )
        lora_attn.set_parameters(attn)
        layer.self_attn = lora_attn

    return text_encoder
if __name__ == "__main__":
    import clip

    clip_model, _ = clip.load("ViT-B/32", device="cpu")
    lora_replace_attention_layers(clip_model.visual, lora_r=16, lora_alpha=32, lora_dropout=0.1)