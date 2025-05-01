import os
import math
import urllib.request
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# URLs for the original OpenAI CLIP models
_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
}


class LayerNorm(nn.LayerNorm):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, is_cross_attention=False):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x, attn_mask=None):
        # Custom attention mask handling for variable length sequences
        if attn_mask is None and self.attn_mask is not None:
            # Use a portion of the pre-computed mask if available
            seq_len = x.size(0)
            mask_len = min(seq_len, self.attn_mask.size(0))
            attn_mask = self.attn_mask[:mask_len, :mask_len]

        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, inputs):
        if isinstance(inputs, list) and len(inputs) >= 4:
            # Handle prompt-augmented input
            x, prompt_emb, idx, missing_type = inputs[:4]
            prompt_len = prompt_emb.shape[0]

            # Concatenate prompt with x
            xp = torch.cat([prompt_emb, x], dim=0)

            # Self-attention on concatenated input
            attn_out = self.ln_1(xp + self.attention(xp))

            # Split back to prompt and x parts
            prompt_emb, x = attn_out[:prompt_len], attn_out[prompt_len:]

            # Feed-forward on both parts
            prompt_emb = self.ln_2(prompt_emb + self.mlp(prompt_emb))
            x = self.ln_2(x + self.mlp(x))

            return [x, prompt_emb, idx + 1, missing_type]
        else:
            # Standard processing without prompts
            x = inputs
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
            return x


class CustomTransformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None, prompt_length=0, max_position_embeddings=77):
        super().__init__()
        self.width = width
        self.layers = layers
        self.prompt_length = prompt_length
        self.max_position_embeddings = max_position_embeddings

        # Build resblocks with modified attention mask
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, attn_mask)
            for _ in range(layers)
        ])

        # For handling prompt-based input
        self.supports_prompts = prompt_length > 0

    def forward(self, x):
        if isinstance(x, list) and self.supports_prompts:
            # Process with prompt augmentation
            for i, block in enumerate(self.resblocks):
                x = block(x)
            return x
        else:
            # Standard forward pass
            return self.resblocks(x)


class ExtendedCLIPTextEncoder(nn.Module):
    """
    Custom CLIP text encoder that supports longer sequences and prompt-based learning
    """

    def __init__(
            self,
            clip_model,
            max_position_embeddings=1024,
            prompt_length=0,
            prompt_depth=0
    ):
        super().__init__()
        self.dtype = torch.float32  # Default dtype

        # Get model dimensions from the CLIP model
        self.context_length = max_position_embeddings
        self.transformer_width = clip_model.ln_final.weight.shape[0]
        self.transformer_heads = self.transformer_width // 64
        self.transformer_layers = len(
            [k for k in clip_model.state_dict() if k.startswith("transformer.resblocks") and ".ln_1.weight" in k])
        self.vocab_size = clip_model.token_embedding.weight.shape[0]
        self.prompt_length = prompt_length
        self.prompt_depth = prompt_depth

        # Transfer token embedding from the original model
        self.token_embedding = nn.Embedding(self.vocab_size, self.transformer_width)
        self.token_embedding.weight.data.copy_(clip_model.token_embedding.weight.data)

        # Create extended positional embedding
        self.positional_embedding = nn.Parameter(torch.zeros(max_position_embeddings, self.transformer_width))

        # Initialize the extended positional embeddings with the original ones
        orig_pos_embed = clip_model.positional_embedding.data
        orig_pos_embed_size = orig_pos_embed.shape[0]

        # Copy the original positional embeddings
        self.positional_embedding.data[:orig_pos_embed_size] = orig_pos_embed

        # Interpolate the rest if needed
        if max_position_embeddings > orig_pos_embed_size:
            # Sinusoidal extension (similar to how T5 does it)
            position_ids = torch.arange(max_position_embeddings, dtype=torch.float32)
            freqs = torch.exp(
                torch.arange(0, self.transformer_width, 2, dtype=torch.float32) *
                -(math.log(10000.0) / self.transformer_width)
            )
            pos_enc = position_ids.unsqueeze(1) * freqs.unsqueeze(0)
            pos_enc = torch.cat([torch.sin(pos_enc), torch.cos(pos_enc)], dim=1)
            if self.transformer_width % 2 == 1:
                pos_enc = torch.cat([pos_enc, torch.zeros(max_position_embeddings, 1)], dim=1)

            # Blend smoothly between original and extended embeddings
            blend_start = max(0, orig_pos_embed_size - 10)
            blend_length = min(20, max_position_embeddings - blend_start)

            if blend_length > 0:
                blend_weights = torch.linspace(1.0, 0.0, blend_length).unsqueeze(1)
                blend_region_orig = self.positional_embedding.data[blend_start:blend_start + blend_length]
                blend_region_ext = pos_enc[blend_start:blend_start + blend_length]
                blended = blend_weights * blend_region_orig + (1 - blend_weights) * blend_region_ext
                self.positional_embedding.data[blend_start:blend_start + blend_length] = blended

                # Fill the rest with the extended embeddings
                if blend_start + blend_length < max_position_embeddings:
                    self.positional_embedding.data[blend_start + blend_length:] = pos_enc[blend_start + blend_length:]

        # Create a custom attention mask for the transformer
        # Note: We'll make this dynamically during forward pass to handle variable lengths
        attn_mask = self._build_attention_mask(max_position_embeddings + prompt_length)

        # Custom transformer with extended position support
        self.transformer = CustomTransformer(
            width=self.transformer_width,
            layers=self.transformer_layers,
            heads=self.transformer_heads,
            attn_mask=attn_mask,
            prompt_length=prompt_length,
            max_position_embeddings=max_position_embeddings
        )

        # Final normalization layer
        self.ln_final = LayerNorm(self.transformer_width)

        # Text projection layer
        self.text_projection = nn.Parameter(torch.zeros(self.transformer_width, clip_model.text_projection.shape[1]))
        self.text_projection.data.copy_(clip_model.text_projection.data)

        # Copy weights from CLIP model for the rest of the components
        self._copy_transformer_weights(clip_model)

    def _build_attention_mask(self, seq_length):
        # Build a causal attention mask for the transformer
        mask = torch.empty(seq_length, seq_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _copy_transformer_weights(self, clip_model):
        """Copy weights from original CLIP model to our custom transformer"""
        clip_dict = clip_model.state_dict()

        # Copy layer normalization and transformer weights
        for i in range(self.transformer_layers):
            prefix = f"transformer.resblocks.{i}."

            # Copy attention weights
            self.transformer.resblocks[i].attn.in_proj_weight.data.copy_(
                clip_dict[prefix + "attn.in_proj_weight"])
            self.transformer.resblocks[i].attn.in_proj_bias.data.copy_(
                clip_dict[prefix + "attn.in_proj_bias"])
            self.transformer.resblocks[i].attn.out_proj.weight.data.copy_(
                clip_dict[prefix + "attn.out_proj.weight"])
            self.transformer.resblocks[i].attn.out_proj.bias.data.copy_(
                clip_dict[prefix + "attn.out_proj.bias"])

            # Copy MLP weights
            self.transformer.resblocks[i].mlp[0].weight.data.copy_(
                clip_dict[prefix + "mlp.c_fc.weight"])
            self.transformer.resblocks[i].mlp[0].bias.data.copy_(
                clip_dict[prefix + "mlp.c_fc.bias"])
            self.transformer.resblocks[i].mlp[2].weight.data.copy_(
                clip_dict[prefix + "mlp.c_proj.weight"])
            self.transformer.resblocks[i].mlp[2].bias.data.copy_(
                clip_dict[prefix + "mlp.c_proj.bias"])

            # Copy layer norms
            self.transformer.resblocks[i].ln_1.weight.data.copy_(
                clip_dict[prefix + "ln_1.weight"])
            self.transformer.resblocks[i].ln_1.bias.data.copy_(
                clip_dict[prefix + "ln_1.bias"])
            self.transformer.resblocks[i].ln_2.weight.data.copy_(
                clip_dict[prefix + "ln_2.weight"])
            self.transformer.resblocks[i].ln_2.bias.data.copy_(
                clip_dict[prefix + "ln_2.bias"])

        # Copy final layer norm
        self.ln_final.weight.data.copy_(clip_dict["ln_final.weight"])
        self.ln_final.bias.data.copy_(clip_dict["ln_final.bias"])

    def forward(self, input_ids, attention_mask=None, all_prompts=None, missing_type=None):
        """
        Forward pass with support for variable sequence lengths and prompt insertion

        Args:
            input_ids: Tokenized input text [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            all_prompts: Optional prompt embeddings to include
            missing_type: Type of missing modality (for conditioning prompt generation)

        Returns:
            Text features
        """
        batch_size = input_ids.size(0)

        # Get embeddings from tokens
        x = self.token_embedding(input_ids).type(self.dtype)  # [batch_size, n_ctx, d_model]

        # Add positional embeddings
        seq_len = input_ids.size(1)
        position_embeddings = self.positional_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + position_embeddings

        # Permute for transformer: [batch_size, seq_len, dim] -> [seq_len, batch_size, dim]
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Process with transformer
        if all_prompts is not None and self.prompt_length > 0:
            # Process with prompt augmentation
            combined = [x, all_prompts, 0,
                        missing_type if missing_type is not None else torch.zeros(batch_size).to(x.device)]
            outputs = self.transformer(combined)
            x = outputs[0]  # Extract the text representation
        else:
            # Standard forward pass
            x = self.transformer(x)

        # Permute back: [seq_len, batch_size, dim] -> [batch_size, seq_len, dim]
        x = x.permute(1, 0, 2)  # LND -> NLD

        # Final layer norm
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # Get features from the eot embedding (eot_token is the highest number in each sequence)
        if attention_mask is not None:
            # Use the position of the last non-padding token
            eot_indices = attention_mask.sum(dim=1) - 1
            eot_indices = torch.clamp(eot_indices, min=0)
        else:
            # Fallback to the original method - using the highest token ID as the EOT marker
            eot_indices = input_ids.argmax(dim=-1)

        x = x[torch.arange(x.shape[0]), eot_indices] @ self.text_projection

        return x


# Original OpenAI CLIP implementation
class CLIP(nn.Module):
    def __init__(self, embed_dim, context_length=77, vocab_size=49408, transformer_width=512,
                 transformer_heads=8, transformer_layers=12):
        super().__init__()

        self.context_length = context_length
        self.vocab_size = vocab_size

        # Text encoding
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.transformer = CustomTransformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self._build_attention_mask(context_length)
        )
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))

        # Initialize parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # Initialize transformer weights
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5

        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp[0].weight, std=fc_std)
            nn.init.normal_(block.mlp[2].weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def _build_attention_mask(self, context_length):
        mask = torch.triu(torch.ones(context_length, context_length) * float('-inf'), diagonal=1)
        return mask

    def forward(self, text):
        # Not implementing full forward pass as we're only using it to load weights
        pass


def load_clip_model(name="ViT-B/16", device="cpu", jit=False, download_root=None):
    """
    Load a CLIP model from pre-trained weights

    Args:
        name: Model name (ViT-B/32, ViT-B/16, ViT-L/14)
        device: Device to load the model on
        jit: Whether to load the optimized JIT model
        download_root: Path to download the model files to

    Returns:
        CLIP model with loaded weights
    """
    if name not in _MODELS:
        raise ValueError(f"Model {name} not found; available models = {list(_MODELS.keys())}")

    model_path = _MODELS[name]

    if download_root is not None:
        os.makedirs(download_root, exist_ok=True)
        model_path = os.path.join(download_root, os.path.basename(model_path))

        if not os.path.exists(model_path):
            print(f"Downloading {_MODELS[name]} to {model_path}...")
            urllib.request.urlretrieve(_MODELS[name], model_path)

    # Load the model weights
    with open(model_path, 'rb') as opened_file:
        try:
            # Loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # Loading saved state dict
            state_dict = torch.load(opened_file, map_location="cpu")

    # Handle state dictionary format
    if state_dict is None:
        # Extract state dict from JIT model
        state_dict = {name: param for name, param in model.named_parameters()}
        for name, buffer in model.named_buffers():
            state_dict[name] = buffer

    # Create model based on architecture
    if name == "ViT-B/32":
        # Standard ViT-B/32 architecture
        clip_model = CLIP(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
    elif name == "ViT-B/16":
        # ViT-B/16 architecture
        clip_model = CLIP(
            embed_dim=512,
            context_length=77,
            vocab_size=49408,
            transformer_width=512,
            transformer_heads=8,
            transformer_layers=12
        )
    elif name == "ViT-L/14":
        # ViT-L/14 architecture
        clip_model = CLIP(
            embed_dim=768,
            context_length=77,
            vocab_size=49408,
            transformer_width=768,
            transformer_heads=12,
            transformer_layers=24
        )
    else:
        raise ValueError(f"Unknown model architecture: {name}")

    # Load weights from state dict
    clip_model.load_state_dict(state_dict)
    return clip_model.to(device)


def create_extended_text_encoder(model_name="ViT-B/16", max_position_embeddings=1024,
                                 prompt_length=0, prompt_depth=0, device="cpu", download_root=None):
    """
    Create an extended CLIP text encoder with support for longer sequences

    Args:
        model_name: CLIP model name ("ViT-B/32", "ViT-B/16", "ViT-L/14")
        max_position_embeddings: Maximum sequence length to support
        prompt_length: Length of prompts to use (0 to disable)
        prompt_depth: Depth of prompts to use (0 to disable)
        device: Device to load the model on
        download_root: Path to download the model files to

    Returns:
        Extended CLIP text encoder
    """
    # Load the original CLIP model
    clip_model = load_clip_model(name=model_name, device=device, download_root=download_root)

    # Create extended text encoder
    extended_encoder = ExtendedCLIPTextEncoder(
        clip_model=clip_model,
        max_position_embeddings=max_position_embeddings,
        prompt_length=prompt_length,
        prompt_depth=prompt_depth
    ).to(device)

    return extended_encoder


# Simple BPE tokenizer for CLIP - similar to what's used in the original CLIP
class SimpleTokenizer:
    def __init__(self, bpe_path="bpe_simple_vocab_16e6.txt.gz"):
        # This is a placeholder - in a real implementation, this would load the BPE vocabulary
        # For simplicity, we're just creating a basic tokenizer
        self.vocab = {f"<|token_{i}|>": i for i in range(49408)}
        self.encoder = {v: k for k, v in self.vocab.items()}
        self.bpe_ranks = {}
        self.byte_encoder = {}

        # Special tokens
        self.vocab["<|startoftext|>"] = 49406
        self.vocab["<|endoftext|>"] = 49407

        # Add these tokens to the encoder as well
        self.encoder[49406] = "<|startoftext|>"
        self.encoder[49407] = "<|endoftext|>"

    def tokenize(self, text):
        # This is a simplified tokenizer that just splits on spaces
        # Real implementation would use BPE encoding
        tokens = text.lower().split()
        return tokens

    def encode(self, text):
        tokens = self.tokenize(text)
        # Add start and end tokens
        tokens = ["<|startoftext|>"] + tokens + ["<|endoftext|>"]

        # Convert to token IDs, using a simple lookup
        # In a real implementation, this would use the BPE vocabulary
        # For simplicity, we're just returning placeholder IDs
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                # For unknown tokens, use token_0
                token_ids.append(self.vocab["<|token_0|>"])

        return token_ids

    def decode(self, token_ids):
        # Convert token IDs back to text
        tokens = [self.encoder.get(token_id, "<|token_0|>") for token_id in token_ids]
        # Remove special tokens
        tokens = [token for token in tokens if token not in ("<|startoftext|>", "<|endoftext|>")]
        return " ".join(tokens)
