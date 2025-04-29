# Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import BertModel, BertTokenizer
from torchvision.models import vit_b_16
import torchaudio.models
import torch
import os
import urllib
import hashlib
import warnings

def build_model(state_dict: dict, prompt_length, prompt_depth):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, prompt_length, prompt_depth
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    #convert_weights(model)
    try:
        model.load_state_dict(state_dict)
    except:
        missing_keys, _ = model.load_state_dict(state_dict, strict=False)
        print('Weights not found for some missing keys: ', missing_keys)
    return model

class PromptEmbedding(nn.Module):
    def __init__(self, embed_dim, prompt_length, prompt_depth):
        super().__init__()
        self.prompt_depth = prompt_depth
        self.prompt_embeddings = nn.Parameter(torch.randn(prompt_depth, prompt_length, embed_dim))

    def forward(self, x, layer_idx):
        if layer_idx < self.prompt_depth:
            prompt = self.prompt_embeddings[layer_idx]
            # 在序列前拼接 prompt
            x = torch.cat([prompt.unsqueeze(0).expand(x.size(0), -1, -1), x], dim=1)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_1 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.GELU(),
            nn.Linear(width * 4, width)
        )
        self.ln_2 = nn.LayerNorm(width)

    def forward(self, x):
        x_res = x
        x = self.ln_1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = x + x_res

        x_res = x
        x = self.ln_2(x)
        x = self.mlp(x)
        return x + x_res

class Transformer(nn.Module):
    def __init__(self, width, layers, heads, prompt_length, prompt_depth):
        super().__init__()
        self.resblocks = nn.ModuleList([
            TransformerLayer(width, heads) for _ in range(layers)
        ])
        self.prompt = PromptEmbedding(width, prompt_length, prompt_depth)

    def forward(self, x: torch.Tensor):
        for idx, blk in enumerate(self.resblocks):
            x = self.prompt(x, idx)
            x = blk(x)
        return x

class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim,
        image_resolution,
        vision_layers,
        vision_width,
        vision_patch_size,
        context_length,
        vocab_size,
        transformer_width,
        transformer_heads,
        transformer_layers,
        prompt_length,
        prompt_depth
    ):
        super().__init__()
        self.context_length = context_length

        # Vision encoder (ViT only for now)
        self.conv1 = nn.Conv2d(3, vision_width, kernel_size=vision_patch_size, stride=vision_patch_size, bias=False)
        scale = vision_width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(vision_width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((image_resolution // vision_patch_size) ** 2 + 1, vision_width))
        self.ln_pre = nn.LayerNorm(vision_width)
        self.visual_transformer = Transformer(vision_width, vision_layers, transformer_heads, prompt_length, prompt_depth)
        self.ln_post = nn.LayerNorm(vision_width)
        self.visual_proj = nn.Parameter(scale * torch.randn(vision_width, embed_dim))

        # Text encoder
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding_text = nn.Parameter(scale * torch.randn(context_length, transformer_width))
        self.text_transformer = Transformer(transformer_width, transformer_layers, transformer_heads, prompt_length, prompt_depth)
        self.ln_final = nn.LayerNorm(transformer_width)
        self.text_proj = nn.Parameter(scale * torch.randn(transformer_width, embed_dim))

    def encode_image(self, image):
        x = self.conv1(image)  # shape = [*, width, grid, grid]
        x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = torch.cat([self.class_embedding.unsqueeze(0).expand(x.shape[0], -1).unsqueeze(1), x], dim=1)
        x = x + self.positional_embedding
        x = self.ln_pre(x)
        x = self.visual_transformer(x)
        x = self.ln_post(x[:, 0, :])  # only use CLS token
        return x @ self.visual_proj

    def encode_text(self, text):
        x = self.token_embedding(text) + self.positional_embedding_text
        x = self.text_transformer(x)
        x = self.ln_final(x[:, 0, :])
        return x @ self.text_proj

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)
        return image_features, text_features

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}



def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]

    download_target = os.path.join(root, filename)
    if os.path.exists(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn("Hash mismatch, re-downloading.")
    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        output.write(source.read())

    return download_target

def load_clip_to_cpu(backbone_name: str, prompt_length: int = 0, prompt_depth: int = 0):
    url = _MODELS[backbone_name]
    model_path = _download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu")  # JIT 格式
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(state_dict or model.state_dict(), prompt_length, prompt_depth)
    return model

# --------- Quality Estimator ---------
class QualityEstimator(nn.Module):
    def __init__(self, input_dim, hidden_dim, prompt_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, prompt_dim)
        )

    def forward(self, modality_features: dict) -> dict:
        """
        modality_features: Dict[str, Tensor]  -> {modality_name: feature_tensor}
        return: Dict[str, Tensor] -> {modality_name: quality_prompt}
        """
        quality_prompts = {}
        for name, feat in modality_features.items():
            quality_prompts[name] = self.fc(feat)
        return quality_prompts


# --------- Missing Modality Generator ---------
class MissingModalityGenerator(nn.Module):
    def __init__(self, prompt_dim, hidden_dim, output_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(prompt_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.quality_head = nn.Linear(output_dim, prompt_dim)

    def forward(self, available_feats: torch.Tensor, quality_prompt: torch.Tensor) -> tuple:
        """
        available_feats: Tensor[B, H]
        quality_prompt: Tensor[B, D_prompt]
        return: (generated_feature, quality_prompt)
        """
        x = torch.cat([available_feats, quality_prompt], dim=-1)
        gen_feat = self.generator(x)
        gen_prompt = self.quality_head(gen_feat)
        return gen_feat, gen_prompt


# --------- Modality Encoder ---------
class ModalityEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class CLIPEncoder(nn.Module):
    def __init__(self, backbone_name="ViT-B/32", prompt_length=0, prompt_depth=0, output_dim=512, freeze=True):
        super().__init__()
        self.clip_model = load_clip_to_cpu(backbone_name, prompt_length, prompt_depth)
        self.output_proj = nn.Linear(self.clip_model.visual.output_dim, output_dim)

        if freeze:
            for param in self.clip_model.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        x: image tensor [B, C, H, W]
        return: [B, output_dim]
        """
        image_features = self.clip_model.encode_image(x)
        return self.output_proj(image_features)


# --------- Cross-Modality Transformer Block ---------
class MultiStreamTransformer(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers, prompt_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.prompt_proj = nn.Linear(prompt_dim, hidden_dim)

    def forward(self, modality_feats: dict, quality_prompts: dict) -> torch.Tensor:
        """
        modality_feats: {modality: [B, H]}
        quality_prompts: {modality: [B, D_prompt]}
        return: Tensor[B, H]
        """
        all_feats = []
        for name in modality_feats:
            fused_feat = modality_feats[name] + self.prompt_proj(quality_prompts[name])
            all_feats.append(fused_feat.unsqueeze(1))  # [B, 1, H]
        x = torch.cat(all_feats, dim=1)  # [B, N_modality, H]

        for layer in self.layers:
            x = layer(x)

        return x.mean(dim=1)  # 融合后的多模态特征 [B, H]


# --------- Task Head ---------
class TaskHead(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# --------- Overall Model ---------
class QualityAwareMultiModalModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(cfg['input_dims'][name], cfg['hidden_dim'])
            for name in cfg['input_dims']
        })

        self.quality_estimator = QualityEstimator(cfg['hidden_dim'], cfg['hidden_dim'], cfg['prompt_dim'])
        self.generator = MissingModalityGenerator(cfg['prompt_dim'], cfg['hidden_dim'], cfg['hidden_dim'])
        self.transformer = MultiStreamTransformer(cfg['hidden_dim'], cfg['num_heads'], cfg['num_layers'], cfg['prompt_dim'])
        self.task_head = TaskHead(cfg['hidden_dim'], cfg['output_dim'])

    def forward(self, inputs: dict, modality_mask: dict) -> torch.Tensor:
        """
        inputs: {modality: Tensor[B, D_in]}
        modality_mask: {modality: Bool}  # True if present
        """
        encoded_feats, quality_prompts = {}, {}

        # Step 1: Encode available modalities
        for name, x in inputs.items():
            if modality_mask[name]:
                feat = self.encoders[name](x)
                encoded_feats[name] = feat
        # Step 2: Generate quality prompts
        quality_prompts.update(self.quality_estimator(encoded_feats))

        # Step 3: Generate missing modalities
        for name in inputs:
            if not modality_mask[name]:
                # 平均其他可用模态特征
                available_feats = torch.stack([v for v in encoded_feats.values()], dim=0).mean(dim=0)
                gen_feat, gen_prompt = self.generator(available_feats, torch.zeros_like(quality_prompts[next(iter(quality_prompts))]))
                encoded_feats[name] = gen_feat
                quality_prompts[name] = gen_prompt

        # Step 4: Transformer融合
        fused_feat = self.transformer(encoded_feats, quality_prompts)

        # Step 5: Task head
        return self.task_head(fused_feat)
