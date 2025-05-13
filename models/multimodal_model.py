import inspect

from timm import create_model
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel
from models.quality_aware_prompting import EnhancedModalityQualityEstimator, \
    QualityAwareFeatureFusion

from .modality_generator import CycleGenerationModel, CrossModalGenerator


class InterLayerPromptBlock(nn.Module):
    def __init__(self, embed_dim, prompt_len):
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),  # Fix: LayerNorm needs to match the concatenated dimension
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.prompt_len = prompt_len

    def forward(self, prev_prompt, feats):
        # feats: [B, N+1, D], prev_prompt: [B, prompt_len, D]
        feat_token = feats.mean(dim=1, keepdim=True).expand(-1, self.prompt_len, -1)  # [B, prompt_len, D]
        concat = torch.cat([prev_prompt, feat_token], dim=-1)  # [B, prompt_len, 2*D]
        new_prompt = self.prompt_proj(concat)
        return new_prompt


class PromptedViT(nn.Module):
    def __init__(self, base_model, prompt_len=5, prompt_depth=6):
        super().__init__()
        self.vit = base_model
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth
        self.embed_dim = self.vit.embed_dim
        self.num_layers = len(self.vit.blocks)

        assert prompt_depth <= self.num_layers, "prompt_depth must not exceed number of layers"

        # Initialize first layer prompts (learnable)
        self.init_prompt = nn.Parameter(torch.randn(1, prompt_len, self.embed_dim))

        # Define InterLayerPromptBlock only for the first prompt_depth layers
        self.prompt_blocks = nn.ModuleList([
            InterLayerPromptBlock(self.embed_dim, prompt_len)
            for _ in range(prompt_depth)
        ])

    def forward(self, x):
        B = x.size(0)
        x = self.vit.patch_embed(x)  # [B, N, D]
        cls_token = self.vit.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_token, x), dim=1)  # [B, N+1, D]
        x = x + self.vit.pos_embed[:, :x.size(1), :]
        x = self.vit.pos_drop(x)

        prompt = self.init_prompt.expand(B, -1, -1)  # [B, prompt_len, D]

        for i, block in enumerate(self.vit.blocks):
            if i < self.prompt_depth:
                # Concatenate prompts and process
                x_with_prompt = torch.cat([prompt, x], dim=1)
                x_with_prompt = block(x_with_prompt)  # [B, prompt_len + N + 1, D]
                prompt, x = x_with_prompt[:, :self.prompt_len], x_with_prompt[:, self.prompt_len:]
                prompt = self.prompt_blocks[i](prompt, x)  # Update prompt for next layer
            else:
                # Normal TransformerBlock execution without prompts
                x = block(x)

        x = self.vit.norm(x)
        return self.vit.head(x[:, 0])  # Return CLS token


class CLIPTextPromptEncoder(nn.Module):
    def __init__(self, model_name='openai/clip-vit-base-patch16', prompt_len=5, prompt_depth=6):
        super().__init__()
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.prompt_len = prompt_len
        self.prompt_depth = prompt_depth
        self.hidden_size = self.text_encoder.config.hidden_size
        self.num_layers = len(self.text_encoder.text_model.encoder.layers)

        # Initialize first layer prompts (learnable)
        self.init_prompt = nn.Parameter(torch.randn(1, prompt_len, self.hidden_size))

        # Inter-layer prompt generators
        self.prompt_blocks = nn.ModuleList([
            InterLayerPromptBlock(self.hidden_size, prompt_len)
            if i < prompt_depth else nn.Identity()
            for i in range(self.num_layers)
        ])

    def forward(self, input_ids, attention_mask):
        B = input_ids.size(0)
        prompt = self.init_prompt.expand(B, -1, -1)  # [B, prompt_len, D]

        # Get the initial embeddings using the CLIP text model's embedding layer
        # CLIP's structure is different - we need to access the text_model
        inputs_embeds = self.text_encoder.text_model.embeddings(input_ids)
        hidden_states = inputs_embeds

        # Manually forward through each Transformer layer in CLIP's text encoder
        for i, block in enumerate(self.text_encoder.text_model.encoder.layers):
            if i < self.prompt_depth:
                hidden_states = torch.cat([prompt, hidden_states], dim=1)
                # Extend attention mask
                extended_mask = torch.cat([
                    torch.ones(B, self.prompt_len, device=attention_mask.device),
                    attention_mask
                ], dim=1)

                # Format the attention mask for CLIP's encoder layers
                extended_attention_mask = self._prepare_attention_mask(extended_mask, hidden_states.shape[1])
            else:
                extended_attention_mask = self._prepare_attention_mask(attention_mask, hidden_states.shape[1])

            # Pass through the layer
            layer_outputs = block(
                hidden_states,
                attention_mask=extended_attention_mask,
                causal_attention_mask=None,
                output_attentions=False
            )
            hidden_states = layer_outputs[0]

            if i < self.prompt_depth:
                # Extract prompt and hidden states
                prompt, hidden_states = hidden_states[:, :self.prompt_len], hidden_states[:, self.prompt_len:]
                # Update prompt for next layer
                prompt = self.prompt_blocks[i](prompt, hidden_states)

        # Apply final layer norm from CLIP's text model
        hidden_states = self.text_encoder.text_model.final_layer_norm(hidden_states)

        # Return the text embedding (first token)
        return hidden_states[:, 0]

    def _prepare_attention_mask(self, attention_mask, input_shape):
        # Create a 4D attention mask of shape [batch_size, 1, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        # Convert mask to the format expected by the model (1.0 for padding and 0.0 for non-padding)
        attention_mask = (1.0 - attention_mask) * -10000.0
        return attention_mask


class MultimodalPromptModel(nn.Module):
    def __init__(
            self,
            image_encoder_backbone,
            text_encoder_backbone,  # 直接接收编码器实例
            image_prompt_len=5,
            text_prompt_len=5,
            prompt_depth=6,
            fusion_dim=512,
            num_classes=23,
            freeze_image_encoder=False,
            freeze_text_encoder=False,
            use_quality_prompt=True,
            use_cross_modal_prompt=True,
            use_modality_generator=True,
            max_length=512,
            encoder_type='clip',
            use_clip_encoders=False  # 标志是否使用CLIP编码器
    ):
        super().__init__()
        self.max_length = max_length
        self.encoder_type = encoder_type.lower()
        self.use_clip_encoders = use_clip_encoders

        # 保存编码器
        self.image_encoder = image_encoder_backbone
        self.text_encoder = text_encoder_backbone

        # 处理CLIP模式
        # 处理CLIP模式
        if use_clip_encoders:
            # 获取维度信息
            self.image_dim = self.image_encoder.config.hidden_size
            self.text_dim = self.text_encoder.config.hidden_size

            print(f"Using CLIP encoders - Image dim: {self.image_dim}, Text dim: {self.text_dim}")

            # 根据CLIP模型的真实结构设置组件
            # CLIP视觉模型
            self.image_patch_embed = self.image_encoder.vision_model.embeddings  # 正确访问embeddings
            self.image_blocks = self.image_encoder.vision_model.encoder.layers
            self.image_norm = self.image_encoder.vision_model.post_layernorm

            # CLIP文本模型
            self.text_embeddings = self.text_encoder.text_model.embeddings
            self.text_blocks = self.text_encoder.text_model.encoder.layers
            self.text_norm = self.text_encoder.text_model.final_layer_norm

            # 创建兼容接口
            self.image_cls_token = nn.Parameter(torch.zeros(1, 1, self.image_dim))
            self.image_pos_embed = nn.Parameter(torch.zeros(1, 197, self.image_dim))  # 仅作为占位符
            self.image_pos_drop = nn.Dropout(0.0)

        else:
            # 处理常规编码器
            # 检查是否为CLIP视觉模型
            if hasattr(self.image_encoder, 'config') and hasattr(self.image_encoder.config, 'hidden_size'):
                # CLIP视觉模型
                self.image_dim = self.image_encoder.config.hidden_size

                # 正确访问CLIP视觉模型的组件
                if hasattr(self.image_encoder, 'vision_model'):
                    self.image_patch_embed = self.image_encoder.vision_model.embeddings
                    self.image_blocks = self.image_encoder.vision_model.encoder.layers
                    self.image_norm = self.image_encoder.vision_model.post_layernorm
                else:
                    # 旧版本或不同结构的CLIP视觉模型
                    vision_model = getattr(self.image_encoder, 'vision_model', self.image_encoder)
                    self.image_patch_embed = vision_model.embeddings
                    self.image_blocks = vision_model.encoder.layers
                    self.image_norm = getattr(vision_model, 'post_layernorm',
                                              getattr(vision_model, 'layer_norm', None))

                # 创建兼容接口
                self.image_cls_token = nn.Parameter(torch.zeros(1, 1, self.image_dim))
                self.image_pos_embed = nn.Parameter(torch.zeros(1, 197, self.image_dim))
                self.image_pos_drop = nn.Dropout(0.0)
            else:
                # 标准ViT模型
                self.image_dim = self.image_encoder.embed_dim
                self.image_patch_embed = self.image_encoder.patch_embed
                self.image_cls_token = self.image_encoder.cls_token
                self.image_pos_embed = self.image_encoder.pos_embed
                self.image_pos_drop = self.image_encoder.pos_drop
                self.image_blocks = self.image_encoder.blocks
                self.image_norm = self.image_encoder.norm

            # 检查文本编码器类型
            if hasattr(self.text_encoder, 'text_model'):
                # CLIP文本模型
                self.text_dim = self.text_encoder.config.hidden_size
                self.text_embeddings = self.text_encoder.text_model.embeddings
                self.text_blocks = self.text_encoder.text_model.encoder.layers
                self.text_norm = self.text_encoder.text_model.final_layer_norm
            elif hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layer'):
                # RoBERTa等Transformer模型
                self.text_dim = self.text_encoder.config.hidden_size
                self.text_embeddings = self.text_encoder.embeddings
                self.text_blocks = self.text_encoder.encoder.layer
                self.text_norm = self.text_encoder.encoder.layer[-1].output.LayerNorm
            else:
                # 其他类型的文本模型
                self.text_dim = self.text_encoder.config.hidden_size
                if hasattr(self.text_encoder, 'encoder') and hasattr(self.text_encoder.encoder, 'layers'):
                    self.text_embeddings = self.text_encoder.embeddings
                    self.text_blocks = self.text_encoder.encoder.layers
                    self.text_norm = getattr(self.text_encoder, 'final_layer_norm',
                                             getattr(self.text_encoder, 'layer_norm', None))
                else:
                    raise ValueError(f"Unsupported text encoder structure")

        print("image_dim:", self.image_dim, " text_dim: ", self.text_dim)



        # 初始提示参数
        self.use_quality_prompt = use_quality_prompt
        self.use_cross_modal_prompt = use_cross_modal_prompt
        self.use_modality_generator = use_modality_generator
        self.prompt_depth = prompt_depth
        self.image_prompt_len = image_prompt_len
        self.text_prompt_len = text_prompt_len
        self.image_init_prompt = nn.Parameter(torch.randn(1, image_prompt_len, self.image_dim))
        self.text_init_prompt = nn.Parameter(torch.randn(1, text_prompt_len, self.text_dim))

        # 集成模态生成器（使用CycleGenerationModel替代CrossModalGenerator）
        if use_modality_generator:
            modality_dims = {
                'image': self.image_dim,
                'text': self.text_dim
            }
            self.modality_generator = CycleGenerationModel(modality_dims, fusion_hidden_dim=fusion_dim)

        # 使用增强的质量评估器替换原有评估器
        if use_quality_prompt:
            self.quality_estimator = EnhancedModalityQualityEstimator(self.image_dim, self.text_dim)



        # 模态内层间提示
        self.cross_modal_layer = nn.ModuleList([
            InterLayerPromptBlock(self.image_dim, image_prompt_len)
            for _ in range(prompt_depth)
        ])
        self.image_InterPrompt_layer = nn.ModuleList([
            InterLayerPromptBlock(self.image_dim, image_prompt_len)
            for _ in range(prompt_depth)
        ])

        self.text_InterPrompt_layer = nn.ModuleList([
            InterLayerPromptBlock(self.text_dim, text_prompt_len)
            for _ in range(prompt_depth)
        ])

        # 融合和分类层
        fusion_input_dim = self.image_dim + self.text_dim + 4  # 基础特征 + 缺失类型
        if use_quality_prompt:
            fusion_input_dim += 13  # 增加质量评分（2个质量特征5+3个详细评分1）

        # 使用质量引导的特征融合替换跨模态提示
        if use_cross_modal_prompt:
            self.feature_fusion = QualityAwareFeatureFusion(
                self.image_dim,
                self.text_dim,
                fusion_dim
            )

        print("fusion_input_dim=", fusion_input_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # 冻结预训练参数（如果需要）
        if freeze_image_encoder:
            # 分别处理模块和参数
            for module in [self.image_patch_embed, self.image_blocks, self.image_norm]:
                for param in module.parameters():
                    param.requires_grad = False

            # 直接处理参数
            if isinstance(self.image_cls_token, nn.Parameter):
                self.image_cls_token.requires_grad = False
            if isinstance(self.image_pos_embed, nn.Parameter):
                self.image_pos_embed.requires_grad = False

        if freeze_text_encoder:
            for module in [self.text_embeddings, self.text_blocks, self.text_norm]:
                for param in module.parameters():
                    param.requires_grad = False

    def _prepare_attention_mask(self, attention_mask, input_shape):
        """准备用于Transformer层的注意力掩码"""
        # 扩展维度以创建4D掩码
        # [batch_size, 1, 1, seq_length] 或 [batch_size, 1, seq_length, seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            # 已经是扩展形式
            extended_attention_mask = attention_mask

        # 将掩码转换为Transformer所需的格式
        # (1.0 表示被掩码的位置，0.0 表示有效位置)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def _prepare_clip_attention_mask(self, attention_mask):
        """
        为CLIP视觉模型准备注意力掩码

        Args:
            attention_mask: 形状为[batch_size, seq_len]的掩码

        Returns:
            形状为[batch_size, 1, seq_len, seq_len]的掩码，适用于CLIP注意力层
        """
        batch_size, seq_length = attention_mask.shape

        # 创建一个方形的掩码矩阵，形状为[batch_size, seq_len, seq_len]
        # 其中第i行第j列表示token_i是否可以关注token_j
        # 在CLIP中，每个token都可以关注所有token，所以我们用全1矩阵
        causal_mask = torch.ones((batch_size, seq_length, seq_length), device=attention_mask.device)

        # 增加一个维度，得到形状[batch_size, 1, seq_len, seq_len]
        causal_mask = causal_mask.unsqueeze(1)

        # CLIP中，0表示被掩码的位置（被忽略），非0表示有效的注意力位置
        # 因此不需要像某些模型那样进行(1.0 - mask) * -10000.0的变换

        return causal_mask

    def _extract_features(self, image, input_ids, attention_mask):
        """提取图像和文本的原始特征"""
        batch_size = image.size(0) if image is not None else input_ids.size(0)
        device = next(self.parameters()).device

        # 处理图像特征 (如果存在)
        image_embed = None
        if image is not None:
            if self.use_clip_encoders:
                # 对于CLIP视觉模型，我们直接使用embeddings方法
                # 避免使用forward方法，因为我们想要自己控制Transformer层
                try:
                    # 尝试直接获取embeddings输出
                    if hasattr(self.image_encoder, 'embeddings'):
                        image_embed = self.image_encoder.embeddings(pixel_values=image)
                    else:
                        # 回退方案：运行模型一次，但禁用注意力层
                        with torch.no_grad():
                            # 简化：只获取embeddings的输出
                            vision_outputs = self.image_encoder(
                                pixel_values=image,
                                output_hidden_states=True,
                                output_attentions=False,
                                return_dict=True
                            )
                            # 使用第一个隐藏状态（embeddings的输出）
                            image_embed = vision_outputs.hidden_states[0] if hasattr(vision_outputs,
                                                                                     'hidden_states') else None

                    # 如果仍然无法获取embeddings输出，打印错误
                    if image_embed is None:
                        print("WARNING: Unable to get proper embeddings from CLIP vision model.")
                        # 创建一个随机嵌入作为后备
                        image_embed = torch.randn(batch_size, 197, self.image_dim, device=device)
                except Exception as e:
                    print(f"Error extracting CLIP vision embeddings: {e}")
                    # 创建一个随机嵌入作为后备
                    image_embed = torch.randn(batch_size, 197, self.image_dim, device=device)
            else:
                # 标准ViT模型
                image_embed = self.image_patch_embed(image)  # [B, N, D_img]
                image_cls = self.image_cls_token.expand(batch_size, -1, -1)  # [B, 1, D_img]
                image_embed = torch.cat((image_cls, image_embed), dim=1)  # [B, N+1, D_img]
                image_embed = image_embed + self.image_pos_embed[:, :image_embed.size(1), :]
                image_embed = self.image_pos_drop(image_embed)

        # 处理文本特征 (如果存在)
        text_embed = None
        if input_ids is not None:
            if self.use_clip_encoders:
                # 对于CLIP文本模型，尝试直接使用embeddings
                try:
                    if hasattr(self.text_encoder, 'embeddings'):
                        text_embed = self.text_encoder.embeddings(input_ids)
                    else:
                        # 回退方案：使用text_model的embeddings
                        text_embed = self.text_embeddings(input_ids)
                except Exception as e:
                    print(f"Error extracting CLIP text embeddings: {e}")
                    # 创建一个随机嵌入作为后备
                    seq_len = input_ids.size(1)
                    text_embed = torch.randn(batch_size, seq_len, self.text_dim, device=device)
            else:
                # 标准文本模型
                text_embed = self.text_embeddings(input_ids)  # [B, L, D_txt]

        return image_embed, text_embed

    def _process_modalities(self, image_embed, text_embed, attention_mask, missing_type):
        """处理模态，包括模态缺失的情况，并使用原始特征进行重建"""
        batch_size = image_embed.size(0) if image_embed is not None else text_embed.size(0)
        device = next(self.parameters()).device
        missing_type = missing_type.to(device)
        # Ensure missing_type is the correct shape and on the right device
        if missing_type is None:
            # Default to no missing modalities if not specified
            missing_type = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(missing_type, int):
            # If it's a single integer, expand to batch size
            missing_type = torch.full((batch_size,), missing_type, dtype=torch.long, device=device)
        elif missing_type.size(0) != batch_size:
            # If batch size doesn't match, this is a critical error
            raise ValueError(
                f"missing_type batch size ({missing_type.size(0)}) doesn't match data batch size ({batch_size})")

        # Make sure missing_type is on the same device as the model

        # 检测缺失的模态
        is_image_missing = (missing_type == 1) | (missing_type == 3)
        is_text_missing = (missing_type == 2) | (missing_type == 3)

        # 提取原始特征的tokens用于重建损失计算
        token_count = 5 # 每个模态使用的token数量

        original_features = {
            'image': None if image_embed is None else image_embed[:, :token_count].reshape(batch_size, token_count,
                                                                                           -1).detach(),
            'text': None if text_embed is None else text_embed[:, :token_count].reshape(batch_size, token_count,
                                                                                        -1).detach()
        }

        # 处理模态生成
        generated_features = {'image': None, 'text': None}
        reconstructed_features = {'image': None, 'text': None}

        if self.use_modality_generator:
            # 准备输入数据 - 使用可见的模态
            gen_input = {
                'image': None if image_embed is None else image_embed[:, :token_count].detach(),  # 使用多个token
                'text': None if text_embed is None else text_embed[:, :token_count].detach()  # 使用多个token
            }

            # 对每个样本单独处理缺失情况
            for b in range(batch_size):
                mt = missing_type[b].item()

                # 根据缺失类型准备单样本输入
                sample_input = {
                    'image': None if mt in [1, 3] else (
                        None if gen_input['image'] is None else gen_input['image'][b:b + 1]),
                    'text': None if mt in [2, 3] else (
                        None if gen_input['text'] is None else gen_input['text'][b:b + 1])
                }

                # 生成缺失模态
                sample_gen, sample_recon = self.modality_generator(sample_input, mt)

                # 处理生成的图像特征
                if mt in [1, 3] and 'image' in sample_gen and sample_gen['image'] is not None:
                    if generated_features['image'] is None:
                        # 创建存储空间，考虑到多个token
                        # [batch_size, token_count, image_dim]
                        token_dim = sample_gen['image'].size(-1)
                        # 检查样本生成shape是 [1, token_count, dim] 还是 [token_count, dim]
                        if sample_gen['image'].dim() == 3:  # [1, token_count, dim]
                            tokens_per_sample = sample_gen['image'].size(1)
                        else:  # [token_count, dim]
                            tokens_per_sample = sample_gen['image'].size(0) if sample_gen['image'].dim() > 1 else 1
                        generated_features['image'] = torch.zeros(batch_size, tokens_per_sample, token_dim,
                                                                  device=device)

                    # 正确处理单个token和多个token的情况
                    if sample_gen['image'].dim() == 1:  # 单个token [image_dim]
                        generated_features['image'][b, 0] = sample_gen['image']
                    elif sample_gen['image'].dim() == 2:  # 多个token [token_count, image_dim]
                        generated_features['image'][b] = sample_gen['image']
                    else:  # 带batch维度 [1, token_count, image_dim]
                        generated_features['image'][b] = sample_gen['image'].squeeze(0)

                # 处理生成的文本特征
                if mt in [2, 3] and 'text' in sample_gen and sample_gen['text'] is not None:
                    if generated_features['text'] is None:
                        # 创建存储空间，考虑到多个token
                        # [batch_size, token_count, text_dim]
                        token_dim = sample_gen['text'].size(-1)
                        # 检查样本生成shape是 [1, token_count, dim] 还是 [token_count, dim]
                        if sample_gen['text'].dim() == 3:  # [1, token_count, dim]
                            tokens_per_sample = sample_gen['text'].size(1)
                        else:  # [token_count, dim]
                            tokens_per_sample = sample_gen['text'].size(0) if sample_gen['text'].dim() > 1 else 1
                        generated_features['text'] = torch.zeros(batch_size, tokens_per_sample, token_dim,
                                                                 device=device)

                    # 正确处理单个token和多个token的情况
                    if sample_gen['text'].dim() == 1:  # 单个token [text_dim]
                        generated_features['text'][b, 0] = sample_gen['text']
                    elif sample_gen['text'].dim() == 2:  # 多个token [token_count, text_dim]
                        generated_features['text'][b] = sample_gen['text']
                    else:  # 带batch维度 [1, token_count, text_dim]
                        generated_features['text'][b] = sample_gen['text'].squeeze(0)

                # 处理重建的特征
                for mod in sample_recon:
                    if sample_recon[mod] is not None:
                        if reconstructed_features[mod] is None:
                            # 创建存储空间，考虑到多个token
                            token_dim = sample_recon[mod].size(-1)
                            # 检查样本重建shape是 [1, token_count, dim] 还是 [token_count, dim]
                            if sample_recon[mod].dim() == 3:  # [1, token_count, dim]
                                tokens_per_sample = sample_recon[mod].size(1)
                            else:  # [token_count, dim]
                                tokens_per_sample = sample_recon[mod].size(0) if sample_recon[mod].dim() > 1 else 1
                            reconstructed_features[mod] = torch.zeros(
                                batch_size, tokens_per_sample, token_dim, device=device)

                        # 正确处理单个token和多个token的情况
                        if sample_recon[mod].dim() == 1:  # 单个token [dim]
                            reconstructed_features[mod][b, 0] = sample_recon[mod]
                        elif sample_recon[mod].dim() == 2:  # 多个token [token_count, dim]
                            reconstructed_features[mod][b] = sample_recon[mod]
                        else:  # 带batch维度 [1, token_count, dim]
                            reconstructed_features[mod][b] = sample_recon[mod].squeeze(0)

        # # 使用生成的特征替换缺失的模态
        if is_image_missing.any() and generated_features['image'] is not None:
            # 如果image_embed为None，创建新的
            if image_embed is None:
                patch_count = 196  # 调整为您的实际patch数量
                image_embed = torch.zeros(batch_size, 1 + patch_count, self.image_dim, device=device)

            # 将生成的特征复制到适当的token位置
            for b in range(batch_size):
                if is_image_missing[b]:
                    # token_count = min(generated_features['image'].size(1), token_count)
                    # 将生成的前N个token替换到image_embed中
                    tokens_available = generated_features['image'].size(1)
                    max_tokens_to_copy = min(tokens_available, image_embed.size(1))

                    for t in range(max_tokens_to_copy):
                        if t < image_embed.size(1):  # 确保不超出范围
                            image_embed[b, t] = generated_features['image'][b, t]
        # 使用生成的特征替换缺失的模态

        if is_text_missing.any() and generated_features['text'] is not None:
            # 如果text_embed为None，创建新的
            if text_embed is None:
                # 替换硬编码的序列长度77为可配置值
                seq_len = self.max_length if hasattr(self, 'max_length') else 512  # RoBERTa默认最大长度为512
                text_embed = torch.zeros(batch_size, seq_len, self.text_dim, device=device)
                # 也需要创建注意力掩码
                if attention_mask is None:
                    attention_mask = torch.zeros(batch_size, seq_len, device=device)
                    attention_mask[:, :token_count] = 1  # 标记前token_count个token为有效

            # 将生成的特征复制到适当的token位置
            for b in range(batch_size):
                if is_text_missing[b]:
                    # token_count = min(generated_features['text'].size(1), token_count)
                    # 将生成的前N个token替换到text_embed中
                    tokens_available = generated_features['text'].size(1)
                    max_tokens_to_copy = min(tokens_available, text_embed.size(1))

                    for t in range(max_tokens_to_copy):
                        if t < text_embed.size(1):  # 确保不超出范围
                            text_embed[b, t] = generated_features['text'][b, t]
                            # 确保注意力掩码中对应token是有效的
                            if attention_mask is not None:
                                attention_mask[b, t] = 1


        return image_embed, text_embed, attention_mask, original_features, \
            (generated_features, reconstructed_features) if self.use_modality_generator else (None, None)

    def forward(self, image=None, input_ids=None, attention_mask=None, missing_type=None):
        """
        前向传播函数

        Args:
            image: 图像输入张量，完整
            input_ids: 文本输入ID，完整
            attention_mask: 文本注意力掩码，完整
            missing_type: 模态缺失类型 (none=0, image=1, text=2, both=3)
        """
        """
        前向传播函数，修改以接收原始特征用于重建
        """
        # 确保batch_size和device能够被正确识别
        if image is not None:
            batch_size = image.size(0)
            device = image.device
        elif input_ids is not None:
            batch_size = input_ids.size(0)
            device = input_ids.device
        else:
            raise ValueError("Both image and text inputs cannot be None simultaneously")

        # 提取原始特征（包括可见和不可见的模态）
        image_embed, text_embed = self._extract_features(image, input_ids, attention_mask)

        original_image_embed = image_embed.clone()
        original_text_embed = text_embed.clone()
        # 提取原始特征用于重建（确保即使模态缺失也有完整特征）

        # 处理模态（包括缺失模态的生成）
        image_embed, text_embed, attention_mask, original_features, (generated_features, reconstructed_features) \
            = (self._process_modalities
            (
            image_embed, text_embed, attention_mask, missing_type

        ))

        # 初始化提示
        image_prompt = self.image_init_prompt.expand(batch_size, -1, -1)  # [B, prompt_len, D_img]
        text_prompt = self.text_init_prompt.expand(batch_size, -1, -1)  # [B, prompt_len, D_txt]

        # 获取初步特征用于质量评估
        temp_img = image_embed.clone()
        temp_txt = text_embed.clone()

        for i in range(2):  # 使用前两层获取初步特征
            # 修改处理CLIP视觉层的部分（在forward方法中）
            if self.use_clip_encoders:
                # CLIP视觉模型需要特殊处理
                # 为CLIP层创建适当的注意力掩码
                img_seq_len = temp_img.shape[1]
                # 创建一个二维形状的掩码 [batch_size, seq_len]
                img_attention_mask = torch.ones((batch_size, img_seq_len), device=device)
                # 将掩码转换为CLIP注意力层期望的格式
                # 将形状为[batch_size, seq_len]的掩码转换为[batch_size, 1, seq_len, seq_len]
                extended_img_mask = self._prepare_clip_attention_mask(img_attention_mask)
                # CLIP的视觉层需要attention_mask和causal_attention_mask
                temp_img = self.image_blocks[i](
                    hidden_states=temp_img,
                    attention_mask=extended_img_mask,
                    causal_attention_mask=None,
                    output_attentions=False
                )[0]
            else:
                # 标准ViT层
                temp_img = self.image_blocks[i](temp_img)

            # 处理文本的注意力掩码
            extended_attention_mask = self._prepare_attention_mask(attention_mask, text_embed.shape[1])

            if self.encoder_type == 'clip' or self.use_clip_encoders:
                temp_txt = self.text_blocks[i](
                    temp_txt,
                    attention_mask=extended_attention_mask,
                    causal_attention_mask=None,
                    output_attentions=False
                )[0]
            elif self.encoder_type == 'roberta':
                temp_txt = self.text_blocks[i](
                    temp_txt,
                    attention_mask=extended_attention_mask
                )[0]

        # 计算模态质量（如果启用）
        quality_scores = None
        # if self.use_quality_prompt:
        #     # 使用CLS token进行质量评估
        #     img_cls_feat = temp_img[:, 0]  # [B, D_img]
        #     txt_cls_feat = temp_txt[:, 0]  # [B, D_txt]
        #     quality_scores = self.quality_estimator(img_cls_feat, txt_cls_feat, missing_type)

        if self.use_quality_prompt:
            # Update reference statistics when training
            img_cls_feat = temp_img[:, 0]  # [B, D_img]
            txt_cls_feat = temp_txt[:, 0]  # [B, D_txt]
            quality_scores = self.quality_estimator(img_cls_feat, txt_cls_feat, missing_type)
            self.quality_estimator.update_reference_statistics(
                image_embed[:, 0] if image_embed is not None else None,
                text_embed[:, 0] if text_embed is not None else None
            )

        # 跨层处理
        for i in range(self.prompt_depth):
            # 图像：拼接提示并处理
            img_with_prompt = torch.cat([image_prompt, image_embed], dim=1)
            # 在forward方法中，修改两处处理CLIP视觉层的代码（i < prompt_depth的情况）
            if self.use_clip_encoders:
                # CLIP视觉模型需要特殊处理
                # 为CLIP层创建适当的注意力掩码
                img_seq_len = img_with_prompt.shape[1]

                # 创建掩码并扩展维度
                img_attention_mask = torch.ones((batch_size, img_seq_len), device=device)
                extended_img_mask = self._prepare_clip_attention_mask(img_attention_mask)

                # CLIP的视觉层需要attention_mask和causal_attention_mask
                img_with_prompt = self.image_blocks[i](
                    hidden_states=img_with_prompt,
                    attention_mask=extended_img_mask,
                    causal_attention_mask=None,
                    output_attentions=False
                )[0]
            else:
                # 标准ViT层
                img_with_prompt = self.image_blocks[i](img_with_prompt)
            image_prompt, image_embed = img_with_prompt[:, :self.image_prompt_len], img_with_prompt[:,
                                                                                    self.image_prompt_len:]

            # 文本：拼接提示并处理
            txt_with_prompt = torch.cat([text_prompt, text_embed], dim=1)
            extended_mask = torch.cat([
                torch.ones(batch_size, self.text_prompt_len, device=attention_mask.device),
                attention_mask
            ], dim=1)
            extended_attention_mask = self._prepare_attention_mask(extended_mask, txt_with_prompt.shape[1])

            if self.encoder_type == 'clip':
                txt_with_prompt = self.text_blocks[i](
                    txt_with_prompt,
                    attention_mask=extended_attention_mask,
                    causal_attention_mask=None,
                    output_attentions=False
                )[0]
            elif self.encoder_type == 'roberta':
                txt_with_prompt = self.text_blocks[i](
                    txt_with_prompt,
                    attention_mask=extended_attention_mask
                )[0]

            text_prompt, text_embed = txt_with_prompt[:, :self.text_prompt_len], txt_with_prompt[:,
                                                                                 self.text_prompt_len:]

            # 跨模态交互更新提示
            if i < len(self.cross_modal_layer):
                # 从当前层特征更新提示
                # image_prompt = self.cross_modal_layer[i](image_prompt, text_embed)
                # text_prompt = self.cross_modal_layer[i](text_prompt, image_embed)
                image_prompt = self.image_InterPrompt_layer[i](image_prompt, image_embed)
                text_prompt = self.text_InterPrompt_layer[i](text_prompt, text_embed)

        # 继续处理剩余的Transformer层
        for i in range(self.prompt_depth, len(self.image_blocks)):
            if self.use_clip_encoders:
                # CLIP视觉模型需要特殊处理
                img_seq_len = image_embed.shape[1]

                # 创建掩码并扩展维度
                img_attention_mask = torch.ones((batch_size, img_seq_len), device=device)
                extended_img_mask = self._prepare_clip_attention_mask(img_attention_mask)

                # CLIP的视觉层需要attention_mask和causal_attention_mask
                image_embed = self.image_blocks[i](
                    hidden_states=image_embed,
                    attention_mask=extended_img_mask,
                    causal_attention_mask=None,
                    output_attentions=False
                )[0]
            else:
                # 标准ViT层
                image_embed = self.image_blocks[i](image_embed)

            extended_attention_mask = self._prepare_attention_mask(attention_mask, text_embed.shape[1])
            if self.encoder_type == 'clip':
                text_embed = self.text_blocks[i](
                    text_embed,
                    attention_mask=extended_attention_mask,
                    causal_attention_mask=None,
                    output_attentions=False
                )[0]
            elif self.encoder_type == 'roberta':
                text_embed = self.text_blocks[i](
                    text_embed,
                    attention_mask=extended_attention_mask
                )[0]


        # 应用最终的层归一化
        image_embed = self.image_norm(image_embed)
        text_embed = self.text_norm(text_embed)

        # print(image_embed.shape,text_embed.shape)
        # 提取CLS token特征
        image_feat = image_embed[:, 0]  # [B, D_img]
        text_feat = text_embed[:, 0]  # [B, D_txt]

        # 在应用质量引导的特征融合之前，打印维度信息
        # print(f"DEBUG - Feature dimensions before fusion:")
        # print(f"  image_feat shape: {image_feat.shape}")
        # print(f"  text_feat shape: {text_feat.shape}")
        # if quality_scores is not None:
        #     print(f"  quality_scores image: {quality_scores['image']['final_score'].shape}")
        #     print(f"  quality_scores text: {quality_scores['text']['final_score'].shape}")
        #     print(f"  quality_scores consistency: {quality_scores['cross_consistency'].shape}")

        # 质量引导的特征融合（如果启用）
        fusion_weights = None
        quality_guided_feat = None
        if self.use_quality_prompt and self.use_cross_modal_prompt:
            # print(f"  Calling feature_fusion with image_feat: {image_feat.shape}, text_feat: {text_feat.shape}")
            quality_guided_feat, fusion_weights = self.feature_fusion(image_feat, text_feat, quality_scores)
            # print(f"  After fusion: quality_guided_feat: {quality_guided_feat.shape}, fusion_weights: {fusion_weights.shape}")

        # Prepare features for concatenation
        features_to_concat = [image_feat, text_feat, F.one_hot(missing_type, num_classes=4).float()]

        # Add quality scores if enabled
        if self.use_quality_prompt:
            features_to_concat.extend([
                quality_scores['image']['final_score'],
                quality_scores['text']['final_score'],
                quality_scores['image']['quality'],
                quality_scores['text']['quality'],
                quality_scores['cross_consistency']
            ])

        # Concatenate all features
        fused = torch.cat(features_to_concat, dim=1)
        base_hidden = self.fusion(fused)
        hidden = None
        if quality_guided_feat is not None:
            alpha = 0.5  # Fixed weight or learnable parameter
            hidden = alpha * base_hidden + (1 - alpha) * quality_guided_feat
        logits = self.classifier(hidden)

        # 返回额外信息用于分析和损失计算
        additional_info = {
            'quality_scores': quality_scores,
            'fusion_weights': fusion_weights,
            'generated_modalities': {
                'image': image is None,  # 标记图像是否为生成的
                'text': input_ids is None  # 标记文本是否为生成的
            },
            'original_features': original_features,  # 原始特征（用于重建损失）
            'generated_features': generated_features,  # 生成的特征
            'reconstructed_features': reconstructed_features  # 重建的特征
        }

        return (logits, additional_info)
        # return logits if not self.training else (logits, additional_info)




# Create a factory function to initialize the model
# 修改create_multimodal_prompt_model函数以支持不同的image_size和patch_size
def create_multimodal_prompt_model(
        image_model_name='vit_base_patch16_224',
        text_model_name='roberta-base',
        image_prompt_len=5,
        text_prompt_len=5,
        prompt_depth=6,
        fusion_dim=512,
        num_classes=101,
        freeze_image_encoder=False,
        freeze_text_encoder=False,
        use_quality_prompt=True,
        use_cross_modal_prompt=True,
        image_size=224,
        patch_size=16,
        max_length=512,
        encoder_type='clip'
):
    # 检查是否两个模型名称相同且包含"clip"关键字
    use_clip_encoders = False

    # CLIP模式 - 当图像和文本模型名称相同且包含"clip"
    if "clip" in image_model_name.lower() and image_model_name == text_model_name:
        use_clip_encoders = True
        print(f"Detected matching CLIP model names. Using unified CLIP encoders: {image_model_name}")

        from transformers import CLIPVisionModel, CLIPTextModel

        # 加载CLIP模型组件
        vision_encoder = CLIPVisionModel.from_pretrained(image_model_name)
        text_encoder = CLIPTextModel.from_pretrained(text_model_name)

        print("CLIP vision model:", vision_encoder.__class__.__name__)
        print("CLIP text model:", text_encoder.__class__.__name__)

        # 创建并返回使用CLIP编码器的模型
        return MultimodalPromptModel(
            image_encoder_backbone=vision_encoder,
            text_encoder_backbone=text_encoder,
            image_prompt_len=image_prompt_len,
            text_prompt_len=text_prompt_len,
            prompt_depth=prompt_depth,
            fusion_dim=fusion_dim,
            num_classes=num_classes,
            freeze_image_encoder=freeze_image_encoder,
            freeze_text_encoder=freeze_text_encoder,
            use_quality_prompt=use_quality_prompt,
            use_cross_modal_prompt=use_cross_modal_prompt,
            max_length=max_length,
            encoder_type=encoder_type,
            use_clip_encoders=True
        )
    # 标准模式 - 不同的编码器
    # 初始化图像编码器骨干网络
    if "patch" in image_model_name and not "clip" in image_model_name.lower():
        # 处理标准ViT模型
        base_name_parts = image_model_name.split("_")
        model_type = base_name_parts[0]  # vit
        model_size = base_name_parts[1]  # base
        orig_patch_size = int(base_name_parts[2].replace("patch", ""))  # 16

        # 创建新的模型名称
        new_model_name = f"{model_type}_{model_size}_patch{patch_size}_{image_size}"
        image_encoder = create_model(new_model_name, pretrained=True, num_classes=0)
    elif "clip" in image_model_name.lower() and not use_clip_encoders:
        # 单独使用CLIP视觉模型
        from transformers import CLIPVisionModel
        image_encoder = CLIPVisionModel.from_pretrained(image_model_name)
    else:
        # 其他类型的图像模型
        image_encoder = create_model(image_model_name, pretrained=True, num_classes=0)

    # 初始化文本编码器
    if "clip" in text_model_name.lower() and not use_clip_encoders:
        # 单独使用CLIP文本模型
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained(text_model_name)
    elif "roberta" in text_model_name.lower():
        # RoBERTa模型
        from transformers import RobertaModel
        text_encoder = RobertaModel.from_pretrained(text_model_name)
    else:
        # 默认使用CLIPTextModel
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained(text_model_name)

    # 创建并返回完整的多模态模型
    return MultimodalPromptModel(
        image_encoder_backbone=image_encoder,
        text_encoder_backbone=text_encoder,
        image_prompt_len=image_prompt_len,
        text_prompt_len=text_prompt_len,
        prompt_depth=prompt_depth,
        fusion_dim=fusion_dim,
        num_classes=num_classes,
        freeze_image_encoder=freeze_image_encoder,
        freeze_text_encoder=freeze_text_encoder,
        use_quality_prompt=use_quality_prompt,
        use_cross_modal_prompt=use_cross_modal_prompt,
        max_length=max_length,
        encoder_type=encoder_type,
        use_clip_encoders=use_clip_encoders
    )
