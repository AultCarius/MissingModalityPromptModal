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

# from .modality_generator import CycleGenerationModel, CrossModalGenerator
# from models.improved_modality_generator import CycleGenerationModel
from models.improved_modality_generator import CycleGenerationModel
from models.modality_generator import EnhancedCycleGenerationModel

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

        from utils.analysis import FusionAnalyzer
        self.fusion_analyzer = FusionAnalyzer(save_dir='fusion_analysis_results')

        self.max_length = max_length
        self.encoder_type = encoder_type.lower()
        self.use_clip_encoders = use_clip_encoders

        # 保存编码器
        self.image_encoder = image_encoder_backbone
        self.text_encoder = text_encoder_backbone

        # 处理CLIP模式
        if use_clip_encoders:
            # 获取维度信息
            self.image_dim = self.image_encoder.config.hidden_size
            self.text_dim = self.text_encoder.config.hidden_size

            # 获取CLIP的投影维度
            if hasattr(self.image_encoder.config, 'projection_dim'):
                self.clip_projection_dim = self.image_encoder.config.projection_dim
            else:
                self.clip_projection_dim = 512  # CLIP默认投影维度

            print(f"Using CLIP encoders - Image dim: {self.image_dim}, Text dim: {self.text_dim}, Projection dim: {self.clip_projection_dim}")

            # 根据CLIP模型的真实结构设置组件
            # CLIP视觉模型 - 修正访问路径
            if hasattr(self.image_encoder, 'vision_model'):
                # 如果是完整的CLIP模型
                self.image_patch_embed = self.image_encoder.vision_model.embeddings
                self.image_blocks = self.image_encoder.vision_model.encoder.layers
                self.image_norm = self.image_encoder.vision_model.post_layernorm
                self.vision_projection = self.image_encoder.visual_projection if hasattr(self.image_encoder,
                                                                                         'visual_projection') else None
            elif hasattr(self.image_encoder, 'embeddings'):
                # 如果是CLIPVisionModel
                self.image_patch_embed = self.image_encoder.embeddings
                self.image_blocks = self.image_encoder.encoder.layers
                self.image_norm = self.image_encoder.post_layernorm
                self.vision_projection = nn.Linear(self.image_dim, self.clip_projection_dim, bias=False)
                # 使用正交初始化
                nn.init.orthogonal_(self.vision_projection.weight)
            else:
                raise ValueError("Unsupported CLIP vision model structure")

            # CLIP文本模型 - 修正访问路径
            if hasattr(self.text_encoder, 'text_model'):
                self.text_embeddings = self.text_encoder.text_model.embeddings
                self.text_blocks = self.text_encoder.text_model.encoder.layers
                self.text_norm = self.text_encoder.text_model.final_layer_norm
                # 获取text投影层
                self.text_projection = self.text_encoder.text_projection if hasattr(self.text_encoder, 'text_projection') else None
            elif hasattr(self.text_encoder, 'embeddings'):
                self.text_embeddings = self.text_encoder.embeddings
                self.text_blocks = self.text_encoder.encoder.layers
                self.text_norm = self.text_encoder.final_layer_norm
                # 如果是单独的text model，可能没有投影层，需要创建
                self.text_projection = nn.Linear(self.text_dim, self.clip_projection_dim, bias=False)
                # 使用正交初始化
                nn.init.orthogonal_(self.text_projection.weight)

            # 创建兼容接口
            self.image_cls_token = nn.Parameter(torch.zeros(1, 1, self.image_dim))
            # CLIP vision模型的position embeddings数量
            if hasattr(self.image_encoder, 'vision_model'):
                pos_embed_size = self.image_encoder.vision_model.embeddings.position_embedding.num_embeddings
            else:
                pos_embed_size = self.image_encoder.embeddings.position_embedding.num_embeddings

            self.image_pos_embed = nn.Parameter(torch.zeros(1, pos_embed_size, self.image_dim))
            self.image_pos_drop = nn.Dropout(0.0)

            # 初始化position embeddings
            nn.init.trunc_normal_(self.image_pos_embed, std=0.02)
            nn.init.trunc_normal_(self.image_cls_token, std=0.02)

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
            # modality_dims = {
            #     'image': self.image_dim,
            #     'text': self.text_dim
            # }
            modality_dims = {
                'image': self.clip_projection_dim ,
                'text': self.clip_projection_dim
            }

            # self.modality_generator = CycleGenerationModel(modality_dims, fusion_hidden_dim=fusion_dim)

            self.modality_generator = EnhancedCycleGenerationModel(
                modality_dims,
                fusion_hidden_dim=fusion_dim,
                num_layers=3,  # Can be configured via a parameter
                num_heads=4  # Can be configured via a parameter
            )





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
        fusion_input_dim = self.clip_projection_dim *2 + 4  # 基础特征 + 缺失类型
        # fusion_input_dim = self.image_dim + self.text_dim + 4  # 基础特征 + 缺失类型
        if use_quality_prompt:
            fusion_input_dim += 13  # 增加质量评分（2个质量特征5+3个详细评分1）

        # # 使用增强的质量评估器替换原有评估器
        # if use_quality_prompt:
        #     self.quality_estimator = EnhancedModalityQualityEstimator(self.image_dim, self.text_dim)
        #
        # # 使用质量引导的特征融合替换跨模态提示
        # if use_cross_modal_prompt:
        #     from models.quality_aware_prompting import ImprovedQualityAwareFeatureFusion
        #     self.feature_fusion = ImprovedQualityAwareFeatureFusion(
        #         self.image_dim,
        #         self.text_dim,
        #         fusion_dim
        #     )



        # 标记使用CLIP投影
        self.use_clip_projections = True

        # 更新融合维度设置，使用CLIP投影维度
        if use_quality_prompt:
            # 质量评估器也需要使用投影后的维度
            self.quality_estimator = EnhancedModalityQualityEstimator(
                self.clip_projection_dim, self.clip_projection_dim
            )

        # 特征融合也使用投影维度
        if use_cross_modal_prompt:
            from models.quality_aware_prompting import ImprovedQualityAwareFeatureFusion
            self.feature_fusion = ImprovedQualityAwareFeatureFusion(
                self.clip_projection_dim,
                self.clip_projection_dim,
                fusion_dim
            )

        print("fusion_input_dim=", fusion_input_dim)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

        # self.image_classifier = nn.Sequential(
        #     nn.Linear(self.image_dim, fusion_dim),
        #     nn.LayerNorm(fusion_dim),
        #     nn.GELU(),
        #     nn.Dropout(0.25),  # Higher dropout for image path
        #     nn.Linear(fusion_dim, num_classes)
        # )
        #
        # # Text-specific classifier
        # # 2. 设计更强大的文本分类器
        # self.text_classifier = nn.Sequential(
        #     nn.Linear(self.text_dim, fusion_dim * 2),  # 更宽的隐藏层
        #     nn.LayerNorm(fusion_dim * 2),
        #     nn.GELU(),
        #     nn.Dropout(0.1),  # 降低dropout以保留更多信息
        #     nn.Linear(fusion_dim * 2, fusion_dim),
        #     nn.GELU(),
        #     nn.LayerNorm(fusion_dim),
        #     nn.Linear(fusion_dim, num_classes)
        # )

        # Modality fusion weights learner
        self.fusion_weight_net = nn.Sequential(
            nn.Linear(4, 64),  # Input: missing_type one-hot
            nn.ReLU(),
            nn.Linear(64, 2),  # Output: [image_weight, text_weight]
            nn.Softmax(dim=1)  # Ensure weights sum to 1
        )
        self.text_feat_enhancer = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.text_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.text_dim, self.text_dim),
            nn.LayerNorm(self.text_dim)
        )
        # 初始化文本分类器最后一层的偏置，使其更容易产生正向预测
        # with torch.no_grad():
        #     # 获取最后一层
        #     last_layer = None
        #     for module in reversed(self.text_classifier):
        #         if isinstance(module, nn.Linear):
        #             last_layer = module
        #             break
        #
        #     if last_layer is not None:
        #         # 将偏置初始化为小正值，促进正向预测
        #         last_layer.bias.fill_(0.1)  # 使用0.1作为初始偏置




        self.image_feat_norm = nn.LayerNorm(self.clip_projection_dim )
        self.text_feat_norm = nn.LayerNorm(self.clip_projection_dim )
        self.base_fusion_norm = nn.LayerNorm(fusion_dim)
        self.quality_fusion_norm = nn.LayerNorm(fusion_dim)


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

    def apply_clip_projections(self, image_feat, text_feat):
        """
        应用CLIP投影层，将图像和文本特征投影到统一的语义空间

        Args:
            image_feat: 图像特征 [batch_size, image_dim]
            text_feat: 文本特征 [batch_size, text_dim]

        Returns:
            projected_image_feat: 投影后的图像特征 [batch_size, projection_dim]
            projected_text_feat: 投影后的文本特征 [batch_size, projection_dim]
        """
        projected_image_feat = None
        projected_text_feat = None

        if self.use_clip_projections:
            # 应用图像投影
            if image_feat is not None and self.vision_projection is not None:
                projected_image_feat = self.vision_projection(image_feat)
                # 归一化（CLIP的标准做法）
                projected_image_feat = F.normalize(projected_image_feat, p=2, dim=-1)

            # 应用文本投影
            if text_feat is not None and self.text_projection is not None:
                projected_text_feat = self.text_projection(text_feat)
                # 归一化（CLIP的标准做法）
                projected_text_feat = F.normalize(projected_text_feat, p=2, dim=-1)
        else:
            # 不使用CLIP投影时，直接返回原特征
            projected_image_feat = image_feat
            projected_text_feat = text_feat

        return projected_image_feat, projected_text_feat

    def analyze_feature_distributions(self, original_features, generated_features, reconstructed_features, missing_type,
                                      step="before_processing"):
        """
        Analyze and print statistics about feature distributions for real, generated, and reconstructed features.

        Args:
            original_features: Dictionary of original features for each modality
            generated_features: Dictionary of generated features for each modality
            reconstructed_features: Dictionary of reconstructed features for each modality
            missing_type: Tensor indicating missing modality types
            step: String indicating when this analysis is being performed
        """
        batch_size = missing_type.size(0)
        device = missing_type.device

        # Convert missing_type to boolean masks
        is_image_missing = (missing_type == 1) | (missing_type == 3)
        is_text_missing = (missing_type == 2) | (missing_type == 3)

        print(f"\n===== Feature Distribution Analysis ({step}) =====")
        print(f"Batch size: {batch_size}, Image missing: {is_image_missing.sum().item()}/{batch_size}, "
              f"Text missing: {is_text_missing.sum().item()}/{batch_size}")

        # Analyze image features
        print("\n----- Image Features -----")
        self._analyze_modality_features(
            original_features.get('image'),
            None if generated_features is None else generated_features.get('image'),
            None if reconstructed_features is None else reconstructed_features.get('image'),
            is_image_missing,
            modality="image"
        )

        # Analyze text features
        print("\n----- Text Features -----")
        self._analyze_modality_features(
            original_features.get('text'),
            None if generated_features is None else generated_features.get('text'),
            None if reconstructed_features is None else reconstructed_features.get('text'),
            is_text_missing,
            modality="text"
        )

        print("===== End of Analysis =====\n")

    def analyze_features_at_key_points(self, step_name, image_features=None, text_features=None):
        """
        分析特定步骤的特征分布

        Args:
            step_name: 步骤名称
            image_features: 图像特征
            text_features: 文本特征
        """
        print(f"\n===== Feature Analysis at {step_name} =====")

        if image_features is not None and image_features.numel() > 0:
            # 计算图像特征统计信息
            img_mean = image_features.mean().item()
            img_std = image_features.std().item()
            img_min = image_features.min().item()
            img_max = image_features.max().item()
            img_abs_mean = image_features.abs().mean().item()
            img_norm = torch.norm(image_features, dim=1).mean().item() if image_features.dim() > 1 else torch.norm(
                image_features).item()

            print(f"Image features:")
            print(f"  Shape: {list(image_features.shape)}")
            print(f"  Mean: {img_mean:.6f}, Std: {img_std:.6f}")
            print(f"  Min: {img_min:.6f}, Max: {img_max:.6f}")
            print(f"  Abs Mean: {img_abs_mean:.6f}, L2 Norm: {img_norm:.6f}")

        if text_features is not None and text_features.numel() > 0:
            # 计算文本特征统计信息
            txt_mean = text_features.mean().item()
            txt_std = text_features.std().item()
            txt_min = text_features.min().item()
            txt_max = text_features.max().item()
            txt_abs_mean = text_features.abs().mean().item()
            txt_norm = torch.norm(text_features, dim=1).mean().item() if text_features.dim() > 1 else torch.norm(
                text_features).item()

            print(f"Text features:")
            print(f"  Shape: {list(text_features.shape)}")
            print(f"  Mean: {txt_mean:.6f}, Std: {txt_std:.6f}")
            print(f"  Min: {txt_min:.6f}, Max: {txt_max:.6f}")
            print(f"  Abs Mean: {txt_abs_mean:.6f}, L2 Norm: {txt_norm:.6f}")

        # 如果两者都存在，比较它们
        if image_features is not None and text_features is not None and image_features.numel() > 0 and text_features.numel() > 0:
            print(f"\nComparison (Image vs Text):")
            print(f"  Std ratio: {img_std / txt_std:.4f}x")
            print(f"  Abs Mean ratio: {img_abs_mean / txt_abs_mean:.4f}x")
            print(f"  L2 Norm ratio: {img_norm / txt_norm:.4f}x")
            print(f"  Range ratio: {(img_max - img_min) / (txt_max - txt_min):.4f}x")

        print("========================================\n")

    def _analyze_modality_features(self, original, generated, reconstructed, missing_mask, modality):
        """
        Helper method to analyze features for a specific modality.

        Args:
            original: Original features tensor
            generated: Generated features tensor
            reconstructed: Reconstructed features tensor
            missing_mask: Boolean mask indicating which samples have this modality missing
            modality: String indicating which modality ("image" or "text")
        """
        # Check if features exist
        has_original = original is not None and original.numel() > 0
        has_generated = generated is not None and generated.numel() > 0
        has_reconstructed = reconstructed is not None and reconstructed.numel() > 0

        # Check for NaN or Inf values
        def check_invalid(tensor, name):
            if tensor is None:
                return False
            has_nan = torch.isnan(tensor).any().item()
            has_inf = torch.isinf(tensor).any().item()
            if has_nan or has_inf:
                print(f"WARNING: {name} has {'NaN' if has_nan else 'Inf'} values!")
            return has_nan or has_inf

        # Compute statistics for a tensor
        def compute_stats(tensor, flatten=True):
            if tensor is None:
                return {}

            # Flatten the tensor for consistent analysis
            if flatten and tensor.dim() > 2:
                tensor = tensor.reshape(tensor.size(0), -1)

            # Basic statistics
            stats = {
                "mean": tensor.mean().item(),
                "std": tensor.std().item(),
                "min": tensor.min().item(),
                "max": tensor.max().item(),
                "abs_mean": tensor.abs().mean().item(),
                "zeros_percent": (tensor == 0).float().mean().item() * 100,
                "shape": list(tensor.shape)
            }

            # Additional statistics
            with torch.no_grad():
                # Compute L2 norm (feature magnitude)
                if tensor.dim() > 1:
                    stats["norm_mean"] = torch.norm(tensor, dim=1).mean().item()

                # Check for activation patterns
                stats["positive_percent"] = (tensor > 0).float().mean().item() * 100
                stats["negative_percent"] = (tensor < 0).float().mean().item() * 100

                # Compute histogram for distribution visualization
                hist_vals = tensor.flatten().cpu().numpy()
                try:
                    from numpy import histogram
                    hist, bins = histogram(hist_vals, bins=10)
                    stats["histogram"] = {
                        "counts": hist.tolist(),
                        "bins": bins.tolist()
                    }
                except:
                    stats["histogram"] = None

            return stats

        # Extract non-missing samples from original features
        if has_original and not missing_mask.all():
            real_samples = original[~missing_mask] if missing_mask.any() else original
            real_stats = compute_stats(real_samples)
            print(f"Real {modality} features (non-missing):")
            print(f"  Shape: {real_stats['shape']}")
            print(f"  Mean: {real_stats['mean']:.6f}, Std: {real_stats['std']:.6f}")
            print(f"  Min: {real_stats['min']:.6f}, Max: {real_stats['max']:.6f}")
            print(f"  Abs Mean: {real_stats['abs_mean']:.6f}, L2 Norm Mean: {real_stats.get('norm_mean', 'N/A')}")
            print(
                f"  Positive: {real_stats['positive_percent']:.2f}%, Negative: {real_stats['negative_percent']:.2f}%, Zeros: {real_stats['zeros_percent']:.2f}%")

            # Check for invalid values
            check_invalid(real_samples, f"Real {modality} features")

        # Extract generated features statistics
        if has_generated and missing_mask.any():
            gen_samples = generated[missing_mask] if generated.shape[0] > 1 else generated
            gen_stats = compute_stats(gen_samples)
            print(f"Generated {modality} features:")
            print(f"  Shape: {gen_stats['shape']}")
            print(f"  Mean: {gen_stats['mean']:.6f}, Std: {gen_stats['std']:.6f}")
            print(f"  Min: {gen_stats['min']:.6f}, Max: {gen_stats['max']:.6f}")
            print(f"  Abs Mean: {gen_stats['abs_mean']:.6f}, L2 Norm Mean: {gen_stats.get('norm_mean', 'N/A')}")
            print(
                f"  Positive: {gen_stats['positive_percent']:.2f}%, Negative: {gen_stats['negative_percent']:.2f}%, Zeros: {gen_stats['zeros_percent']:.2f}%")

            # Check for invalid values
            check_invalid(gen_samples, f"Generated {modality} features")

            # Compare with real features if both exist
            if has_original and not missing_mask.all():
                print(f"  Distribution comparison (Generated vs Real):")
                print(f"    Mean ratio: {gen_stats['mean'] / real_stats['mean']:.4f} (vs 1.0 ideal)")
                print(f"    Std ratio: {gen_stats['std'] / real_stats['std']:.4f} (vs 1.0 ideal)")
                print(f"    Abs Mean ratio: {gen_stats['abs_mean'] / real_stats['abs_mean']:.4f} (vs 1.0 ideal)")
                if 'norm_mean' in gen_stats and 'norm_mean' in real_stats:
                    print(f"    Norm ratio: {gen_stats['norm_mean'] / real_stats['norm_mean']:.4f} (vs 1.0 ideal)")

        # Extract reconstructed features statistics
        if has_reconstructed:
            # Get statistics for all reconstructed samples
            recon_stats = compute_stats(reconstructed)
            print(f"Reconstructed {modality} features:")
            print(f"  Shape: {recon_stats['shape']}")
            print(f"  Mean: {recon_stats['mean']:.6f}, Std: {recon_stats['std']:.6f}")
            print(f"  Min: {recon_stats['min']:.6f}, Max: {recon_stats['max']:.6f}")
            print(f"  Abs Mean: {recon_stats['abs_mean']:.6f}")

            # Compare with original features if both exist
            if has_original:
                # Compute reconstruction error
                if original.shape == reconstructed.shape:
                    recon_error = torch.nn.functional.mse_loss(reconstructed, original).item()
                    print(f"  Reconstruction MSE: {recon_error:.6f}")

    # 添加到MultimodalPromptModel类中
    def compute_feature_consistency_loss(self, original_features, reconstructed_features):
        """计算特征一致性损失"""
        consistency_loss = 0.0

        # 检查图像特征一致性
        if 'image' in original_features and 'image' in reconstructed_features:
            # 原始和重建特征都存在
            if original_features['image'] is not None and reconstructed_features['image'] is not None:
                # 计算均方误差损失
                img_loss = F.mse_loss(
                    reconstructed_features['image'],
                    original_features['image']
                )

                # 图像特征一致性损失加权1.5倍 - 增强对图像缺失的处理
                consistency_loss += 1.5 * img_loss

        # 检查文本特征一致性
        if 'text' in original_features and 'text' in reconstructed_features:
            # 原始和重建特征都存在
            if original_features['text'] is not None and reconstructed_features['text'] is not None:
                # 计算均方误差损失
                txt_loss = F.mse_loss(
                    reconstructed_features['text'],
                    original_features['text']
                )

                # 文本特征一致性损失正常权重
                consistency_loss += txt_loss

        return consistency_loss

    def _prepare_clip_image_attention_mask(self, attention_mask, seq_len):
        """
        为CLIP视觉Transformer准备注意力掩码（非causal）
        Args:
            attention_mask: [batch_size, seq_len]，通常全为1
            seq_len: token数量（例如patch + cls + prompt）

        Returns:
            [batch_size, 1, seq_len, seq_len] attention mask
        """
        batch_size = attention_mask.shape[0]
        # full attention with valid tokens
        mask = attention_mask.unsqueeze(1).unsqueeze(2).repeat(1, 1, seq_len, 1)  # [B,1,S,S]
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)
        return mask

    def _prepare_clip_text_attention_mask(self, attention_mask, seq_len):
        """
        为CLIP文本Transformer准备注意力掩码（非causal）
        Args:
            attention_mask: [batch_size, seq_len]，1表示有效token，0表示padding
            seq_len: token数量（包括prompt或eot）

        Returns:
            [batch_size, 1, 1, seq_len] attention mask，用于广播
        """
        extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B,1,1,S]
        extended_mask = extended_mask.float().masked_fill(extended_mask == 0, float("-inf")).masked_fill(
            extended_mask == 1, 0.0)
        return extended_mask

    def _prepare_causal_attention_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        构造 causal attention mask，用于阻止每个位置看到它右边的 token。

        Args:
            seq_len (int): 输入序列长度（prompt + text）
            device (torch.device): 设备

        Returns:
            causal_mask: [1, 1, seq_len, seq_len] 的 mask，float 类型，填充为 0 或 -inf
        """
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.float32, device=device) * float('-inf'),
            diagonal=1
        )  # 上三角为 -inf，下三角为 0
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        return causal_mask

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

    def _prepare_attention_mask_clip(self, attention_mask, hidden_states):
        """为CLIP文本编码器准备注意力掩码"""
        if attention_mask is None:
            return None

        mask = torch.empty(self.context_length + self.prompt_length,
                           self.context_length + self.prompt_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    def _prepare_clip_attention_mask(self, attention_mask):
        """
        为CLIP视觉模型准备注意力掩码
        # todo:修改掩码

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

    def _extract_features(self, image, input_ids, attention_mask, missing_type=None):
        """提取图像和文本的原始特征，正确处理零填充的缺失模态
            对于缺失的,模态保持零填充,而不是使用编码器去处理这个零填充的特征.
        """
        batch_size = image.size(0) if image is not None else input_ids.size(0)
        device = next(self.parameters()).device

        # 检测缺失模态 - 基于missing_type参数和零填充检测
        if missing_type is not None:
            is_image_missing = (missing_type == 1) | (missing_type == 3)
            is_text_missing = (missing_type == 2) | (missing_type == 3)
        else:
            # 如果未提供missing_type，通过检测零填充来识别缺失模态
            is_image_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)
            is_text_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)

            if image is not None:
                is_image_missing = torch.sum(torch.abs(image.view(batch_size, -1)), dim=1) < 1e-6
            if input_ids is not None:
                is_text_missing = torch.sum(torch.abs(input_ids.float()), dim=1) < 1e-6

        # 处理图像特征 (如果存在且不是缺失的)
        image_embed = torch.zeros(batch_size, 197, self.image_dim, device=device)  # 默认零嵌入
        if image is not None and not is_image_missing.all():
            # 只处理非缺失样本
            non_missing = ~is_image_missing
            if non_missing.any():
                non_missing_images = image[non_missing]

                if self.use_clip_encoders:
                    try:
                        # 使用CLIP vision model的embeddings
                        if hasattr(self.image_encoder, 'vision_model'):
                            # 完整CLIP模型
                            embeddings = self.image_encoder.vision_model.embeddings
                        else:
                            # CLIPVisionModel
                            embeddings = self.image_encoder.embeddings

                        # 获取patch embeddings
                        non_missing_embeds = embeddings(pixel_values=non_missing_images)

                        # CLIP vision模型通常已经包含了CLS token和position embeddings
                        # 所以我们直接使用embeddings的输出

                    except Exception as e:
                        print(f"Error extracting CLIP vision embeddings: {e}")
                        # 创建备用的随机嵌入
                        seq_len = 197  # CLIP ViT-B/16的典型序列长度
                        non_missing_embeds = torch.randn(non_missing.sum(), seq_len, self.image_dim, device=device)
                else:
                    # 标准ViT模型处理（原有代码）
                    patch_embeds = self.image_patch_embed(non_missing_images)
                    cls_token = self.image_cls_token.expand(non_missing.sum(), -1, -1)
                    non_missing_embeds = torch.cat((cls_token, patch_embeds), dim=1)
                    non_missing_embeds = non_missing_embeds + self.image_pos_embed[:, :non_missing_embeds.size(1), :]
                    non_missing_embeds = self.image_pos_drop(non_missing_embeds)

                # 将非缺失样本的嵌入放回原始批次张量
                image_embed[non_missing] = non_missing_embeds

        # 处理文本特征 (如果存在且不是缺失的)
        text_embed = torch.zeros(batch_size, input_ids.size(1), self.text_dim, device=device)
        if input_ids is not None and not is_text_missing.all():
            try:
                if self.use_clip_encoders:
                    # 使用CLIP text model的embeddings
                    if hasattr(self.text_encoder, 'text_model'):
                        # 完整CLIP模型
                        all_text_embeds = self.text_encoder.text_model.embeddings(input_ids)
                    else:
                        # CLIPTextModel
                        all_text_embeds = self.text_encoder.embeddings(input_ids)
                else:
                    # 其他文本模型
                    all_text_embeds = self.text_embeddings(input_ids)

                # 将所有编码结果赋值
                text_embed = all_text_embeds

            except Exception as e:
                print(f"处理文本嵌入时出错: {str(e)}")
                text_embed = torch.zeros(batch_size, input_ids.size(1), self.text_dim, device=device)

        for b in range(batch_size):
            if is_text_missing[b]:
                # 为缺失的文本创建特殊的mask：只关注第一个token（CLS位置）
                attention_mask[b, 0] = 1  # 至少保证CLS token有效
                # 其他位置保持0，这样在后续生成时不会干扰

        return image_embed, text_embed, is_image_missing, is_text_missing

    def _process_modalities(self, image_embed, text_embed, attention_mask, missing_type):
        """处理模态，包括模态缺失的情况，并使用GAN生成缺失的模态特征"""
        batch_size = image_embed.size(0) if image_embed is not None else text_embed.size(0)
        device = next(self.parameters()).device
        missing_type = missing_type.to(device)

        # 确保missing_type形状正确并在正确的设备上
        if missing_type is None:
            # 如果未指定，假设所有样本都没有缺失模态
            missing_type = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(missing_type, int):
            # 如果是单个整数，扩展到批次大小
            missing_type = torch.full((batch_size,), missing_type, dtype=torch.long, device=device)
        elif missing_type.size(0) != batch_size:
            # 如果批次大小不匹配，这是一个严重错误
            raise ValueError(
                f"missing_type batch size ({missing_type.size(0)}) doesn't match data batch size ({batch_size})")

        # 检测缺失的模态
        is_image_missing = (missing_type == 1) | (missing_type == 3)
        is_text_missing = (missing_type == 2) | (missing_type == 3)

        # 提取原始特征的tokens用于重建损失计算和GAN训练
        # Track which modalities are detected as missing through zero check (for debugging)
        image_is_zeros = torch.sum(torch.abs(image_embed.view(batch_size, -1)), dim=1) < 1e-6
        text_is_zeros = torch.sum(torch.abs(text_embed.view(batch_size, -1)), dim=1) < 1e-6

        # Extract token features for reconstruction (first few tokens of each modality)
        token_count = 1  # Number of tokens to use per modality

        # Save original features for reconstruction loss
        original_features = {
            'image': image_embed[:, :token_count].reshape(batch_size, token_count, -1).detach(),
            'text': text_embed[:, :token_count].reshape(batch_size, token_count, -1).detach()
        }

        # Create masks to track which features are real (non-zero)
        original_features_mask = {
            'image': ~image_is_zeros.unsqueeze(1).expand(-1, token_count),
            'text': ~text_is_zeros.unsqueeze(1).expand(-1, token_count)
        }

        should_analyze_gen = (torch.rand(1).item() < 0.01)  # 1% chance to analyze
        if should_analyze_gen:
            self.analyze_feature_distributions(
                original_features,
                None,  # No generated features yet
                None,  # No reconstructed features yet
                missing_type,
                step="before_generation"
            )

        # Process modality generation
        generated_features = {'image': None, 'text': None}
        reconstructed_features = {'image': None, 'text': None}
        cycle_features = None

        if self.use_modality_generator:
            # Prepare input data
            gen_input = {
                'image': None if image_embed is None else image_embed[:, :token_count].detach(),
                'text': None if text_embed is None else text_embed[:, :token_count].detach()
            }

            # Flag for complete sample training
            generate_all = self.training and not (is_image_missing.any() or is_text_missing.any())

            # Use improved batch processing in the generator
            generated_features, reconstructed_features, _ = self.modality_generator(
                gen_input,
                missing_type,
                generate_all=generate_all
            )

        # Analyze features after generation
        if should_analyze_gen:
            self.analyze_feature_distributions(
                original_features,
                generated_features,
                reconstructed_features,
                missing_type,
                step="after_generation"
            )

        # Use generated features to replace missing modalities
        if is_image_missing.any() and generated_features.get('image') is not None:
            # If image_embed is None, create a new tensor
            if image_embed is None:
                patch_count = 196  # Adjust to actual patch count
                image_embed = torch.zeros(batch_size, 1 + patch_count, self.image_dim, device=device)

            # Copy generated features to appropriate positions
            for b in range(batch_size):
                if is_image_missing[b]:
                    gen_image = generated_features['image']
                    if gen_image.dim() == 3:  # [batch, tokens, dim]
                        tokens_available = gen_image.size(1)
                        max_tokens_to_copy = min(tokens_available, image_embed.size(1))

                        for t in range(max_tokens_to_copy):
                            if t < image_embed.size(1):
                                image_embed[b, t] = gen_image[b, t]
                    else:  # [batch, dim]
                        image_embed[b, 0] = gen_image[b]

        # 使用生成的特征替换缺失的模态
        if is_text_missing.any() and generated_features.get('text') is not None:
            # If text_embed is None, create a new tensor
            if text_embed is None:
                seq_len = 77  # Adjust to actual sequence length
                text_embed = torch.zeros(batch_size, seq_len, self.text_dim, device=device)
                attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
                # todo:检查这里的attention_mask

            # Copy generated features to appropriate positions
            for b in range(batch_size):
                if is_text_missing[b]:
                    gen_text = generated_features['text']
                    if gen_text.dim() == 3:  # [batch, tokens, dim]
                        tokens_available = gen_text.size(1)
                        max_tokens_to_copy = min(tokens_available, text_embed.size(1))

                        for t in range(max_tokens_to_copy):
                            if t < text_embed.size(1):
                                text_embed[b, t] = gen_text[b, t]
                                attention_mask[b, t] = 1  # Mark as attended
                    else:  # [batch, dim]
                        text_embed[b, 0] = gen_text[b]
                        attention_mask[b, 0] = 1  # Mark as attended

        # Prepare additional information for loss calculation and debugging
        additional_info = {
            'original_features': original_features,
            'generated_features': generated_features,
            'reconstructed_features': reconstructed_features,
            'cycle_features': cycle_features,
            'missing_type': missing_type,
            'is_image_missing': is_image_missing,
            'is_text_missing': is_text_missing,
            'original_features_mask': original_features_mask,
            'image_is_zeros': image_is_zeros,
            'text_is_zeros': text_is_zeros
        }

        return image_embed, text_embed, attention_mask, additional_info

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
        # 处理missing_type
        # 确保 missing_type 形状正确
        if missing_type is None:
            missing_type = torch.zeros(batch_size, dtype=torch.long, device=device)
        elif isinstance(missing_type, int):
            missing_type = torch.full((batch_size,), missing_type, dtype=torch.long, device=device)
        elif missing_type.size(0) != batch_size:
            self.logger.warning(f"missing_type 批大小不匹配: {missing_type.size(0)} vs {batch_size}")
            # 调整大小以匹配
            if missing_type.size(0) > batch_size:
                missing_type = missing_type[:batch_size]
            else:
                padding = torch.zeros(batch_size - missing_type.size(0), dtype=missing_type.dtype, device=device)
                missing_type = torch.cat([missing_type, padding], dim=0)

        # 提取特征，同时获取缺失掩码
        image_embed, text_embed, is_image_missing, is_text_missing = self._extract_features(
            image, input_ids, attention_mask, missing_type
        )

        # 1. 在提取特征后，分析原始特征分布
        # 创建一个随机采样检查标志，避免分析所有批次
        should_analyze = (torch.rand(1).item() < 0.001)  # 1%的批次进行分析
        shoule_analyze_fusion = (torch.rand(1).item() < 0.01)
        # 特征分布分析
        if should_analyze and image_embed is not None and text_embed is not None:
            # 获取CLS token
            img_cls = image_embed[:, 0].detach()
            txt_cls = text_embed[:, 0].detach()
            self.analyze_features_at_key_points("原始提取的特征", img_cls, txt_cls)

        original_image_embed = image_embed.clone() if image_embed is not None else None
        original_text_embed = text_embed.clone() if text_embed is not None else None

        # TODO: 这里暂时注释,改为晚期CLSTOKEN的重建
        # Process modalities - this includes generation of missing modalities
        # image_embed, text_embed, attention_mask, additional_info = self._process_modalities(
        #     image_embed, text_embed, attention_mask, missing_type
        # )


        # 初始化提示
        image_prompt = self.image_init_prompt.expand(batch_size, -1, -1)  # [B, prompt_len, D_img]
        text_prompt = self.text_init_prompt.expand(batch_size, -1, -1)  # [B, prompt_len, D_txt]

        # 获取初步特征用于质量评估
        temp_img = image_embed.clone()
        temp_txt = text_embed.clone()

        # Debug: check embeddings after processing
        if should_analyze:
            pre_img_feature = image_embed[:, 0].detach() if image_embed.size(1) > 0 else None
            pre_text_feature = text_embed[:, 0].detach() if text_embed.size(1) > 0 else None
            if pre_img_feature is not None and pre_text_feature is not None:
                self.analyze_features_at_key_points("模态处理后", pre_img_feature, pre_text_feature)

        # for i in range(2):  # 使用前两层获取初步特征
        #     # 修改处理CLIP视觉层的部分（在forward方法中）
        #     if self.use_clip_encoders:
        #         # CLIP视觉模型需要特殊处理
        #         # 为CLIP层创建适当的注意力掩码
        #         img_seq_len = temp_img.shape[1]
        #         # 创建一个二维形状的掩码 [batch_size, seq_len]
        #         img_attention_mask = torch.ones((batch_size, img_seq_len), device=device)
        #         # 将掩码转换为CLIP注意力层期望的格式
        #         # 将形状为[batch_size, seq_len]的掩码转换为[batch_size, 1, seq_len, seq_len]
        #         extended_img_mask = self._prepare_clip_image_attention_mask(img_attention_mask, img_seq_len)
        #         # extended_img_mask = self._prepare_clip_attention_mask(img_attention_mask)
        #
        #         temp_img = self.image_blocks[i](
        #             hidden_states=temp_img,
        #             attention_mask=extended_img_mask,
        #             causal_attention_mask=None,
        #             output_attentions=False
        #         )[0]
        #     else:
        #         # 标准ViT层
        #         temp_img = self.image_blocks[i](temp_img)
        #
        #     # 处理文本的注意力掩码
        #     # extended_attention_mask = self._prepare_attention_mask(attention_mask, text_embed.shape[1])
        #     extended_attention_mask = self._prepare_clip_text_attention_mask(attention_mask, text_embed.shape[1])
        #
        #     if self.encoder_type == 'clip' or self.use_clip_encoders:
        #         temp_txt = self.text_blocks[i](
        #             temp_txt,
        #             attention_mask=extended_attention_mask,
        #             causal_attention_mask=None,
        #             output_attentions=False
        #         )[0]
        #     elif self.encoder_type == 'roberta':
        #         temp_txt = self.text_blocks[i](
        #             temp_txt,
        #             attention_mask=extended_attention_mask
        #         )[0]

        # 2. 在应用Transformer层前
        if should_analyze:
            pre_trans_img = image_embed[:, 0].detach() if image_embed.size(1) > 0 else None
            pre_trans_txt = text_embed[:, 0].detach() if text_embed.size(1) > 0 else None
            if pre_trans_img is not None and pre_trans_txt is not None:
                self.analyze_features_at_key_points("Before transformer", pre_trans_img, pre_trans_txt)

        # 计算模态质量（如果启用）
        quality_scores = None

        # 跨层处理
        # 在跨层处理中修复CLIP层的调用
        # 在跨层处理中修复CLIP层的调用
        for i in range(self.prompt_depth):
            # 图像：拼接提示并处理
            img_with_prompt = torch.cat([image_prompt, image_embed], dim=1)

            if self.use_clip_encoders:
                # CLIP视觉模型需要特殊处理
                img_seq_len = img_with_prompt.shape[1]

                # 创建attention mask
                img_attention_mask = torch.ones((batch_size, img_seq_len), device=device)
                extended_img_mask = self._prepare_clip_image_attention_mask(img_attention_mask, img_seq_len)

                # CLIP的视觉Transformer层调用
                if hasattr(self.image_blocks[i], '__call__'):
                    # 对于CLIPVisionModel的层
                    img_with_prompt = self.image_blocks[i](
                        img_with_prompt,
                        attention_mask=extended_img_mask,
                        causal_attention_mask=None,
                        output_attentions=False
                    )[0]  # 只取hidden_states
                else:
                    # 备用方案
                    img_with_prompt = self.image_blocks[i](img_with_prompt)
            else:
                # 标准ViT层
                img_with_prompt = self.image_blocks[i](img_with_prompt)

            image_prompt, image_embed = img_with_prompt[:, :self.image_prompt_len], img_with_prompt[:,
                                                                                    self.image_prompt_len:]

            # 文本：拼接提示并处理
            txt_with_prompt = torch.cat([text_prompt, text_embed], dim=1)
            full_attention_mask = torch.cat([
                torch.ones(batch_size, self.text_prompt_len, device=attention_mask.device),  # 对应 prompt
                attention_mask  # 对应原始 text
            ], dim=1)  # [B, full_seq_len] = [B, 125]
            # 构建 extended mask 和 causal mask
            extended_mask = self._prepare_clip_text_attention_mask(full_attention_mask, txt_with_prompt.shape[1])

            # CLIP文本模型的层调用
            if self.use_clip_encoders:
                # 为CLIP text model准备causal attention mask
                seq_len = txt_with_prompt.size(1)
                causal_attention_mask = self._prepare_causal_attention_mask(seq_len,device=attention_mask.device)
                if hasattr(self.text_blocks[i], '__call__'):
                    # 对于CLIPTextModel的层
                    layer_output = self.text_blocks[i](
                        txt_with_prompt,
                        attention_mask=extended_mask,
                        causal_attention_mask=causal_attention_mask,
                        output_attentions=False
                    )
                    txt_with_prompt = layer_output[0]  # 只取hidden_states
                else:
                    # 备用方案
                    txt_with_prompt = self.text_blocks[i](txt_with_prompt)
            else:
                # 其他文本模型的处理
                extended_attention_mask = self._prepare_clip_text_attention_mask(extended_mask,
                                                                                 txt_with_prompt.shape[1])

                if self.encoder_type == 'roberta':
                    txt_with_prompt = self.text_blocks[i](
                        txt_with_prompt,
                        attention_mask=extended_attention_mask
                    )[0]

            text_prompt, text_embed = txt_with_prompt[:, :self.text_prompt_len], txt_with_prompt[:,
                                                                                 self.text_prompt_len:]

            # 跨模态交互更新提示
            if i < len(self.cross_modal_layer):
                image_prompt = self.image_InterPrompt_layer[i](image_prompt, image_embed)
                text_prompt = self.text_InterPrompt_layer[i](text_prompt, text_embed)

        # 继续处理剩余的Transformer层
        for i in range(self.prompt_depth, len(self.image_blocks)):
            if self.use_clip_encoders:
                # CLIP视觉模型需要特殊处理
                img_seq_len = image_embed.shape[1]

                # 创建掩码并扩展维度
                img_attention_mask = torch.ones((batch_size, img_seq_len), device=device)
                extended_img_mask = self._prepare_clip_image_attention_mask(img_attention_mask, img_seq_len)
                # extended_img_mask = self._prepare_clip_attention_mask(img_attention_mask)
                causal_attention_mask = self._prepare_causal_attention_mask(img_seq_len, device=attention_mask.device)

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

            extended_attention_mask = self._prepare_clip_text_attention_mask(attention_mask, text_embed.shape[1])
            causal_attention_mask = self._prepare_causal_attention_mask(text_embed.shape[1], device=attention_mask.device)
            if self.encoder_type == 'clip':
                text_embed = self.text_blocks[i](
                    text_embed,
                    attention_mask=extended_attention_mask,
                    causal_attention_mask=causal_attention_mask,
                    output_attentions=False
                )[0]
            elif self.encoder_type == 'roberta':
                text_embed = self.text_blocks[i](
                    text_embed,
                    attention_mask=extended_attention_mask
                )[0]

        # 3. 在应用最终的层归一化前
        if should_analyze:
            pre_norm_img = image_embed[:, 0].detach()
            pre_norm_txt = text_embed[:, 0].detach()
            self.analyze_features_at_key_points("层归一化前", pre_norm_img, pre_norm_txt)


        # 应用最终的层归一化
        image_embed = self.image_norm(image_embed)
        text_embed = self.text_norm(text_embed)
        # 4. 在层归一化后
        if should_analyze:
            post_norm_img = image_embed[:, 0].detach()
            post_norm_txt = text_embed[:, 0].detach()
            self.analyze_features_at_key_points("层归一化后", post_norm_img, post_norm_txt)

        # print(image_embed.shape,text_embed.shape)
        # 提取CLS token特征
        image_feat = image_embed[:, 0]  # [B, D_img]
        # For CLIP text encoders
        # 对于CLIP文本模型，应该使用EOT token，而不是第一个token
        # 修复文本特征提取
        if self.encoder_type == 'clip':
            # 对于CLIP文本模型，需要正确处理EOT token
            if attention_mask is not None:
                # 计算每个序列的有效长度
                seq_lengths = attention_mask.sum(dim=1) - 1  # EOT token位置

                # 处理边界情况：确保序列长度至少为0
                seq_lengths = torch.clamp(seq_lengths, min=0, max=text_embed.size(1) - 1)

                # 检查是否有无效的attention_mask（全零）
                valid_sequences = attention_mask.sum(dim=1) > 0

                # 对于无效序列，使用第一个位置（通常是开始标记）
                seq_lengths[~valid_sequences] = 0

                # 提取EOT token特征
                batch_indices = torch.arange(text_embed.size(0), device=text_embed.device)
                text_feat = text_embed[batch_indices, seq_lengths]

                # 对于无效序列，将特征置零（这些通常是缺失的文本）
                text_feat[~valid_sequences] = 0
            else:
                # 如果没有attention mask，使用最后一个token
                text_feat = text_embed[:, -1]
        else:
            # 非CLIP模型（如RoBERTa）使用第一个token（CLS token）
            text_feat = text_embed[:, 0]

        # *** 关键步骤：应用CLIP投影层 ***
        # 这里是应用投影的最佳时机：在获得最终的CLS/EOT特征后，但在后续处理前
        if self.use_clip_projections:
            # print("应用CLIP投影层...")  # 调试信息
            image_feat_projected, text_feat_projected = self.apply_clip_projections(image_feat, text_feat)

            # 调试：检查投影前后的特征
            if torch.rand(1).item() < 0.002:  # 1%概率打印调试信息
                print(f"投影前 - 图像特征: mean={image_feat.mean().item():.6f}, std={image_feat.std().item():.6f}")
                print(f"投影前 - 文本特征: mean={text_feat.mean().item():.6f}, std={text_feat.std().item():.6f}")
                print(
                    f"投影后 - 图像特征: mean={image_feat_projected.mean().item():.6f}, std={image_feat_projected.std().item():.6f}")
                print(
                    f"投影后 - 文本特征: mean={text_feat_projected.mean().item():.6f}, std={text_feat_projected.std().item():.6f}")

                # 检查特征相似度
                similarity = F.cosine_similarity(
                    image_feat_projected.mean(dim=0, keepdim=True),
                    text_feat_projected.mean(dim=0, keepdim=True)
                )
                print(f"投影后特征相似度: {similarity.item():.6f}")
        else:
            image_feat_projected = image_feat
            text_feat_projected = text_feat


        # 1. 保存原始 Transformer 输出特征，用于后续生成
        orig_image_feat = image_feat_projected  # [B, D_img]
        orig_text_feat = text_feat_projected  # [B, D_txt]

        # 2. 构造输入给生成器：直接使用原始未归一化的特征
        features_for_generation = {
            'image': orig_image_feat.unsqueeze(1) if orig_image_feat is not None else None,  # [B, 1, D_img]
            'text': orig_text_feat.unsqueeze(1) if orig_text_feat is not None else None  # [B, 1, D_txt]
        }

        # 3. 生成器处理，返回生成后的 CLS 或 EOT token（表示缺失模态）
        processed_image, processed_text, processed_attenmask, additional_info = self._process_modalities(
            features_for_generation['image'],
            features_for_generation['text'],
            attention_mask,
            missing_type
        )
        generated_features = additional_info.get('generated_features', {})

        # 4. 替换缺失模态的原始特征
        # 注意此处仍是未归一化的特征
        if is_image_missing.any() and 'image' in generated_features and generated_features['image'] is not None:
            for b in range(batch_size):
                if is_image_missing[b]:
                    orig_image_feat[b] = generated_features['image'][b].squeeze(0)

        if is_text_missing.any() and 'text' in generated_features and generated_features['text'] is not None:
            for b in range(batch_size):
                if is_text_missing[b]:
                    orig_text_feat[b] = generated_features['text'][b].squeeze(0)

        # 5. 对最终的 image/text 特征统一进行归一化（包括原始或替换后的）
        image_feat_norm = self.image_feat_norm(orig_image_feat)
        text_feat_norm = self.text_feat_norm(orig_text_feat)

        # 6. 可选：采样记录用于分析的模态特征
        if self.training and hasattr(self, 'fusion_analyzer') and self.fusion_analyzer is not None:
            should_collect = (torch.rand(1).item() < 0.1)
            if should_collect:
                try:
                    self.fusion_analyzer.collect_modality_features(
                        image_feat=image_feat_norm.detach(),
                        text_feat=text_feat_norm.detach(),
                        missing_type=missing_type
                    )
                except Exception:
                    pass


        # 5. 在质量评估和特征融合前
        if should_analyze:
            self.analyze_features_at_key_points("显式归一化", image_feat_norm.detach(), text_feat_norm.detach())

        # 这里多少有点问题
        if self.use_quality_prompt:
            quality_scores = self.quality_estimator(image_feat_norm, text_feat_norm, missing_type)


        # 5. 在质量评估和特征融合前
        if should_analyze:
            self.analyze_features_at_key_points("特征融合前", image_feat_norm.detach(), text_feat_norm.detach())

        if not hasattr(self, 'image_classifier'):
            self.image_classifier = nn.Linear(self.clip_projection_dim , self.classifier.out_features).to(image_feat.device)
            self.text_classifier = nn.Linear(self.clip_projection_dim , self.classifier.out_features).to(text_feat.device)
            with torch.no_grad():
                # 为图像分类器使用合适的初始化
                nn.init.xavier_uniform_(self.image_classifier.weight)
                if self.image_classifier.bias is not None:
                    nn.init.zeros_(self.image_classifier.bias)

                # 为文本分类器使用合适的初始化
                nn.init.xavier_uniform_(self.text_classifier.weight)
                if self.text_classifier.bias is not None:
                    nn.init.zeros_(self.text_classifier.bias)

                # 如果原始分类器有偏置，复制到新分类器
                if hasattr(self.classifier, 'bias') and self.classifier.bias is not None:
                    self.image_classifier.bias.copy_(self.classifier.bias)
                    self.text_classifier.bias.copy_(self.classifier.bias)


        # img_logits = self.image_classifier(image_feat)
        # txt_logits = self.text_classifier(text_feat)

        batch_size = image_feat.size(0)
        num_classes = self.classifier.out_features
        device = image_feat.device

        # 创建掩码
        both_present = (~is_image_missing) & (~is_text_missing)
        only_image = (~is_image_missing) & is_text_missing
        only_text = is_image_missing & (~is_text_missing)
        both_missing = is_image_missing & is_text_missing

        img_logits = torch.zeros(batch_size, num_classes, device=device)
        txt_logits = torch.zeros(batch_size, num_classes, device=device)
        combined_logits = torch.zeros(batch_size, num_classes, device=device)

        # 应用各个分类器
        if only_image.any():
            img_logits[only_image] = self.image_classifier(image_feat_norm[only_image])

        if only_text.any():
            txt_logits[only_text] = self.text_classifier(text_feat_norm[only_text])

        if both_present.any():
            # 对于都存在的样本，计算两个分类器的结果并融合
            img_pred = self.image_classifier(image_feat_norm[both_present])
            txt_pred = self.text_classifier(text_feat_norm[both_present])

            # 简单的加权融合（可以使用质量分数作为权重）
            if quality_scores is not None:
                img_weight = quality_scores['image']['final_score'][both_present]
                txt_weight = quality_scores['text']['final_score'][both_present]
                weights_sum = img_weight + txt_weight + 1e-8  # 避免除零

                img_weight_norm = img_weight / weights_sum
                txt_weight_norm = txt_weight / weights_sum

                combined_logits[both_present] = (img_pred * img_weight_norm) + (txt_pred * txt_weight_norm)
            else:
                # 默认等权融合
                combined_logits[both_present] = (img_pred + txt_pred) / 2.0



        # # TODO: 后期融合


        # 质量引导的特征融合（如果启用）
        fusion_weights = None
        quality_guided_feat = None
        if self.use_quality_prompt and self.use_cross_modal_prompt:
            quality_guided_feat, fusion_weights = self.feature_fusion(image_feat_norm, text_feat_norm, quality_scores)



        # 6. 融合后的特征
        if should_analyze and quality_guided_feat is not None:
            self.analyze_features_at_key_points("融合后", quality_guided_feat.detach(), None)


        features_to_concat = None

        # Add quality scores if enabled
        if self.use_quality_prompt:
            image_group = torch.cat([
                image_feat_norm,
                quality_scores['image']['final_score'],
                quality_scores['image']['quality'],
            ], dim=1)

            text_group = torch.cat([
                text_feat_norm,
                quality_scores['text']['final_score'],
                quality_scores['text']['quality'],
            ], dim=1)

            features_to_concat = torch.cat([
                image_group,
                text_group,
                quality_scores['cross_consistency'],
                F.one_hot(missing_type, num_classes=4).float()
            ], dim=1)
        else:
            features_to_concat = torch.cat([image_feat_norm, text_feat_norm, F.one_hot(missing_type, num_classes=4).float()], dim=1)

        # Concatenate all features
        base_hidden = self.fusion(features_to_concat)

        if should_analyze and base_hidden is not None:
            self.analyze_features_at_key_points("原始特征投影后", base_hidden.detach(), None)

        hidden = None
        if quality_guided_feat is not None:

            base_norm = self.base_fusion_norm(base_hidden)
            quality_norm = self.quality_fusion_norm(quality_guided_feat)
            alpha = 0.0  # Fixed weight or learnable parameter
            if should_analyze:
                self.analyze_features_at_key_points("再次归一化后.前者为base_norm,后者为qualitu_norm", base_norm.detach(), quality_norm.detach())

            if shoule_analyze_fusion:
                self.fusion_analyzer.analyze_fusion_features(
                    base_hidden=base_norm,
                    quality_guided_feat=quality_norm,
                    missing_type=missing_type,
                    alpha=alpha,  # 你使用的固定融合权重
                    batch_idx=None,  # 自动生成批次索引
                    save_current=True  # 保存当前批次的分析结果
                )
            hidden = alpha * base_norm + (1 - alpha) * quality_norm

        else:
            hidden = base_hidden

        if should_analyze and hidden is not None:
            self.analyze_features_at_key_points("最终融合hidden", hidden.detach(), None)

        logits = self.classifier(hidden)

        additional_info.update({
            'quality_scores': quality_scores,
            'fusion_weights': fusion_weights,
            'generated_modalities': {
                'image': is_image_missing,  # Mask indicating which images were generated
                'text': is_text_missing  # Mask indicating which texts were generated
            }
        })

        if should_analyze:  # 每5个epoch检查一次
            print(f"Image LayerNorm weights: {self.image_feat_norm.weight.mean().item():.4f}")
            print(f"Text LayerNorm weights: {self.text_feat_norm.weight.mean().item():.4f}")
            print(f"Image LayerNorm weights: {self.base_fusion_norm.weight.mean().item():.4f}")
            print(f"Text LayerNorm weights: {self.quality_fusion_norm.weight.mean().item():.4f}")

        additional_info.update({
            'modality_logits': {
                'image': img_logits,
                'text': txt_logits,
                'combined': combined_logits,
                'original': logits
            }
        })

        return (logits, additional_info)





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

        from transformers import CLIPVisionModel, CLIPTextModel, CLIPModel

        # 方法1：直接加载完整的CLIP模型然后提取组件
        try:
            clip_model = CLIPModel.from_pretrained(image_model_name)
            vision_encoder = clip_model.vision_model
            text_encoder = clip_model.text_model
            print("Successfully loaded CLIP model components from unified model")
        except Exception as e:
            print(f"Failed to load unified CLIP model: {e}")
            # 方法2：分别加载vision和text模型
            try:
                vision_encoder = CLIPVisionModel.from_pretrained(image_model_name)
                text_encoder = CLIPTextModel.from_pretrained(text_model_name)
                print("Successfully loaded separate CLIP vision and text models")
            except Exception as e2:
                print(f"Failed to load separate CLIP models: {e2}")
                raise e2

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
