import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalGenerator(nn.Module):
    """跨模态生成器 - 使用可用模态生成缺失模态的特征表示"""

    def __init__(self, modality_dims, fusion_hidden_dim=256):
        """
        初始化跨模态生成器

        Args:
            modality_dims: 字典，键为模态名称，值为该模态的特征维度
                           例如: {'image': 768, 'text': 512}
            fusion_hidden_dim: 融合层隐藏维度
        """
        super().__init__()
        self.modality_dims = modality_dims
        self.modalities = list(modality_dims.keys())

        # 为每个模态创建编码器，将原始特征映射到共享空间
        self.encoders = nn.ModuleDict({
            mod_name: nn.Sequential(
                nn.Linear(dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU()
            ) for mod_name, dim in modality_dims.items()
        })

        # 为每个模态对创建生成器
        self.generators = nn.ModuleDict()
        for source_mod in self.modalities:
            for target_mod in self.modalities:
                if source_mod != target_mod:
                    self.generators[f"{source_mod}_to_{target_mod}"] = nn.Sequential(
                        nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                        nn.LayerNorm(fusion_hidden_dim),
                        nn.GELU(),
                        nn.Linear(fusion_hidden_dim, modality_dims[target_mod]),
                        nn.Tanh()  # 使输出范围在[-1,1]之间，可根据实际特征分布调整
                    )

        # 对于"both"情况（双模态缺失）需要的先验生成器
        self.prior_generators = nn.ModuleDict({
            mod_name: nn.Sequential(
                nn.Linear(128, fusion_hidden_dim),  # 从更丰富的随机噪声生成
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.LayerNorm(fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, dim),
                nn.Tanh()
            ) for mod_name, dim in modality_dims.items()
        })

    def encode(self, features, modality):
        """将特定模态的特征编码到共享空间"""
        if features is None:
            return None

        # 处理多个token的情况
        if features.dim() > 2:  # [batch_size, token_count, dim]
            # 对每个token分别编码
            encoded_features = self.encoders[modality](features)
            return encoded_features
        else:  # [token_count, dim] 或 [batch_size, dim]
            return self.encoders[modality](features)

    def generate(self, source_features, source_modality, target_modality):
        """基于源模态特征生成目标模态特征"""
        if source_features is None:
            # 如果源特征不可用，使用先验生成器
            batch_size = 1  # 默认批次大小
            for mod in self.modalities:
                if mod in source_features and source_features[mod] is not None:
                    batch_size = source_features[mod].size(0)
                    break

            # 使用随机噪声作为输入生成目标模态特征
            noise = torch.randn(batch_size, 1, device=next(self.parameters()).device)
            return self.prior_generators[target_modality](noise)

        # 编码源特征
        encoded = self.encode(source_features, source_modality)

        # 使用生成器生成目标特征
        return self.generators[f"{source_modality}_to_{target_modality}"](encoded)

    def forward(self, features, missing_types):
        """
        前向传播，根据缺失情况生成特征

        Args:
            features: 字典，键为模态名称，值为该模态的特征（可能为None表示缺失）
            missing_types: 张量，表示每个样本的缺失类型 (none=0, image=1, text=2, both=3)

        Returns:
            完整的特征字典，包含原始和生成的特征
        """
        batch_size = missing_types.size(0)
        device = missing_types.device

        # 创建输出特征字典，初始化为空 list
        output_features = {mod: [] for mod in self.modalities}

        for b in range(batch_size):
            missing_type = missing_types[b].item()

            # 根据缺失类型生成特征
            if missing_type == 1:  # 图像缺失
                if "text" in features and features["text"] is not None:
                    text_feat = features["text"][b:b + 1]
                    gen_image_feat = self.generate(text_feat, "text", "image")
                    output_features["image"].append(gen_image_feat)
                    output_features["text"].append(text_feat)

            elif missing_type == 2:  # 文本缺失
                if "image" in features and features["image"] is not None:
                    image_feat = features["image"][b:b + 1]
                    gen_text_feat = self.generate(image_feat, "image", "text")
                    output_features["text"].append(gen_text_feat)
                    output_features["image"].append(image_feat)

            elif missing_type == 3:  # 双模态都缺失
                noise = torch.randn(1, 1, device=device)
                gen_image_feat = self.prior_generators["image"](noise)
                gen_text_feat = self.prior_generators["text"](noise)
                output_features["image"].append(gen_image_feat)
                output_features["text"].append(gen_text_feat)

            else:  # missing_type == 0, 没缺失
                for mod in self.modalities:
                    if features.get(mod) is not None:
                        output_features[mod].append(features[mod][b:b + 1])

        # 最后 stack 回 tensor
        for mod in self.modalities:
            if len(output_features[mod]) > 0:
                output_features[mod] = torch.cat(output_features[mod], dim=0)
            else:
                output_features[mod] = None

        return output_features


class ModReconstructor(nn.Module):
    """模态重建器 - 用于对比学习以提高生成质量"""

    def __init__(self, modality_dims):
        """
        初始化模态重建器

        Args:
            modality_dims: 字典，键为模态名称，值为该模态的特征维度
        """
        super().__init__()
        self.decoders = nn.ModuleDict({
            mod_name: nn.Sequential(
                nn.Linear(dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim)
            ) for mod_name, dim in modality_dims.items()
        })

        # 添加注意力融合层，用于多token处理
        self.attention_fusion = nn.ModuleDict({
            mod_name: nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
            for mod_name, dim in modality_dims.items()
        })

    def forward(self, features):
        """重建各模态特征"""
        if not isinstance(features, dict):
            raise ValueError(f"ModReconstructor expects input features as a dict, got {type(features)}")

        outputs = {}
        for mod, feat in features.items():
            if feat is not None:
                feat = feat.clone()  # 避免inplace问题

                # 使用自注意力机制处理多个token
                if feat.dim() > 2:  # 批次 + 多token情况
                    attn_output, _ = self.attention_fusion[mod](
                        query=feat,
                        key=feat,
                        value=feat
                    )
                    # 对每个token进行解码
                    outputs[mod] = self.decoders[mod](attn_output)
                else:  # 单token情况
                    outputs[mod] = self.decoders[mod](feat)

        return outputs


class CycleGenerationModel(nn.Module):
    """循环生成模型 - 实现特征生成和重建的循环一致性"""

    def __init__(self, modality_dims, fusion_hidden_dim=256):
        super().__init__()
        self.modality_dims = modality_dims
        self.generator = CrossModalGenerator(modality_dims, fusion_hidden_dim)
        self.reconstructor = ModReconstructor(modality_dims)

    def forward(self, features, missing_type):
        """
        前向传播，针对单个样本或批量样本进行模态生成和重建

        Args:
            features: 字典，键为模态名称，值为该模态的特征（可能为None表示缺失）
            missing_type: 整数或张量，表示缺失类型 (none=0, image=1, text=2, both=3)

        Returns:
            生成的特征和重建的特征
        """
        # 检查是单个样本还是批量样本
        batch_mode = isinstance(missing_type, torch.Tensor) and missing_type.dim() > 0

        if not batch_mode:
            # 单样本处理
            missing_type = int(missing_type)
            generated , reconstructed= self._generate_for_sample(features, missing_type)
            # reconstructed = self.reconstructor(generated)
            return generated, reconstructed
        else:
            # 批量处理，但逐个样本处理以避免批处理问题
            batch_size = missing_type.size(0)
            device = missing_type.device

            # 初始化输出
            generated_features = {mod: None for mod in self.modality_dims.keys()}
            reconstructed_features = {mod: None for mod in self.modality_dims.keys()}

            # 逐个样本处理
            for b in range(batch_size):
                # 提取当前样本的特征
                sample_features = {}
                for mod in features:
                    if features[mod] is not None:
                        # 确保我们只取这个样本的特征
                        sample_features[mod] = features[mod][b:b + 1]
                    else:
                        sample_features[mod] = None

                # 生成当前样本的特征
                mt = missing_type[b].item()
                gen_sample, recon_sample = self._generate_for_sample(sample_features, mt)

                # 合并到批结果中
                for mod in gen_sample:
                    if gen_sample[mod] is not None:
                        if generated_features[mod] is None:
                            # 首次分配空间
                            generated_features[mod] = torch.zeros(
                                batch_size, self.modality_dims[mod], device=device
                            )
                        generated_features[mod][b] = gen_sample[mod]

                for mod in recon_sample:
                    if recon_sample[mod] is not None:
                        if reconstructed_features[mod] is None:
                            # 首次分配空间
                            reconstructed_features[mod] = torch.zeros(
                                batch_size, self.modality_dims[mod], device=device
                            )
                        reconstructed_features[mod][b] = recon_sample[mod]

            return generated_features, reconstructed_features

    def _generate_for_sample(self, features, missing_type):
        """针对单个样本生成缺失模态特征"""
        token_count = 5  # 使用的token数量

        generated = {}
        for mod in self.modality_dims.keys():
            if features.get(mod) is not None:
                # 如果特征是2D的，则提取前token_count个token
                if features[mod].dim() == 2:
                    if features[mod].size(0) >= token_count:
                        generated[mod] = features[mod][:token_count].clone()
                    else:
                        # 如果token不够，则复制现有的token
                        repeat_times = (token_count + features[mod].size(0) - 1) // features[mod].size(0)
                        generated[mod] = features[mod].repeat(repeat_times, 1)[:token_count].clone()
                else:
                    # 对于单个样本的情况
                    generated[mod] = features[mod].clone()
            else:
                generated[mod] = None

        # 根据缺失类型生成特征
        if missing_type == 1:  # 图像缺失
            if features.get("text") is not None:
                text_feat = generated["text"]  # 使用多个token
                encoded_text = self.generator.encode(text_feat, "text")
                generated["image"] = self.generator.generators["text_to_image"](encoded_text)
                # 添加一点噪声提高泛化能力
                generated["image"] = generated["image"] + torch.randn_like(generated["image"]) * 0.01

        elif missing_type == 2:  # 文本缺失
            if features.get("image") is not None:
                image_feat = generated["image"]  # 使用多个token
                encoded_image = self.generator.encode(image_feat, "image")
                generated["text"] = self.generator.generators["image_to_text"](encoded_image)
                # 添加一点噪声提高泛化能力
                generated["text"] = generated["text"] + torch.randn_like(generated["text"]) * 0.01

        elif missing_type == 3:  # 双模态缺失
            # 使用随机噪声生成特征
            noise = torch.randn(token_count, 1, device=next(self.parameters()).device)
            generated["image"] = self.generator.prior_generators["image"](noise)
            generated["text"] = self.generator.prior_generators["text"](noise)

        # 添加正则项
        if missing_type == 1:  # Image missing
            generated["image"] = generated["image"] + torch.randn_like(generated["image"]) * 0.01
        elif missing_type == 2:  # Text missing
            generated["text"] = generated["text"] + torch.randn_like(generated["text"]) * 0.01

        return generated, self.reconstructor(generated)