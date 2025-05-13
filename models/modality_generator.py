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

        # 创建更健壮的判别器
        self.discriminators = nn.ModuleDict()
        for mod_name, dim in modality_dims.items():
            # 使用更多层和Dropout以增强稳定性
            disc = nn.Sequential(
                nn.LayerNorm(dim),  # 添加归一化
                nn.Linear(dim, 128),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.discriminators[mod_name] = disc

        # 在训练开始时打印判别器结构
        print("初始化判别器:")
        for mod_name, disc in self.discriminators.items():
            print(f"模态: {mod_name}, 维度: {modality_dims[mod_name]}")
            print(f"判别器结构: {disc}")

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
            generated, reconstructed = self._generate_for_sample(features, missing_type)
            cycle_features = self._apply_cycle_consistency(generated, missing_type)
            return generated, reconstructed, cycle_features
        else:
            # 批量处理
            batch_size = missing_type.size(0)
            device = missing_type.device

            # 初始化输出字典
            generated_features = {mod: [] for mod in self.modality_dims.keys()}
            reconstructed_features = {mod: [] for mod in self.modality_dims.keys()}
            cycle_features = {mod: [] for mod in self.modality_dims.keys()}

            # 逐个样本处理
            for b in range(batch_size):
                # 提取当前样本的特征
                sample_features = {}
                for mod in features:
                    if features[mod] is not None:
                        # 确保我们只取这个样本的特征
                        if features[mod].dim() == 3:  # [B, tokens, dim]
                            sample_features[mod] = features[mod][b:b + 1]
                        elif features[mod].dim() == 2:  # [B, dim]
                            sample_features[mod] = features[mod][b:b + 1]
                        else:
                            # 处理其他可能的维度情况
                            sample_features[mod] = features[mod][b:b + 1]
                    else:
                        sample_features[mod] = None

                # 获取当前样本的缺失类型
                mt = missing_type[b].item()

                # 为当前样本生成特征
                gen_sample, recon_sample = self._generate_for_sample(sample_features, mt)

                # 应用循环一致性
                cycle_sample = self._apply_cycle_consistency(gen_sample, mt)

                # 将结果添加到列表中
                for mod in self.modality_dims.keys():
                    # 处理生成的特征
                    if mod in gen_sample and gen_sample[mod] is not None:
                        generated_features[mod].append(gen_sample[mod])
                    else:
                        # 如果没有生成该模态，添加None占位符
                        generated_features[mod].append(None)

                    # 处理重建的特征
                    if mod in recon_sample and recon_sample[mod] is not None:
                        reconstructed_features[mod].append(recon_sample[mod])
                    else:
                        reconstructed_features[mod].append(None)

                    # 处理循环一致性特征
                    if mod in cycle_sample and cycle_sample[mod] is not None:
                        cycle_features[mod].append(cycle_sample[mod])
                    else:
                        cycle_features[mod].append(None)

            # 处理合并特征列表
            for mod in self.modality_dims.keys():
                # 检查是否有任何非None值
                if any(x is not None for x in generated_features[mod]):
                    # 提取形状信息用于创建张量
                    non_none_samples = [x for x in generated_features[mod] if x is not None]
                    if non_none_samples:
                        first_sample = non_none_samples[0]

                        # 根据第一个样本的维度创建适当的容器
                        if first_sample.dim() == 1:  # [dim]
                            gen_tensor = torch.zeros(batch_size, first_sample.size(0), device=device)
                        elif first_sample.dim() == 2:  # [tokens, dim]
                            gen_tensor = torch.zeros(batch_size, first_sample.size(0), first_sample.size(1),
                                                     device=device)
                        else:  # [1, tokens, dim] 或其他
                            # 挤压批次维度 ([1, tokens, dim] -> [tokens, dim])
                            first_sample = first_sample.squeeze(0)
                            if first_sample.dim() == 1:
                                gen_tensor = torch.zeros(batch_size, first_sample.size(0), device=device)
                            else:
                                gen_tensor = torch.zeros(batch_size, first_sample.size(0), first_sample.size(1),
                                                         device=device)

                        # 填充张量
                        for i, feat in enumerate(generated_features[mod]):
                            if feat is not None:
                                if feat.dim() > 2:  # [1, tokens, dim]
                                    feat = feat.squeeze(0)  # 移除批次维度
                                gen_tensor[i] = feat

                        generated_features[mod] = gen_tensor
                    else:
                        generated_features[mod] = None
                else:
                    generated_features[mod] = None

                # 对重建特征做同样的处理
                if any(x is not None for x in reconstructed_features[mod]):
                    non_none_samples = [x for x in reconstructed_features[mod] if x is not None]
                    if non_none_samples:
                        first_sample = non_none_samples[0]

                        if first_sample.dim() == 1:  # [dim]
                            recon_tensor = torch.zeros(batch_size, first_sample.size(0), device=device)
                        elif first_sample.dim() == 2:  # [tokens, dim]
                            recon_tensor = torch.zeros(batch_size, first_sample.size(0), first_sample.size(1),
                                                       device=device)
                        else:  # [1, tokens, dim] 或其他
                            first_sample = first_sample.squeeze(0)
                            if first_sample.dim() == 1:
                                recon_tensor = torch.zeros(batch_size, first_sample.size(0), device=device)
                            else:
                                recon_tensor = torch.zeros(batch_size, first_sample.size(0), first_sample.size(1),
                                                           device=device)

                        for i, feat in enumerate(reconstructed_features[mod]):
                            if feat is not None:
                                if feat.dim() > 2:  # [1, tokens, dim]
                                    feat = feat.squeeze(0)  # 移除批次维度
                                recon_tensor[i] = feat

                        reconstructed_features[mod] = recon_tensor
                    else:
                        reconstructed_features[mod] = None
                else:
                    reconstructed_features[mod] = None

                # 对循环一致性特征做同样的处理
                if any(x is not None for x in cycle_features[mod]):
                    non_none_samples = [x for x in cycle_features[mod] if x is not None]
                    if non_none_samples:
                        first_sample = non_none_samples[0]

                        if first_sample.dim() == 1:  # [dim]
                            cycle_tensor = torch.zeros(batch_size, first_sample.size(0), device=device)
                        elif first_sample.dim() == 2:  # [tokens, dim]
                            cycle_tensor = torch.zeros(batch_size, first_sample.size(0), first_sample.size(1),
                                                       device=device)
                        else:  # [1, tokens, dim] 或其他
                            first_sample = first_sample.squeeze(0)
                            if first_sample.dim() == 1:
                                cycle_tensor = torch.zeros(batch_size, first_sample.size(0), device=device)
                            else:
                                cycle_tensor = torch.zeros(batch_size, first_sample.size(0), first_sample.size(1),
                                                           device=device)

                        for i, feat in enumerate(cycle_features[mod]):
                            if feat is not None:
                                if feat.dim() > 2:  # [1, tokens, dim]
                                    feat = feat.squeeze(0)  # 移除批次维度
                                cycle_tensor[i] = feat

                        cycle_features[mod] = cycle_tensor
                    else:
                        cycle_features[mod] = None
                else:
                    cycle_features[mod] = None

            return generated_features, reconstructed_features, cycle_features

    def _apply_cycle_consistency(self, features, missing_type):
        """应用循环一致性，为每个模态创建循环重建版本"""
        cycle_features = {mod: None for mod in self.modality_dims.keys()}

        # 对于图像缺失的情况，使用生成的图像生成文本，然后与原始文本比较
        if missing_type == 1:  # 图像缺失
            if features['image'] is not None and features['text'] is not None:
                # 使用生成的图像生成文本
                img_feat = features['image']  # 已经是生成的图像
                encoded_img = self.generator.encode(img_feat, "image")
                cycle_text = self.generator.generators["image_to_text"](encoded_img)
                cycle_features['text'] = cycle_text

        # 对于文本缺失的情况，使用生成的文本生成图像，然后与原始图像比较
        elif missing_type == 2:  # 文本缺失
            if features['image'] is not None and features['text'] is not None:
                # 使用生成的文本生成图像
                txt_feat = features['text']  # 已经是生成的文本
                encoded_txt = self.generator.encode(txt_feat, "text")
                cycle_image = self.generator.generators["text_to_image"](encoded_txt)
                cycle_features['image'] = cycle_image

        # 对于无缺失的情况，同时执行两个方向的循环生成
        elif missing_type == 0:  # 无缺失
            if features['image'] is not None and features['text'] is not None:
                # 图像→文本→图像
                img_feat = features['image']
                encoded_img = self.generator.encode(img_feat, "image")
                gen_text = self.generator.generators["image_to_text"](encoded_img)
                encoded_gen_text = self.generator.encode(gen_text, "text")
                cycle_image = self.generator.generators["text_to_image"](encoded_gen_text)
                cycle_features['image'] = cycle_image

                # 文本→图像→文本
                txt_feat = features['text']
                encoded_txt = self.generator.encode(txt_feat, "text")
                gen_image = self.generator.generators["text_to_image"](encoded_txt)
                encoded_gen_image = self.generator.encode(gen_image, "image")
                cycle_text = self.generator.generators["image_to_text"](encoded_gen_image)
                cycle_features['text'] = cycle_text

        return cycle_features

    def discriminate(self, features):
        """使用判别器区分特征是否是真实的"""
        results = {}
        for mod, feat in features.items():
            if feat is not None:
                try:
                    # 打印输入形状以便调试
                    # print(f"判别器输入 ({mod}) 形状: {feat.shape}, 类型: {feat.dtype}")

                    # 确保是浮点型张量
                    if not feat.is_floating_point():
                        feat = feat.float()

                    # 创建输入特征的副本，但保留梯度
                    feat_clone = feat.clone()  # 不使用detach()

                    # 根据维度进行适当处理
                    if feat_clone.dim() > 2:  # [batch, tokens, dim]
                        # print(f"  处理多token特征: {feat_clone.shape}")
                        # 对多token特征计算平均值
                        feat_clone = feat_clone.mean(dim=1)  # [batch, dim]

                    # 确保是2D张量 [batch, features]
                    if feat_clone.dim() == 1:  # [dim]
                        # print(f"  添加批次维度: {feat_clone.shape} -> [{feat_clone.shape[0]}, 1]")
                        feat_clone = feat_clone.unsqueeze(0)  # [1, dim]

                    # 检查判别器并打印维度信息
                    if mod not in self.discriminators:
                        print(f"  错误: 模态 '{mod}' 没有对应的判别器")
                        continue

                    # 应用判别器
                    disc_output = self.discriminators[mod](feat_clone)
                    results[mod] = disc_output
                    # print(f"  判别器输出: {disc_output.shape}, 均值: {disc_output.mean().item():.4f}")
                    # print(f"  判别器输出是否需要梯度: {disc_output.requires_grad}")

                except Exception as e:
                    print(f"判别器错误 ({mod}): {str(e)}")
                    import traceback
                    traceback.print_exc()
                    # 创建一个默认结果并指定requires_grad=True
                    batch_size = 1
                    if feat is not None:
                        if feat.dim() > 0:
                            batch_size = feat.size(0) if feat.dim() > 1 else 1
                    default_tensor = torch.ones(batch_size, 1, device=feat.device if feat is not None else next(
                        self.parameters()).device) * 0.5
                    default_tensor.requires_grad_(True)  # 确保需要梯度
                    results[mod] = default_tensor
                    print(f"  使用默认输出: {results[mod].shape}")

        return results

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

