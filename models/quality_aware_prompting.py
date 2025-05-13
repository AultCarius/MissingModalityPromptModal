# Quality Aware Prompting
from timm import create_model
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F


class ImprovedModalityQualityEstimator(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim=256):
        super().__init__()

        # Feature quality estimators
        self.image_quality = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5)  # [clarity, completeness, informativeness, confidence, detail]
        )

        self.text_quality = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5)  # [coherence, relevance, informativeness, confidence, detail]
        )

        # Cross-modal consistency estimator
        self.cross_consistency = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # [semantic_alignment, structural_alignment, contextual_alignment]
        )

        # Reference statistics (will be updated during training)
        self.register_buffer('image_mean', torch.zeros(1, image_dim))
        self.register_buffer('image_std', torch.ones(1, image_dim))
        self.register_buffer('text_mean', torch.zeros(1, text_dim))
        self.register_buffer('text_std', torch.ones(1, text_dim))

        # Learnable quality baseline parameters for generated modalities
        # Initialize with moderate values (0.5) instead of extreme values
        self.generated_image_quality_base = nn.Parameter(torch.tensor([0.5]))
        self.generated_text_quality_base = nn.Parameter(torch.tensor([0.5]))

        # Modality weights
        self.image_weights = nn.Parameter(torch.ones(5) / 5)
        self.text_weights = nn.Parameter(torch.ones(5) / 5)
        self.consistency_weights = nn.Parameter(torch.ones(3) / 3)

        # Fusion weight for balancing individual quality and consistency
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))

        # Learnable parameters to adjust generated quality based on feature similarity
        self.gen_quality_factor = nn.Parameter(torch.tensor([0.5]))

        # Keep track of quality distributions for different types
        self.register_buffer('real_img_quality_mean', torch.tensor([0.8]))
        self.register_buffer('real_img_quality_std', torch.tensor([0.1]))
        self.register_buffer('real_txt_quality_mean', torch.tensor([0.8]))
        self.register_buffer('real_txt_quality_std', torch.tensor([0.1]))
        self.register_buffer('gen_img_quality_mean', torch.tensor([0.5]))
        self.register_buffer('gen_img_quality_std', torch.tensor([0.2]))
        self.register_buffer('gen_txt_quality_mean', torch.tensor([0.5]))
        self.register_buffer('gen_txt_quality_std', torch.tensor([0.2]))

    def update_reference_statistics(self, image_feats, text_feats, is_image_missing=None, is_text_missing=None):
        """Update reference statistics for normalization and quality distribution tracking"""
        with torch.no_grad():
            # Update feature statistics for normalization
            if image_feats is not None and len(image_feats) > 0:
                # Only update using real image features
                if is_image_missing is not None:
                    real_img_mask = ~is_image_missing
                    if real_img_mask.any():
                        real_image_feats = image_feats[real_img_mask]
                        self.image_mean = 0.9 * self.image_mean + 0.1 * real_image_feats.mean(0, keepdim=True).detach()
                        self.image_std = 0.9 * self.image_std + 0.1 * real_image_feats.std(0,
                                                                                           keepdim=True).detach().clamp(
                            min=1e-6)
                else:
                    self.image_mean = 0.9 * self.image_mean + 0.1 * image_feats.mean(0, keepdim=True).detach()
                    self.image_std = 0.9 * self.image_std + 0.1 * image_feats.std(0, keepdim=True).detach().clamp(
                        min=1e-6)

            if text_feats is not None and len(text_feats) > 0:
                # Only update using real text features
                if is_text_missing is not None:
                    real_txt_mask = ~is_text_missing
                    if real_txt_mask.any():
                        real_text_feats = text_feats[real_txt_mask]
                        self.text_mean = 0.9 * self.text_mean + 0.1 * real_text_feats.mean(0, keepdim=True).detach()
                        self.text_std = 0.9 * self.text_std + 0.1 * real_text_feats.std(0, keepdim=True).detach().clamp(
                            min=1e-6)
                else:
                    self.text_mean = 0.9 * self.text_mean + 0.1 * text_feats.mean(0, keepdim=True).detach()
                    self.text_std = 0.9 * self.text_std + 0.1 * text_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

    def update_quality_distributions(self, quality_scores, is_image_missing=None, is_text_missing=None):
        """Track quality score distributions for real and generated features"""
        if is_image_missing is None or is_text_missing is None:
            return

        with torch.no_grad():
            # Update real image quality distribution
            real_img_mask = ~is_image_missing
            if real_img_mask.any():
                real_img_quality = quality_scores['image']['final_score'][real_img_mask]
                self.real_img_quality_mean = 0.9 * self.real_img_quality_mean + 0.1 * real_img_quality.mean().detach()
                self.real_img_quality_std = 0.9 * self.real_img_quality_std + 0.1 * real_img_quality.std().detach().clamp(
                    min=0.05)

            # Update generated image quality distribution
            gen_img_mask = is_image_missing
            if gen_img_mask.any():
                gen_img_quality = quality_scores['image']['final_score'][gen_img_mask]
                self.gen_img_quality_mean = 0.9 * self.gen_img_quality_mean + 0.1 * gen_img_quality.mean().detach()
                self.gen_img_quality_std = 0.9 * self.gen_img_quality_std + 0.1 * gen_img_quality.std().detach().clamp(
                    min=0.05)

            # Update real text quality distribution
            real_txt_mask = ~is_text_missing
            if real_txt_mask.any():
                real_txt_quality = quality_scores['text']['final_score'][real_txt_mask]
                self.real_txt_quality_mean = 0.9 * self.real_txt_quality_mean + 0.1 * real_txt_quality.mean().detach()
                self.real_txt_quality_std = 0.9 * self.real_txt_quality_std + 0.1 * real_txt_quality.std().detach().clamp(
                    min=0.05)

            # Update generated text quality distribution
            gen_txt_mask = is_text_missing
            if gen_txt_mask.any():
                gen_txt_quality = quality_scores['text']['final_score'][gen_txt_mask]
                self.gen_txt_quality_mean = 0.9 * self.gen_txt_quality_mean + 0.1 * gen_txt_quality.mean().detach()
                self.gen_txt_quality_std = 0.9 * self.gen_txt_quality_std + 0.1 * gen_txt_quality.std().detach().clamp(
                    min=0.05)

    def normalize_features(self, image_feats, text_feats):
        """Normalize features using stored statistics"""
        norm_img_feats = None
        if image_feats is not None:
            norm_img_feats = (image_feats - self.image_mean) / self.image_std

        norm_txt_feats = None
        if text_feats is not None:
            norm_txt_feats = (text_feats - self.text_mean) / self.text_std

        return norm_img_feats, norm_txt_feats

    def forward(self, image_feat, text_feat, missing_type=None):
        """
        Assess modality quality with more balanced treatment of generated features

        Args:
            image_feat: image features [B, D_img]
            text_feat: text features [B, D_txt]
            missing_type: tensor indicating missing modality type (0=none, 1=image, 2=text, 3=both)

        Returns:
            Dictionary containing quality scores
        """
        batch_size = max(image_feat.size(0) if image_feat is not None else 0,
                         text_feat.size(0) if text_feat is not None else 0)
        device = image_feat.device if image_feat is not None else text_feat.device

        # Identify missing modalities
        if missing_type is not None:
            is_image_missing = (missing_type == 1) | (missing_type == 3)
            is_text_missing = (missing_type == 2) | (missing_type == 3)
        else:
            is_image_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)
            is_text_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Normalize features for more stable quality estimation
        norm_img, norm_txt = self.normalize_features(image_feat, text_feat)

        # Initialize results
        results = {
            'image': {
                'quality': torch.zeros(batch_size, 5, device=device),
                'final_score': torch.zeros(batch_size, 1, device=device)
            },
            'text': {
                'quality': torch.zeros(batch_size, 5, device=device),
                'final_score': torch.zeros(batch_size, 1, device=device)
            },
            'cross_consistency': torch.zeros(batch_size, 1, device=device)
        }

        # Image quality assessment
        if norm_img is not None:
            # Get quality dimensions for all samples
            img_quality = self.image_quality(norm_img)

            # Apply different scaling for real vs. generated features
            img_scales = torch.ones(batch_size, 1, device=device)

            # More balanced scaling for generated features - use learnable parameter
            # Instead of fixed 0.7 multiplier, use the learned value (initialized to 0.5)
            gen_img_scale = torch.sigmoid(self.generated_image_quality_base)

            # Scale factor increases as training progresses to gradually rely more on generated features
            if self.training:
                # Start with a stronger penalty and gradually reduce it
                img_scales[is_image_missing] = gen_img_scale
            else:
                # During evaluation, use a more balanced scaling
                # This is key - we don't want to completely dismiss generated features
                img_scales[is_image_missing] = gen_img_scale * 1.2  # Slight boost during evaluation

            # Apply scaling to quality scores
            img_quality = img_quality * img_scales

            # Calculate final score using learned weights
            img_weights = F.softmax(self.image_weights, dim=0)
            img_final = torch.sigmoid(torch.sum(img_quality * img_weights, dim=1, keepdim=True))

            # Apply final adjustments - make the quality range more reasonable
            # For real features: keep the original range (typically high)
            # For generated features: use a more moderate range based on learned base quality
            img_final[is_image_missing] = torch.clamp(
                img_final[is_image_missing],
                min=0.3,  # Minimum quality for generated features
                max=0.8  # Maximum quality for generated features
            )

            results['image']['quality'] = img_quality
            results['image']['final_score'] = img_final
        else:
            # Default low scores for missing modality
            results['image']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['image']['final_score'] = torch.full((batch_size, 1), 0.3,
                                                         device=device)  # More reasonable baseline

        # Text quality assessment (similar approach to image)
        if norm_txt is not None:
            txt_quality = self.text_quality(norm_txt)

            txt_scales = torch.ones(batch_size, 1, device=device)
            gen_txt_scale = torch.sigmoid(self.generated_text_quality_base)

            if self.training:
                txt_scales[is_text_missing] = gen_txt_scale
            else:
                txt_scales[is_text_missing] = gen_txt_scale * 1.2  # Slight boost during evaluation

            txt_quality = txt_quality * txt_scales

            txt_weights = F.softmax(self.text_weights, dim=0)
            txt_final = torch.sigmoid(torch.sum(txt_quality * txt_weights, dim=1, keepdim=True))

            # Apply reasonable bounds for generated features
            txt_final[is_text_missing] = torch.clamp(
                txt_final[is_text_missing],
                min=0.3,
                max=0.8
            )

            results['text']['quality'] = txt_quality
            results['text']['final_score'] = txt_final
        else:
            results['text']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['text']['final_score'] = torch.full((batch_size, 1), 0.3, device=device)

        # Cross-modal consistency assessment
        if norm_img is not None and norm_txt is not None:
            concat_feat = torch.cat([norm_img, norm_txt], dim=1)
            consistency_dims = self.cross_consistency(concat_feat)

            # Apply reasonable penalties for generated features
            cons_scales = torch.ones(batch_size, 1, device=device)

            # More balanced scaling for consistency:
            # - If one modality is generated: moderate scaling
            # - If both are generated: lower scaling, but not extreme
            both_generated = is_image_missing & is_text_missing
            one_generated = (is_image_missing | is_text_missing) & ~both_generated

            gen_factor = torch.sigmoid(self.gen_quality_factor)
            cons_scales[one_generated] = 0.7 * gen_factor + 0.3  # Between 0.3 and 1.0 based on learned factor
            cons_scales[both_generated] = 0.5 * gen_factor + 0.2  # Between 0.2 and 0.7 based on learned factor

            consistency_dims = consistency_dims * cons_scales

            # Calculate final consistency score
            consistency_weights = F.softmax(self.consistency_weights, dim=0)
            consistency = torch.sigmoid(torch.sum(consistency_dims * consistency_weights, dim=1, keepdim=True))

            # Ensure reasonable consistency values
            consistency[both_generated] = torch.clamp(consistency[both_generated], min=0.2, max=0.6)
            consistency[one_generated] = torch.clamp(consistency[one_generated], min=0.3, max=0.7)

            results['cross_consistency'] = consistency
        else:
            # Default consistency when one modality is missing completely
            results['cross_consistency'] = torch.full((batch_size, 1), 0.4, device=device)  # More reasonable baseline

        # Adjust modality scores based on consistency
        w = torch.sigmoid(self.fusion_weight)

        if norm_img is not None and norm_txt is not None:
            # Adjust scores based on consistency - balanced approach
            if self.training:
                # During training, gradually increase the importance of consistency
                results['image']['final_score'] = (1 - w) * results['image']['final_score'] + w * results[
                    'cross_consistency']
                results['text']['final_score'] = (1 - w) * results['text']['final_score'] + w * results[
                    'cross_consistency']
            else:
                # During evaluation, use fixed weighting
                results['image']['final_score'] = 0.7 * results['image']['final_score'] + 0.3 * results[
                    'cross_consistency']
                results['text']['final_score'] = 0.7 * results['text']['final_score'] + 0.3 * results[
                    'cross_consistency']

        # Apply final normalization to ensure quality scores follow expected distributions
        if self.training:
            # Track quality distributions
            self.update_quality_distributions(results, is_image_missing, is_text_missing)

            # Normalize scores to follow expected distributions
            # Real features should have a distribution around real_mean with real_std
            # Generated features should have a distribution around gen_mean with gen_std

            # For image features
            if results['image']['final_score'] is not None:
                # Separate real and generated features
                real_img_mask = ~is_image_missing
                gen_img_mask = is_image_missing

                if real_img_mask.any():
                    # Normalize real image quality scores
                    img_scores = results['image']['final_score'][real_img_mask]
                    z_scores = (img_scores - img_scores.mean()) / (img_scores.std() + 1e-5)
                    normalized_scores = z_scores * self.real_img_quality_std + self.real_img_quality_mean
                    results['image']['final_score'][real_img_mask] = normalized_scores.clamp(0.6, 0.95)

                if gen_img_mask.any():
                    # Normalize generated image quality scores
                    img_scores = results['image']['final_score'][gen_img_mask]
                    z_scores = (img_scores - img_scores.mean()) / (img_scores.std() + 1e-5)
                    normalized_scores = z_scores * self.gen_img_quality_std + self.gen_img_quality_mean
                    results['image']['final_score'][gen_img_mask] = normalized_scores.clamp(0.2, 0.7)

            # For text features (similar to image)
            if results['text']['final_score'] is not None:
                real_txt_mask = ~is_text_missing
                gen_txt_mask = is_text_missing

                if real_txt_mask.any():
                    txt_scores = results['text']['final_score'][real_txt_mask]
                    z_scores = (txt_scores - txt_scores.mean()) / (txt_scores.std() + 1e-5)
                    normalized_scores = z_scores * self.real_txt_quality_std + self.real_txt_quality_mean
                    results['text']['final_score'][real_txt_mask] = normalized_scores.clamp(0.6, 0.95)

                if gen_txt_mask.any():
                    txt_scores = results['text']['final_score'][gen_txt_mask]
                    z_scores = (txt_scores - txt_scores.mean()) / (txt_scores.std() + 1e-5)
                    normalized_scores = z_scores * self.gen_txt_quality_std + self.gen_txt_quality_mean
                    results['text']['final_score'][gen_txt_mask] = normalized_scores.clamp(0.2, 0.7)

        # Update reference statistics for future calls
        if self.training:
            self.update_reference_statistics(image_feat, text_feat, is_image_missing, is_text_missing)

        return results


class EnhancedModalityQualityEstimator(nn.Module):
    def __init__(self, image_dim, text_dim, hidden_dim=256):
        super().__init__()

        # 特征质量评估器
        self.image_quality = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5)  # [clarity, completeness, informativeness, confidence, detail]
        )

        self.text_quality = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5)  # [coherence, relevance, informativeness, confidence, detail]
        )

        # 跨模态一致性评估器
        self.cross_consistency = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # [semantic_alignment, structural_alignment, contextual_alignment]
        )

        # 参考样本统计数据 - 用于特征归一化
        self.register_buffer('image_mean', torch.zeros(1, image_dim))
        self.register_buffer('image_std', torch.ones(1, image_dim))
        self.register_buffer('text_mean', torch.zeros(1, text_dim))
        self.register_buffer('text_std', torch.ones(1, text_dim))

        # 可学习的重要性权重
        self.image_weights = nn.Parameter(torch.ones(5) / 5)
        self.text_weights = nn.Parameter(torch.ones(5) / 5)
        self.consistency_weights = nn.Parameter(torch.ones(3) / 3)

        # 质量预测校准因子
        self.quality_calibration = nn.Parameter(torch.tensor([0.5, 0.5]))

    def update_reference_statistics(self, image_feats, text_feats):
        """更新参考统计数据用于标准化(在训练期间调用)"""
        if image_feats is not None and len(image_feats) > 0:
            with torch.no_grad():
                self.image_mean = image_feats.mean(0, keepdim=True).detach()
                self.image_std = image_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

        if text_feats is not None and len(text_feats) > 0:
            with torch.no_grad():
                self.text_mean = text_feats.mean(0, keepdim=True).detach()
                self.text_std = text_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

    def normalize_features(self, image_feats, text_feats):
        """使用存储的统计数据标准化特征"""
        norm_img_feats = None
        if image_feats is not None:
            norm_img_feats = (image_feats - self.image_mean) / self.image_std

        norm_txt_feats = None
        if text_feats is not None:
            norm_txt_feats = (text_feats - self.text_mean) / self.text_std

        return norm_img_feats, norm_txt_feats

    def forward(self, image_feat, text_feat, missing_type=None):
        """
        前向传播函数

        Args:
            image_feat: 图像特征 [B, D_img]
            text_feat: 文本特征 [B, D_txt]
            missing_type: 缺失类型张量 (none=0, image=1, text=2, both=3)

        Returns:
            包含质量评估结果的字典
        """
        batch_size = max(image_feat.size(0) if image_feat is not None else 0,
                         text_feat.size(0) if text_feat is not None else 0)
        device = image_feat.device if image_feat is not None else text_feat.device

        # 标准化特征以获得更稳定的质量评估
        norm_img, norm_txt = self.normalize_features(image_feat, text_feat)

        # 初始化结果
        results = {
            'image': {
                'quality': torch.zeros(batch_size, 5, device=device),
                'final_score': torch.zeros(batch_size, 1, device=device)
            },
            'text': {
                'quality': torch.zeros(batch_size, 5, device=device),
                'final_score': torch.zeros(batch_size, 1, device=device)
            },
            'cross_consistency': torch.zeros(batch_size, 1, device=device)
        }

        # 图像质量评估
        if norm_img is not None:
            # 获取质量维度
            img_quality = self.image_quality(norm_img)

            # 使用学习到的权重计算最终分数
            img_weights = F.softmax(self.image_weights, dim=0)
            img_final = torch.sigmoid(torch.sum(img_quality * img_weights, dim=1, keepdim=True))

            results['image']['quality'] = img_quality
            results['image']['final_score'] = img_final
        else:
            # 缺失模态的默认低分
            results['image']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['image']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # 文本质量评估(类似方法)
        if norm_txt is not None:
            txt_quality = self.text_quality(norm_txt)

            txt_weights = F.softmax(self.text_weights, dim=0)
            txt_final = torch.sigmoid(torch.sum(txt_quality * txt_weights, dim=1, keepdim=True))

            results['text']['quality'] = txt_quality
            results['text']['final_score'] = txt_final
        else:
            results['text']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['text']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # 跨模态一致性评估
        if norm_img is not None and norm_txt is not None:
            concat_feat = torch.cat([norm_img, norm_txt], dim=1)
            consistency_dims = self.cross_consistency(concat_feat)

            # 计算最终一致性分数
            consistency_weights = F.softmax(self.consistency_weights, dim=0)
            consistency = torch.sigmoid(torch.sum(consistency_dims * consistency_weights, dim=1, keepdim=True))

            results['cross_consistency'] = consistency
        else:
            results['cross_consistency'] = torch.full((batch_size, 1), 0.5, device=device)

        # 最终调整基于一致性的模态分数
        if norm_img is not None and norm_txt is not None:
            # 基于一致性调整分数
            cal_factor = torch.sigmoid(self.quality_calibration)
            results['image']['final_score'] = cal_factor[0] * results['image']['final_score'] + (1 - cal_factor[0]) * \
                                              results['cross_consistency']
            results['text']['final_score'] = cal_factor[1] * results['text']['final_score'] + (1 - cal_factor[1]) * \
                                             results['cross_consistency']

        return results


class QualityAwareFeatureFusion(nn.Module):
    def __init__(self, image_dim, text_dim, fusion_dim, num_heads=4):
        super().__init__()

        # 特征投影
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)

        # 质量感知的注意力
        self.quality_attn_weights = nn.Sequential(
            nn.Linear(2, num_heads),  # 从质量分数投影到注意力头权重
            nn.Sigmoid()  # 确保权重为正
        )

        # 多头注意力以进行质量感知的跨模态融合
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 融合后的输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # 用于质量感知的门控机制
        self.quality_gate = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, image_feat, text_feat, quality_scores=None):
        """
        基于质量分数的前向传播函数

        Args:
            image_feat: 图像特征 [B, D_img]
            text_feat: 文本特征 [B, D_txt]
            quality_scores: 质量评估模块的输出

        Returns:
            融合特征和权重
        """
        batch_size = image_feat.size(0)
        device = image_feat.device

        # 投影特征到融合空间
        img_proj = self.image_proj(image_feat)  # [B, fusion_dim]
        txt_proj = self.text_proj(text_feat)  # [B, fusion_dim]

        # 准备质量向量用于注意力调制
        if quality_scores is None or 'image' not in quality_scores or 'text' not in quality_scores:
            # 如果没有质量分数，使用均匀权重
            quality_vector = torch.ones(batch_size, 2, device=device) * 0.5
        else:
            # 使用质量分数
            quality_vector = torch.cat([
                quality_scores['image']['final_score'],
                quality_scores['text']['final_score']
            ], dim=1)  # [B, 2]

        # 生成注意力权重
        attn_weights = self.quality_attn_weights(quality_vector)  # [B, num_heads]

        # 堆叠特征作为序列 [img, txt]
        features = torch.stack([img_proj, txt_proj], dim=1)  # [B, 2, fusion_dim]

        # 应用质量感知的跨模态注意力
        # 我们现在不需要修改原生的attention机制，而是在后处理中应用质量权重
        attn_output, _ = self.cross_attn(
            query=features,
            key=features,
            value=features
        )  # [B, 2, fusion_dim]

        # 提取各模态的表示
        img_attn = attn_output[:, 0]  # [B, fusion_dim]
        txt_attn = attn_output[:, 1]  # [B, fusion_dim]

        # 计算质量门控值
        quality_gate = self.quality_gate(quality_vector)  # [B, 1]

        # 基于质量的加权平均
        weighted_img = img_proj * quality_scores['image']['final_score']
        weighted_txt = txt_proj * quality_scores['text']['final_score']

        # 计算质量加权的特征
        quality_weighted = (weighted_img + weighted_txt) / (
                quality_scores['image']['final_score'] + quality_scores['text']['final_score'] + 1e-8
        )

        # 注意力输出和质量加权之间的插值
        attn_weighted = (img_attn + txt_attn) / 2
        fused_features = quality_gate * quality_weighted + (1 - quality_gate) * attn_weighted

        # 应用最终投影
        output_features = self.output_proj(torch.cat([fused_features, attn_weighted], dim=1))

        # 为可视化目的返回简单权重
        simple_weights = torch.cat([
            quality_scores['image']['final_score'],
            quality_scores['text']['final_score']
        ], dim=1)

        return output_features, simple_weights


    # def forward(self, image_feat, text_feat, quality_scores=None):
    #     batch_size = image_feat.size(0)
    #     device = image_feat.device
    #     fusion_dim = self.image_proj.out_features  # 获取融合维度
    #
    #     # 投影特征到融合空间
    #     img_proj = self.image_proj(image_feat)  # [B, fusion_dim]
    #     txt_proj = self.text_proj(text_feat)  # [B, fusion_dim]
    #
    #     # 准备堆叠的特征（与原始实现一致）
    #     features = torch.stack([img_proj, txt_proj], dim=1)  # [B, 2, fusion_dim]
    #
    #     # 如果有质量分数，使用它们作为权重
    #     if quality_scores is not None and 'image' in quality_scores and 'text' in quality_scores:
    #         if 'final_score' in quality_scores['image'] and 'final_score' in quality_scores['text']:
    #             img_quality = quality_scores['image']['final_score']  # [B, 1]
    #             txt_quality = quality_scores['text']['final_score']  # [B, 1]
    #
    #             # 归一化权重确保它们的和为1
    #             total_quality = img_quality + txt_quality + 1e-8  # 添加小值防止除零
    #             norm_img_quality = img_quality / total_quality
    #             norm_txt_quality = txt_quality / total_quality
    #
    #             # 使用质量分数加权融合
    #             weighted_img = img_proj * norm_img_quality
    #             weighted_txt = txt_proj * norm_txt_quality
    #
    #             # 创建质量加权的特征，形状与原始多头注意力输出一致
    #             attn_output = torch.stack([weighted_img, weighted_txt], dim=1)  # [B, 2, fusion_dim]
    #         else:
    #             # 无质量分数时的备选方案
    #             attn_output = features.clone()  # 简单复制
    #     else:
    #         # 无质量分数时的备选方案
    #         attn_output = features.clone()  # 简单复制
    #         norm_img_quality = torch.ones(batch_size, 1, device=device) * 0.5
    #         norm_txt_quality = torch.ones(batch_size, 1, device=device) * 0.5
    #
    #     # 按照原始实现重塑特征
    #     features_flat = features.reshape(batch_size, -1)  # [B, 2*fusion_dim]
    #     attn_output_flat = attn_output.reshape(batch_size, -1)  # [B, 2*fusion_dim]
    #
    #     # 连接原始特征和加权特征（与原始实现保持一致的维度）
    #     concat_features = torch.cat([features_flat, attn_output_flat], dim=1)  # [B, 4*fusion_dim]
    #
    #     # 最终投影
    #     fused_features = self.output_proj(concat_features)
    #
    #     # 生成简化的权重返回（与原始实现一致）
    #     simple_weights = torch.cat([norm_img_quality, norm_txt_quality], dim=1)  # [B, 2]
    #
    #     return fused_features, simple_weights