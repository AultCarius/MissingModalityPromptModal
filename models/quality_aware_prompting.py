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

        # 图像模态质量评估器
        self.image_quality = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5)  # [清晰度、完整性、信息量、置信度、细节]
        )

        # 文本模态质量评估器
        self.text_quality = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 5)  # [连贯性、相关性、信息量、置信度、细节]
        )

        # 跨模态一致性评估器
        self.cross_consistency = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 3)  # [语义一致性、结构一致性、上下文一致性]
        )

        # 可学习的参考特征统计量（用于标准化）
        self.register_buffer('image_mean', torch.zeros(1, image_dim))
        self.register_buffer('image_std', torch.ones(1, image_dim))
        self.register_buffer('text_mean', torch.zeros(1, text_dim))
        self.register_buffer('text_std', torch.ones(1, text_dim))

        # 可学习的模态质量维度权重
        self.image_weights = nn.Parameter(torch.ones(5) / 5)
        self.text_weights = nn.Parameter(torch.ones(5) / 5)
        self.consistency_weights = nn.Parameter(torch.ones(3) / 3)

        # 生成质量预测器（估计生成图像和文本的可信度）
        self.generation_quality = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # [图像置信度，文本置信度]
        )

        # 模态融合时的一致性权重（可学习）
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))  # 控制单模态质量与跨模态一致性的平衡

    def update_reference_statistics(self, image_feats, text_feats):
        """更新参考统计量（训练阶段调用，用于标准化）"""
        if image_feats is not None and len(image_feats) > 0:
            with torch.no_grad():
                self.image_mean = image_feats.mean(0, keepdim=True).detach()
                self.image_std = image_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

        if text_feats is not None and len(text_feats) > 0:
            with torch.no_grad():
                self.text_mean = text_feats.mean(0, keepdim=True).detach()
                self.text_std = text_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

    def normalize_features(self, image_feats, text_feats):
        """使用存储的统计量对特征进行标准化"""
        norm_img_feats = None
        if image_feats is not None:
            norm_img_feats = (image_feats - self.image_mean) / self.image_std

        norm_txt_feats = None
        if text_feats is not None:
            norm_txt_feats = (text_feats - self.text_mean) / self.text_std

        return norm_img_feats, norm_txt_feats

    def forward(self, image_feat, text_feat, missing_type=None):
        # 获取批大小和设备信息
        batch_size = max(image_feat.size(0) if image_feat is not None else 0,
                         text_feat.size(0) if text_feat is not None else 0)
        device = image_feat.device if image_feat is not None else text_feat.device

        # 判断哪些模态为生成（缺失）模态
        if missing_type is not None:
            is_image_missing = (missing_type == 1) | (missing_type == 3)
            is_text_missing = (missing_type == 2) | (missing_type == 3)
        else:
            is_image_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)
            is_text_missing = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # 对图像和文本特征进行标准化，提高质量估计稳定性
        norm_img, norm_txt = self.normalize_features(image_feat, text_feat)

        # 初始化输出结果结构
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
            img_quality = self.image_quality(norm_img)

            # 对生成特征打折惩罚
            if is_image_missing.any():
                conf_mask = torch.ones(batch_size, 1, device=device)
                conf_mask[is_image_missing] = 0.7  # 生成图像的得分降低
                img_quality = img_quality * conf_mask

            img_weights = F.softmax(self.image_weights, dim=0)
            img_final = torch.sigmoid(torch.sum(img_quality * img_weights, dim=1, keepdim=True))

            results['image']['quality'] = img_quality
            results['image']['final_score'] = img_final
        else:
            # 缺失图像模态时设定为较低分数
            results['image']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['image']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # 文本质量评估（同上）
        if norm_txt is not None:
            txt_quality = self.text_quality(norm_txt)

            if is_text_missing.any():
                conf_mask = torch.ones(batch_size, 1, device=device)
                conf_mask[is_text_missing] = 0.7
                txt_quality = txt_quality * conf_mask

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

            # 处理生成模态影响
            if is_image_missing.any() or is_text_missing.any():
                both_generated = is_image_missing & is_text_missing
                one_generated = (is_image_missing | is_text_missing) & ~both_generated

                conf_mask = torch.ones(batch_size, 1, device=device)
                conf_mask[one_generated] = 0.7
                conf_mask[both_generated] = 0.5

                consistency_dims = consistency_dims * conf_mask

            consistency_weights = F.softmax(self.consistency_weights, dim=0)
            consistency = torch.sigmoid(torch.sum(consistency_dims * consistency_weights, dim=1, keepdim=True))

            results['cross_consistency'] = consistency
        else:
            # 若任一模态缺失，则一致性为中性值 0.5
            results['cross_consistency'] = torch.full((batch_size, 1), 0.5, device=device)

        # 在训练阶段，根据一致性调整单模态评分
        if self.training:
            w = torch.sigmoid(self.fusion_weight)

            if norm_img is not None and norm_txt is not None:
                results['image']['final_score'] = (1 - w) * results['image']['final_score'] + w * results['cross_consistency']
                results['text']['final_score'] = (1 - w) * results['text']['final_score'] + w * results['cross_consistency']

        return results



import torch
import torch.nn as nn

class QualityAwareFeatureFusion(nn.Module):
    def __init__(self, image_dim, text_dim, fusion_dim, num_heads=4):
        super().__init__()

        # 质量加权的特征投影
        self.image_proj = nn.Linear(image_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.num_heads = num_heads

        # 质量分数投影
        self.quality_proj = nn.Sequential(
            nn.Linear(2, 64),  # 简化处理：仅使用最终分数
            nn.GELU(),
            nn.Linear(64, num_heads)
        )

        # 用于质量感知跨模态融合的多头注意力机制
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 计算输出投影的输入维度
        # 每个模态贡献 fusion_dim，并且有两个模态（原始 + 注意力后的）
        output_input_dim = fusion_dim * 2 * 2  # 2 个模态 x 2（原始 + 注意力后）x fusion_dim

        # 输出投影模块，输入维度已校正
        self.output_proj = nn.Sequential(
            nn.Linear(output_input_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # 当质量分数不可用时的默认权重
        self.default_weights = nn.Parameter(torch.ones(1, 2) / 2)

    def forward(self, image_feat, text_feat, quality_scores=None):
        batch_size = image_feat.size(0)
        device = image_feat.device
        fusion_dim = self.image_proj.out_features  # 获取融合维度

        # 将图像和文本特征映射到融合空间
        img_proj = self.image_proj(image_feat)  # [B, fusion_dim]
        txt_proj = self.text_proj(text_feat)    # [B, fusion_dim]

        # 如果质量分数不存在，使用默认加权
        if quality_scores is None or 'image' not in quality_scores or 'text' not in quality_scores:
            attn_weights = self.default_weights.expand(batch_size, 2)
        else:
            # 检查质量分数
            img_score = None
            txt_score = None

            if 'final_score' in quality_scores['image']:
                img_score = quality_scores['image']['final_score']

            if 'final_score' in quality_scores['text']:
                txt_score = quality_scores['text']['final_score']

            if img_score is None or txt_score is None:
                attn_weights = self.default_weights.expand(batch_size, 2)
            else:
                # 构建仅包含最终质量分数的简单质量向量
                quality_vector = torch.cat([
                    img_score,
                    txt_score
                ], dim=1)

                # 将质量向量投影为注意力权重
                attn_weights = torch.softmax(self.quality_proj(quality_vector), dim=1)

        # 为多头注意力准备输入
        # 将两个模态堆叠为序列：[img, txt]
        features = torch.stack([img_proj, txt_proj], dim=1)  # [B, 2, fusion_dim]

        # 检查融合维度是否能被头数整除
        if fusion_dim % self.cross_attn.num_heads != 0:
            print(f"  WARNING: fusion_dim ({fusion_dim}) is not divisible by num_heads ({self.cross_attn.num_heads})")
            print(f"  fusion_dim % num_heads = {fusion_dim % self.cross_attn.num_heads}")

        # 应用跨模态注意力机制（带质量感知权重）
        try:
            attn_output, _ = self.cross_attn(
                query=features,
                key=features,
                value=features,
                need_weights=False
            )
        except Exception as e:
            print(f"  ERROR in cross_attn: {str(e)}")
            # 若注意力计算失败，使用原始特征作为备选
            attn_output = features.clone()

        # 拉平成特征以便后续连接
        features_flat = features.reshape(batch_size, -1)        # [B, 2*fusion_dim]
        attn_output_flat = attn_output.reshape(batch_size, -1)  # [B, 2*fusion_dim]

        # 将原始特征与注意力输出进行拼接
        concat_features = torch.cat([features_flat, attn_output_flat], dim=1)  # [B, 4*fusion_dim]

        # 最终投影，输出融合后的特征
        fused_features = self.output_proj(concat_features)  # [B, fusion_dim]

        # 构造简单的注意力权重返回（用于可视化或加权）
        simple_weights = torch.zeros(batch_size, 2, device=device)
        simple_weights[:, 0] = attn_weights.mean(dim=1)      # 图像的平均注意力权重
        simple_weights[:, 1] = 1 - simple_weights[:, 0]      # 文本的注意力权重为剩余部分

        return fused_features, simple_weights


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