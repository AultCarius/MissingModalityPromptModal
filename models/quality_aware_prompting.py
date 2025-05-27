# Quality Aware Prompting
import random

from timm import create_model
from transformers import CLIPTextModel, CLIPTokenizer
import torch
from transformers import AutoModel
import torch.nn as nn
import torch.nn.functional as F

# class EnhancedModalityQualityEstimator(nn.Module):
#     """
#     增强型模态质量评估器 - 评估模态的质量、完整性和一致性
#     """
#
#     def __init__(self, image_dim, text_dim, hidden_dim=128):
#         super().__init__()
#
#         # 单模态质量评估网络
#         self.image_quality_net = nn.Sequential(
#             nn.Linear(image_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, 3),  # [质量, 完整性, 可靠性]
#             nn.Sigmoid()  # 输出范围为[0,1]
#         )
#
#         self.text_quality_net = nn.Sequential(
#             nn.Linear(text_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, 3),  # [质量, 完整性, 可靠性]
#             nn.Sigmoid()  # 输出范围为[0,1]
#         )
#
#         # 跨模态一致性评估网络
#         self.cross_consistency_net = nn.Sequential(
#             nn.Linear(image_dim + text_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, 1),
#             nn.Sigmoid()  # 输出范围为[0,1]
#         )
#
#         # 权重参数用于综合得到最终质量分数
#         self.quality_weights = nn.Parameter(torch.ones(3) / 3)
#         self.consistency_weight = nn.Parameter(torch.tensor(0.5))
#
#     def forward(self, image_feat, text_feat, missing_type=None):
#         """
#         前向传播函数
#
#         Args:
#             image_feat: 图像特征 [B, D_img]
#             text_feat: 文本特征 [B, D_txt]
#             missing_type: 缺失类型张量 [B], 0=none, 1=image, 2=text, 3=both
#
#         Returns:
#             字典，包含各项质量分数
#         """
#         batch_size = image_feat.size(0)
#         device = image_feat.device
#
#         # 初始化质量评估结果
#         image_quality = torch.zeros(batch_size, 3, device=device)  # [质量, 完整性, 可靠性]
#         text_quality = torch.zeros(batch_size, 3, device=device)  # [质量, 完整性, 可靠性]
#         cross_consistency = torch.zeros(batch_size, 1, device=device)
#
#         # 根据缺失类型调整特征评估
#         if missing_type is None:
#             # 如果没有提供缺失类型，假设所有样本都有完整模态
#             missing_mask = torch.zeros(batch_size, device=device)
#         else:
#             missing_mask = missing_type
#
#         # 对每个样本分别处理
#         for b in range(batch_size):
#             current_missing = missing_mask[b].item() if isinstance(missing_mask, torch.Tensor) else missing_mask
#
#             # 图像质量评估
#             if current_missing != 1 and current_missing != 3:  # 图像不缺失
#                 image_quality[b] = self.image_quality_net(image_feat[b:b + 1]).squeeze(0)
#                 # 原始模态设置完整性为1
#                 image_quality[b, 1] = 1.0
#             else:
#                 # 缺失模态完整性为0，其他指标由模型评估
#                 image_quality[b, 1] = 0.0
#                 # 如果是生成的特征，仍然可以评估质量和可靠性
#                 if torch.sum(torch.abs(image_feat[b])) > 1e-5:  # 如果特征不全为0
#                     temp_quality = self.image_quality_net(image_feat[b:b + 1]).squeeze(0)
#                     image_quality[b, 0] = temp_quality[0] * 0.7  # 生成特征的质量打折
#                     image_quality[b, 2] = temp_quality[2] * 0.7  # 生成特征的可靠性打折
#
#             # 文本质量评估
#             if current_missing != 2 and current_missing != 3:  # 文本不缺失
#                 text_quality[b] = self.text_quality_net(text_feat[b:b + 1]).squeeze(0)
#                 # 原始模态设置完整性为1
#                 text_quality[b, 1] = 1.0
#             else:
#                 # 缺失模态完整性为0，其他指标由模型评估
#                 text_quality[b, 1] = 0.0
#                 # 如果是生成的特征，仍然可以评估质量和可靠性
#                 if torch.sum(torch.abs(text_feat[b])) > 1e-5:  # 如果特征不全为0
#                     temp_quality = self.text_quality_net(text_feat[b:b + 1]).squeeze(0)
#                     text_quality[b, 0] = temp_quality[0] * 0.7  # 生成特征的质量打折
#                     text_quality[b, 2] = temp_quality[2] * 0.7  # 生成特征的可靠性打折
#
#             # 跨模态一致性评估 - 只有当两个模态都存在时才计算
#             if current_missing == 0 or (torch.sum(torch.abs(image_feat[b])) > 1e-5 and
#                                         torch.sum(torch.abs(text_feat[b])) > 1e-5):
#                 concat_feat = torch.cat([image_feat[b:b + 1], text_feat[b:b + 1]], dim=1)
#                 cross_consistency[b] = self.cross_consistency_net(concat_feat)
#
#         # 计算综合质量分数
#         # 使用softmax确保权重和为1
#         normalized_weights = F.softmax(self.quality_weights, dim=0)
#
#         # 对质量、完整性和可靠性进行加权平均
#         image_final_quality = torch.sum(image_quality * normalized_weights.view(1, 3), dim=1, keepdim=True)
#         text_final_quality = torch.sum(text_quality * normalized_weights.view(1, 3), dim=1, keepdim=True)
#
#         # 合并一致性得分（使用sigmoid确保权重在0-1之间）
#         consistency_w = torch.sigmoid(self.consistency_weight)
#
#         # 最终质量得分 = (1-w)*单模态得分 + w*一致性得分
#         image_final_quality = (1 - consistency_w) * image_final_quality + consistency_w * cross_consistency
#         text_final_quality = (1 - consistency_w) * text_final_quality + consistency_w * cross_consistency
#
#         return {
#             "image": {
#                 "quality": image_quality[:, 0:1],
#                 "completeness": image_quality[:, 1:2],
#                 "reliability": image_quality[:, 2:3],
#                 "final_score": image_final_quality
#             },
#             "text": {
#                 "quality": text_quality[:, 0:1],
#                 "completeness": text_quality[:, 1:2],
#                 "reliability": text_quality[:, 2:3],
#                 "final_score": text_final_quality
#             },
#             "cross_consistency": cross_consistency
#         }

# class EnhancedModalityQualityEstimator(nn.Module):
#     def __init__(self, image_dim, text_dim):
#         super().__init__()
#         self.image_quality = nn.Sequential(
#             nn.Linear(image_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)  # 输出3个质量维度
#         )
#
#         self.text_quality = nn.Sequential(
#             nn.Linear(text_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)  # 输出3个质量维度
#         )
#
#         self.cross_consistency = nn.Sequential(
#             nn.Linear(image_dim + text_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1)
#         )
#
#     def forward(self, image_feat, text_feat, missing_type=None):
#         """评估模态质量和跨模态一致性
#
#         Args:
#             image_feat: 图像特征
#             text_feat: 文本特征
#             missing_type: 缺失类型张量 (none=0, image=1, text=2, both=3)
#         """
#         batch_size = max(image_feat.size(0) if image_feat is not None else 0,
#                          text_feat.size(0) if text_feat is not None else 0)
#         device = image_feat.device if image_feat is not None else text_feat.device
#
#         # 初始化结果
#         results = {
#             'image': {'quality': None, 'final_score': None},
#             'text': {'quality': None, 'final_score': None},
#             'cross_consistency': None
#         }
#
#         # 检测零填充的张量
#         is_zero_image = torch.zeros(batch_size, dtype=torch.bool, device=device)
#         is_zero_text = torch.zeros(batch_size, dtype=torch.bool, device=device)
#
#         if image_feat is not None:
#             is_zero_image = torch.all(torch.abs(image_feat) < 1e-6, dim=1)
#         else:
#             is_zero_image = torch.ones(batch_size, dtype=torch.bool, device=device)
#
#         if text_feat is not None:
#             is_zero_text = torch.all(torch.abs(text_feat) < 1e-6, dim=1)
#         else:
#             is_zero_text = torch.ones(batch_size, dtype=torch.bool, device=device)
#
#         # 计算图像质量评分
#         if not is_zero_image.all():
#             # 只对非零图像计算质量
#             valid_image_feat = image_feat[~is_zero_image]
#             valid_image_quality = self.image_quality(valid_image_feat)
#             valid_image_final = torch.sigmoid(valid_image_quality.mean(dim=1, keepdim=True))
#
#             # 将结果分配回原始批次
#             image_quality = torch.zeros(batch_size, 3, device=device)
#             image_final = torch.zeros(batch_size, 1, device=device)
#
#             image_quality[~is_zero_image] = valid_image_quality
#             image_final[~is_zero_image] = valid_image_final
#
#             # 对于零图像，分配低质量分数
#             image_quality[is_zero_image] = torch.tensor([-2.0, -2.0, -2.0], device=device)
#             image_final[is_zero_image] = torch.tensor([0.1], device=device)  # 低但不为零
#
#             results['image']['quality'] = image_quality
#             results['image']['final_score'] = image_final
#         else:
#             # 所有图像都缺失
#             results['image']['quality'] = torch.full((batch_size, 3), -2.0, device=device)
#             results['image']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)
#
#         # 计算文本质量评分（类似逻辑）
#         if not is_zero_text.all():
#             valid_text_feat = text_feat[~is_zero_text]
#             valid_text_quality = self.text_quality(valid_text_feat)
#             valid_text_final = torch.sigmoid(valid_text_quality.mean(dim=1, keepdim=True))
#
#             text_quality = torch.zeros(batch_size, 3, device=device)
#             text_final = torch.zeros(batch_size, 1, device=device)
#
#             text_quality[~is_zero_text] = valid_text_quality
#             text_final[~is_zero_text] = valid_text_final
#
#             text_quality[is_zero_text] = torch.tensor([-2.0, -2.0, -2.0], device=device)
#             text_final[is_zero_text] = torch.tensor([0.1], device=device)
#
#             results['text']['quality'] = text_quality
#             results['text']['final_score'] = text_final
#         else:
#             results['text']['quality'] = torch.full((batch_size, 3), -2.0, device=device)
#             results['text']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)
#
#         # 计算跨模态一致性
#         cross_consistency = torch.zeros(batch_size, 1, device=device)
#
#         # 只对既有图像又有文本的样本计算一致性
#         both_valid = ~(is_zero_image | is_zero_text)
#         if both_valid.any():
#             valid_image = image_feat[both_valid]
#             valid_text = text_feat[both_valid]
#
#             # 计算两个模态特征的一致性
#             concat_feat = torch.cat([valid_image, valid_text], dim=1)
#             valid_consistency = torch.sigmoid(self.cross_consistency(concat_feat))
#
#             # 重新分配回原始批次
#             cross_consistency[both_valid] = valid_consistency
#
#         # 对于缺失模态的样本，分配默认一致性分数
#         cross_consistency[is_zero_image | is_zero_text] = 0.5  # 中等一致性分数
#
#         results['cross_consistency'] = cross_consistency
#
#         return results

class EnhancedModalityQualityEstimator(nn.Module):
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

        # Reference statistics for normalization
        self.register_buffer('image_mean', torch.zeros(1, image_dim))
        self.register_buffer('image_std', torch.ones(1, image_dim))
        self.register_buffer('text_mean', torch.zeros(1, text_dim))
        self.register_buffer('text_std', torch.ones(1, text_dim))

        # Learnable importance weights for quality dimensions
        self.image_weights = nn.Parameter(torch.ones(5) / 5)
        self.text_weights = nn.Parameter(torch.ones(5) / 5)
        self.consistency_weights = nn.Parameter(torch.ones(3) / 3)

        # Modality fusion weights and importance
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))
        self.modality_importance = nn.Parameter(torch.FloatTensor([1.0, 1.0]))  # [image, text]

        # Performance tracking for adaptive weighting
        self.register_buffer('performance_stats', torch.zeros(2))  # [image_missing_perf, text_missing_perf]

        # Mode flags for different behaviors
        self.evaluation_mode = False  # Set to True during evaluation for different scoring behavior

    def update_reference_statistics(self, image_feats, text_feats):
        """Update reference statistics for feature normalization"""
        if image_feats is not None and image_feats.numel() > 0:
            with torch.no_grad():
                # Only update if we have valid data
                if not torch.isnan(image_feats).any() and not torch.isinf(image_feats).any():
                    self.image_mean = image_feats.mean(0, keepdim=True).detach()
                    self.image_std = image_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

        if text_feats is not None and text_feats.numel() > 0:
            with torch.no_grad():
                # Only update if we have valid data
                if not torch.isnan(text_feats).any() and not torch.isinf(text_feats).any():
                    self.text_mean = text_feats.mean(0, keepdim=True).detach()
                    self.text_std = text_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

    def normalize_features(self, image_feats, text_feats):
        """Normalize features using stored statistics"""
        norm_img_feats = None
        if image_feats is not None:
            # 修改为更温和的归一化
            norm_img_feats = image_feats / (torch.norm(image_feats, dim=1, keepdim=True) + 1e-6)

        norm_txt_feats = None
        if text_feats is not None:
            # 修改为更温和的归一化
            norm_txt_feats = text_feats / (torch.norm(text_feats, dim=1, keepdim=True) + 1e-6)

        return norm_img_feats, norm_txt_feats

    def update_performance_stats(self, img_missing_perf, txt_missing_perf):
        """Update modality performance statistics for adaptive weighting"""
        # Convert inputs to tensors if needed
        if not isinstance(img_missing_perf, torch.Tensor):
            img_missing_perf = torch.tensor(img_missing_perf, device=self.performance_stats.device)
        if not isinstance(txt_missing_perf, torch.Tensor):
            txt_missing_perf = torch.tensor(txt_missing_perf, device=self.performance_stats.device)

        # Update performance stats
        self.performance_stats = torch.stack([img_missing_perf, txt_missing_perf])

        # Calculate adaptive importance weights based on performance
        perf_ratio = txt_missing_perf / (img_missing_perf + 1e-8)  # Prevent division by zero
        perf_ratio = torch.clamp(perf_ratio, 0.8, 1.25)  # Limit to reasonable range

        # Create new balanced weights
        new_importance = torch.zeros_like(self.modality_importance)
        new_importance[0] = perf_ratio  # Image importance
        new_importance[1] = 2.0 - perf_ratio  # Text importance

        # Update importance weights
        with torch.no_grad():
            self.modality_importance.copy_(new_importance)

    def forward(self, image_feat, text_feat, missing_type=None):
        """
        Forward pass to evaluate modality quality and consistency

        Args:
            image_feat: Image features [B, D_img]
            text_feat: Text features [B, D_txt]
            missing_type: Missing modality type (0=none, 1=image, 2=text, 3=both)

        Returns:
            Dictionary with quality scores for each modality and cross-consistency
        """
        batch_size = max(image_feat.size(0) if image_feat is not None else 0,
                         text_feat.size(0) if text_feat is not None else 0)
        device = image_feat.device if image_feat is not None else text_feat.device

        # Determine missing modalities based on missing_type or zero-detection
        if missing_type is not None:
            is_image_missing = (missing_type == 1) | (missing_type == 3)
            is_text_missing = (missing_type == 2) | (missing_type == 3)
        else:
            is_image_missing = torch.sum(torch.abs(image_feat), dim=1) < 1e-6 if image_feat is not None else torch.ones(
                batch_size, dtype=torch.bool, device=device)
            is_text_missing = torch.sum(torch.abs(text_feat), dim=1) < 1e-6 if text_feat is not None else torch.ones(
                batch_size, dtype=torch.bool, device=device)

        # Normalize features for more stable evaluation
        norm_img, norm_txt = self.normalize_features(image_feat, text_feat)

        # Initialize results dictionary
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
            # Get quality dimensions
            img_quality = self.image_quality(norm_img)

            # Apply confidence mask based on whether image is real or generated
            conf_mask = torch.ones(batch_size, 1, device=device)

            # In evaluation mode, we use different scoring for real vs generated
            if self.evaluation_mode:
                # For real images, use higher confidence (1.0)
                # For generated images with real text, use medium confidence (0.7)
                # For both missing, use lower confidence (0.5)
                conf_mask[is_image_missing & ~is_text_missing] = 0.7  # Generated image, real text
                conf_mask[is_image_missing & is_text_missing] = 0.5  # Both generated
            else:
                # During training, use higher confidence for generated features to encourage learning
                conf_mask[is_image_missing & ~is_text_missing] = 0.85  # Generated image, real text
                conf_mask[is_image_missing & is_text_missing] = 0.7  # Both generated

            # Apply confidence mask to quality scores
            img_quality = img_quality * conf_mask

            # Calculate final score using learned weights
            img_weights = F.softmax(self.image_weights, dim=0)
            img_final = torch.sigmoid(torch.sum(img_quality * img_weights, dim=1, keepdim=True))

            # Special adjustment for real images - boost their scores during evaluation
            if self.evaluation_mode:
                real_img_mask = ~is_image_missing
                img_final[real_img_mask] = img_final[real_img_mask] * 1.1
                img_final = torch.clamp(img_final, 0.0, 1.0)  # Ensure in [0,1] range

            results['image']['quality'] = img_quality
            results['image']['final_score'] = img_final
        else:
            # Default low scores for missing modality
            results['image']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['image']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # Text quality assessment (similar approach)
        if norm_txt is not None:
            txt_quality = self.text_quality(norm_txt)

            # Apply confidence mask based on whether text is real or generated
            conf_mask = torch.ones(batch_size, 1, device=device)

            # In evaluation mode, we use different scoring for real vs generated
            if self.evaluation_mode:
                # For real text, use higher confidence (1.0)
                # For generated text with real image, use medium confidence (0.7)
                # For both missing, use lower confidence (0.5)
                conf_mask[is_text_missing & ~is_image_missing] = 0.7  # Generated text, real image
                conf_mask[is_text_missing & is_image_missing] = 0.5  # Both generated
            else:
                # During training, use higher confidence for generated features to encourage learning
                conf_mask[is_text_missing & ~is_image_missing] = 0.85  # Generated text, real image
                conf_mask[is_text_missing & is_image_missing] = 0.7  # Both generated

            # Apply confidence mask to quality scores
            txt_quality = txt_quality * conf_mask

            # Calculate final score using learned weights
            txt_weights = F.softmax(self.text_weights, dim=0)
            txt_final = torch.sigmoid(torch.sum(txt_quality * txt_weights, dim=1, keepdim=True))

            # Special adjustment for real text - boost their scores during evaluation
            if self.evaluation_mode:
                real_txt_mask = ~is_text_missing
                txt_final[real_txt_mask] = txt_final[real_txt_mask] * 1.1
                txt_final = torch.clamp(txt_final, 0.0, 1.0)  # Ensure in [0,1] range

            results['text']['quality'] = txt_quality
            results['text']['final_score'] = txt_final
        else:
            # Default low scores for missing modality
            results['text']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['text']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # Cross-modal consistency assessment
        if norm_img is not None and norm_txt is not None:
            # Concatenate features for consistency evaluation
            concat_feat = torch.cat([norm_img, norm_txt], dim=1)
            consistency_dims = self.cross_consistency(concat_feat)

            # Apply confidence mask based on modality presence
            conf_mask = torch.ones(batch_size, 1, device=device)

            # Separate treatment for different missing pattern combinations
            both_missing = is_image_missing & is_text_missing
            img_missing_only = is_image_missing & ~is_text_missing
            txt_missing_only = is_text_missing & ~is_image_missing

            # Apply appropriate confidence values
            conf_mask[both_missing] = 0.5  # Both generated
            conf_mask[img_missing_only] = 0.8  # Only image generated
            conf_mask[txt_missing_only] = 0.7  # Only text generated

            # Apply confidence mask to consistency dimensions
            consistency_dims = consistency_dims * conf_mask

            # Calculate final consistency score using learned weights
            consistency_weights = F.softmax(self.consistency_weights, dim=0)
            consistency = torch.sigmoid(torch.sum(consistency_dims * consistency_weights, dim=1, keepdim=True))

            results['cross_consistency'] = consistency
        else:
            # Default medium score when a modality is missing entirely
            results['cross_consistency'] = torch.full((batch_size, 1), 0.5, device=device)

        # Modality score fusion based on consistency
        if norm_img is not None and norm_txt is not None:
            # Base fusion weight
            w_base = torch.sigmoid(self.fusion_weight)

            # Create fusion weight tensor for [image_weight, text_weight, consistency_weight]
            fusion_weights = torch.zeros(batch_size, 3, device=device)

            # Normal samples (no missing modalities)
            normal_mask = (~is_image_missing) & (~is_text_missing)
            fusion_weights[normal_mask, 0] = (1 - w_base) * self.modality_importance[0]  # Image weight
            fusion_weights[normal_mask, 1] = (1 - w_base) * self.modality_importance[1]  # Text weight
            fusion_weights[normal_mask, 2] = w_base  # Consistency weight

            # Image missing, text present
            img_missing_mask = is_image_missing & (~is_text_missing)
            fusion_weights[img_missing_mask, 0] = 0.3  # Lower weight for generated image
            fusion_weights[img_missing_mask, 1] = 0.6  # Higher weight for real text
            fusion_weights[img_missing_mask, 2] = 0.1  # Lower consistency weight

            # Text missing, image present
            txt_missing_mask = (~is_image_missing) & is_text_missing
            fusion_weights[txt_missing_mask, 0] = 0.6  # Higher weight for real image
            fusion_weights[txt_missing_mask, 1] = 0.3  # Lower weight for generated text
            fusion_weights[txt_missing_mask, 2] = 0.1  # Lower consistency weight

            # Both missing
            both_missing_mask = is_image_missing & is_text_missing
            fusion_weights[both_missing_mask, 0] = 0.45  # Medium weight for generated image
            fusion_weights[both_missing_mask, 1] = 0.45  # Medium weight for generated text
            fusion_weights[both_missing_mask, 2] = 0.1  # Lower consistency weight

            # Normalize weights to sum to 1
            fusion_weights = fusion_weights / (fusion_weights.sum(dim=1, keepdim=True) + 1e-8)

            # Apply fusion weights to compute final scores
            image_final = (fusion_weights[:, 0:1] * results['image']['final_score'] +
                           fusion_weights[:, 2:3] * results['cross_consistency'])

            text_final = (fusion_weights[:, 1:2] * results['text']['final_score'] +
                          fusion_weights[:, 2:3] * results['cross_consistency'])

            # Update the results
            results['image']['final_score'] = image_final
            results['text']['final_score'] = text_final

            # Store fusion weights for analysis
            results['fusion_weights'] = fusion_weights

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
            nn.Linear(fusion_dim *2 , fusion_dim),# TODO:修改了fusion_dim*2来适配简易加权融合
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

        # 添加图像缺失补偿网络 - 新增
        self.image_missing_compensator = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, fusion_dim)
        )


    def forward(self, image_feat, text_feat, quality_scores=None):
        batch_size = image_feat.size(0)
        device = image_feat.device

        # 在开始时添加详细日志
        # log_details = (torch.rand(1).item() < 0.01)  # 1%的随机概率记录详情
        log_details = 0  # 1%的随机概率记录详情

        if log_details:
            print(f"\n=== FUSION TRACING (batch_size={batch_size}) ===")
            print(f"Image feature stats: shape={image_feat.shape}, mean={image_feat.mean().item():.6f}, "
                  f"std={image_feat.std().item():.6f}, norm={torch.norm(image_feat, dim=1).mean().item():.6f}")
            print(f"Text feature stats: shape={text_feat.shape}, mean={text_feat.mean().item():.6f}, "
                  f"std={text_feat.std().item():.6f}, norm={torch.norm(text_feat, dim=1).mean().item():.6f}")

            if quality_scores is not None:
                img_q = quality_scores['image']['final_score'].mean().item()
                txt_q = quality_scores['text']['final_score'].mean().item()
                cons = quality_scores['cross_consistency'].mean().item()
                print(f"Quality scores: image={img_q:.6f}, text={txt_q:.6f}, consistency={cons:.6f}")

        # 原始forward函数代码
        img_proj = self.image_proj(image_feat)
        txt_proj = self.text_proj(text_feat)

        if log_details:
            print(f"Projected features: img_proj={img_proj.shape}, img_proj_mean={img_proj.mean().item():.6f}, "
                  f"txt_proj={txt_proj.shape}, txt_proj_mean={txt_proj.mean().item():.6f}")

        # 准备质量向量
        if quality_scores is None or 'image' not in quality_scores or 'text' not in quality_scores:
            quality_vector = torch.ones(batch_size, 2, device=device) * 0.5

            if log_details:
                print("Using default quality vector [0.5, 0.5]")
        else:
            quality_vector = torch.cat([
                quality_scores['image']['final_score'],
                quality_scores['text']['final_score']
            ], dim=1)

            if log_details:
                print(f"Quality vector: {quality_vector.mean(dim=0).tolist()}")
        # ---------

        # 检测图像缺失情况 - 使用质量分数比例
        if quality_scores is not None:
            # 获取图像和文本质量分数
            img_quality = quality_scores['image']['final_score']  # [batch_size, 1]
            txt_quality = quality_scores['text']['final_score']  # [batch_size, 1]

            # 计算质量比例
            quality_ratio = img_quality / (txt_quality + 1e-8)  # 防止除零

            # 使用简单阈值检测图像缺失 - 图像质量显著低于文本质量
            is_img_missing = (quality_ratio < 0.7).float()  # 阈值0.7
        else:
            # 默认情况下假设没有缺失
            is_img_missing = torch.zeros(batch_size, 1, device=device)

        # 生成注意力权重
        attn_weights = self.quality_attn_weights(quality_vector)

        if log_details:
            print(f"Attention weights: {attn_weights.mean(dim=0).tolist()}")

        # 堆叠特征作为序列 [img, txt]
        features = torch.stack([img_proj, txt_proj], dim=1)

        # 应用质量感知的跨模态注意力
        attn_output, _ = self.cross_attn(
            query=features,
            key=features,
            value=features
        )

        # 提取各模态的表示
        img_attn = attn_output[:, 0]
        txt_attn = attn_output[:, 1]

        # 检查是否有样本级别的模态权重信息
        if quality_scores is not None and 'sample_weights' in quality_scores:
            sample_weights = quality_scores['sample_weights']  # [batch_size, 3]

            # 提取缺失信息
            missing_info = quality_scores.get('missing_info', {})
            is_image_missing = missing_info.get('image', torch.zeros(batch_size, dtype=torch.bool, device=device))
            is_text_missing = missing_info.get('text', torch.zeros(batch_size, dtype=torch.bool, device=device))

            # 创建调整系数张量，而不是循环修改
            img_adjust = torch.ones(batch_size, device=device)
            txt_adjust = torch.ones(batch_size, device=device)

            # 对于图像缺失样本
            img_adjust[is_image_missing] = 0.9  # 轻微降低
            txt_adjust[is_image_missing] = 1.1  # 轻微提高

            # 对于文本缺失样本 - 同样应用适度的平衡
            img_adjust[is_text_missing] = 0.9  # 轻微降低
            txt_adjust[is_text_missing] = 1.1  # 轻微提高

            # 创建新的注意力输出，避免原地修改
            new_img_attn = img_attn * img_adjust.unsqueeze(1)
            new_txt_attn = txt_attn * txt_adjust.unsqueeze(1)

            # 使用新的注意力输出
            img_attn = new_img_attn
            txt_attn = new_txt_attn

        if log_details:
            print(f"Attention output: img_attn={img_attn.mean().item():.6f}, txt_attn={txt_attn.mean().item():.6f}")

        # 计算质量门控值
        quality_gate = self.quality_gate(quality_vector)

        if log_details:
            print(f"Quality gate: {quality_gate.mean().item():.6f}")
        #---------
        # 对图像缺失情况应用特殊处理
        missing_compensation = torch.zeros_like(img_attn)
        for i in range(batch_size):
            if is_img_missing[i] > 0.5:  # 阈值0.5判断为缺失
                # 应用补偿
                missing_compensation[i] = self.image_missing_compensator(img_attn[i])

        # 添加补偿到图像特征
        img_attn = img_attn + missing_compensation * is_img_missing

        # 基于质量的加权平均
        weighted_img = img_proj * quality_scores['image']['final_score']
        weighted_txt = txt_proj * quality_scores['text']['final_score']

        if log_details:
            print(f"Weighted features: img={weighted_img.mean().item():.6f}, txt={weighted_txt.mean().item():.6f}")

        # 计算质量加权的特征
        quality_weighted = (weighted_img + weighted_txt) / (
                quality_scores['image']['final_score'] + quality_scores['text']['final_score'] + 1e-8
        )

        if log_details:
            print(f"Quality weighted: {quality_weighted.mean().item():.6f}")

        # 注意力输出和质量加权之间的插值
        attn_weighted = (img_attn + txt_attn) / 2
        fused_features = quality_gate * quality_weighted + (1 - quality_gate) * attn_weighted

        if log_details:
            print(f"Attention weighted: {attn_weighted.mean().item():.6f}")
            print(f"Final fused features: {fused_features.mean().item():.6f}")

        # 添加特征规范化 - 保证特征分布合理?
        fused_features = F.layer_norm(fused_features, fused_features.shape[1:])

        # 应用最终投影
        output_features = self.output_proj(torch.cat([fused_features, attn_weighted], dim=1))

        if log_details:
            print(f"Output features: {output_features.mean().item():.6f}")
            print("=== END FUSION TRACING ===\n")

        # 为可视化目的返回简单权重
        simple_weights = torch.cat([
            quality_scores['image']['final_score'],
            quality_scores['text']['final_score']
        ], dim=1)

        return output_features, simple_weights

    def forward1(self, image_feat, text_feat, quality_scores=None):
        """
        Simplified weighted fusion using final_score for image and text modalities.
        """
        batch_size = image_feat.size(0)
        device = image_feat.device

        # 1. 投影到共同维度（如果你想保留）
        img_proj = self.image_proj(image_feat)  # [B, D]
        txt_proj = self.text_proj(text_feat)  # [B, D]

        # 2. 获取 final_score 或默认权重
        if quality_scores is None or 'image' not in quality_scores or 'text' not in quality_scores:
            img_weight = torch.ones(batch_size, 1, device=device) * 0.5
            txt_weight = torch.ones(batch_size, 1, device=device) * 0.5
        else:
            img_weight = quality_scores['image']['final_score']  # [B, 1]
            txt_weight = quality_scores['text']['final_score']  # [B, 1]

        # 3. 加权融合：注意力加权求和后归一化
        weights_sum = img_weight + txt_weight + 1e-8  # 避免除零
        fused_feat = (img_proj * img_weight + txt_proj * txt_weight) / weights_sum  # [B, D]

        # 4. 可选：归一化后再投影输出
        fused_feat = F.layer_norm(fused_feat, fused_feat.shape[1:])
        output_feat = self.output_proj(fused_feat)  # [B, D_out]

        return output_feat, torch.cat([img_weight, txt_weight], dim=1)


class ImprovedQualityAwareFeatureFusion(nn.Module):
    """改进的质量感知特征融合"""

    def __init__(self, image_dim, text_dim, fusion_dim, num_heads=8):
        super().__init__()

        # 特征投影
        self.image_proj = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )

        # 质量感知的融合权重生成器
        self.fusion_weight_generator = nn.Sequential(
            nn.Linear(3, 64),  # [img_quality, txt_quality, consistency]
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # [img_weight, txt_weight, interaction_weight]
            nn.Softmax(dim=1)
        )

        # 跨模态交互模块
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 最终投影（不再需要维度翻倍）
        self.output_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # 缺失模态补偿网络
        self.missing_compensator = nn.ModuleDict({
            'image': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Linear(fusion_dim * 2, fusion_dim)
            ),
            'text': nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim * 2),
                nn.GELU(),
                nn.Linear(fusion_dim * 2, fusion_dim)
            )
        })

    def forward(self, image_feat, text_feat, quality_scores=None):
        batch_size = image_feat.size(0)
        device = image_feat.device

        # 投影特征到统一空间
        img_proj = self.image_proj(image_feat)
        txt_proj = self.text_proj(text_feat)

        # 获取质量信息
        if quality_scores is None:
            quality_vector = torch.ones(batch_size, 3, device=device) * 0.5
        else:
            quality_vector = torch.cat([
                quality_scores['image']['final_score'],
                quality_scores['text']['final_score'],
                quality_scores['cross_consistency']
            ], dim=1)

        # 检测缺失模态
        img_quality = quality_vector[:, 0]
        txt_quality = quality_vector[:, 1]

        # 动态阈值
        quality_threshold = 0.4
        is_image_missing = img_quality < quality_threshold
        is_text_missing = txt_quality < quality_threshold

        # 缺失模态补偿
        compensated_img = img_proj.clone()
        compensated_txt = txt_proj.clone()

        if is_image_missing.any():
            img_compensation = self.missing_compensator['image'](txt_proj[is_image_missing])
            compensated_img[is_image_missing] = (
                    0.2 * img_proj[is_image_missing] + 0.8 * img_compensation
            )

        if is_text_missing.any():
            txt_compensation = self.missing_compensator['text'](img_proj[is_text_missing])
            compensated_txt[is_text_missing] = (
                    0.2 * txt_proj[is_text_missing] + 0.8 * txt_compensation
            )

        # 生成融合权重
        fusion_weights = self.fusion_weight_generator(quality_vector)
        img_weight = fusion_weights[:, 0:1]
        txt_weight = fusion_weights[:, 1:2]
        interaction_weight = fusion_weights[:, 2:3]

        # 跨模态交互（只计算一次）
        stacked_feats = torch.stack([compensated_img, compensated_txt], dim=1)
        interaction_output, _ = self.cross_modal_attn(
            query=stacked_feats,
            key=stacked_feats,
            value=stacked_feats
        )
        interaction_feat = interaction_output.mean(dim=1)  # 池化得到交互特征

        # 三路融合（不重复使用任何特征）
        fused_features = (
                img_weight * compensated_img +
                txt_weight * compensated_txt +
                interaction_weight * interaction_feat
        )

        # 最终投影
        output_features = self.output_proj(fused_features)

        # 返回调试信息
        debug_info = {
            'fusion_weights': fusion_weights,
            'quality_vector': quality_vector,
            'missing_flags': {
                'image': is_image_missing,
                'text': is_text_missing
            }
        }

        return output_features, debug_info