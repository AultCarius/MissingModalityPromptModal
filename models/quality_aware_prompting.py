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

        # Learnable reference vectors
        self.register_buffer('image_mean', torch.zeros(1, image_dim))
        self.register_buffer('image_std', torch.ones(1, image_dim))
        self.register_buffer('text_mean', torch.zeros(1, text_dim))
        self.register_buffer('text_std', torch.ones(1, text_dim))

        # Learnable importance weights
        self.image_weights = nn.Parameter(torch.ones(5) / 5)
        self.text_weights = nn.Parameter(torch.ones(5) / 5)
        self.consistency_weights = nn.Parameter(torch.ones(3) / 3)

        # Generation confidence predictor
        self.generation_quality = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)  # [image_confidence, text_confidence]
        )

        # Modality fusion weights
        self.fusion_weight = nn.Parameter(torch.tensor([0.5]))  # Balance between individual quality and consistency

    def update_reference_statistics(self, image_feats, text_feats):
        """Update reference statistics for normalization (call during training)"""
        if image_feats is not None and len(image_feats) > 0:
            with torch.no_grad():
                self.image_mean = image_feats.mean(0, keepdim=True).detach()
                self.image_std = image_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

        if text_feats is not None and len(text_feats) > 0:
            with torch.no_grad():
                self.text_mean = text_feats.mean(0, keepdim=True).detach()
                self.text_std = text_feats.std(0, keepdim=True).detach().clamp(min=1e-6)

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
        batch_size = max(image_feat.size(0) if image_feat is not None else 0,
                         text_feat.size(0) if text_feat is not None else 0)
        device = image_feat.device if image_feat is not None else text_feat.device

        # Track which features are real vs. generated
        if missing_type is not None:
            is_image_missing = (missing_type == 1) | (missing_type == 3)
            is_text_missing = (missing_type == 2) | (missing_type == 3)
        else:
            # 修改: 如果没有提供缺失类型，检测零填充特征
            is_image_missing = torch.sum(torch.abs(image_feat),
                                         dim=(1, 2)) < 1e-6 if image_feat is not None else torch.ones(batch_size,
                                                                                                      dtype=torch.bool,
                                                                                                      device=device)
            is_text_missing = torch.sum(torch.abs(text_feat),
                                        dim=(1, 2)) < 1e-6 if text_feat is not None else torch.ones(batch_size,
                                                                                                    dtype=torch.bool,
                                                                                                    device=device)

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
            # Get quality dimensions
            img_quality = self.image_quality(norm_img)

            # Apply penalties for generated features
            if is_image_missing.any():
                # Create confidence mask (1.0 for real, 0.7 for generated)
                conf_mask = torch.ones(batch_size, 1, device=device)
                conf_mask[is_image_missing] = 0.7

                # Apply to quality scores
                img_quality = img_quality * conf_mask

            # Calculate final score using learned weights
            img_weights = F.softmax(self.image_weights, dim=0)
            img_final = torch.sigmoid(torch.sum(img_quality * img_weights, dim=1, keepdim=True))

            results['image']['quality'] = img_quality
            results['image']['final_score'] = img_final
        else:
            # Default low scores for missing modality
            results['image']['quality'] = torch.full((batch_size, 5), -1.0, device=device)
            results['image']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # Text quality assessment (similar approach)
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

        # Cross-modal consistency assessment
        if norm_img is not None and norm_txt is not None:
            concat_feat = torch.cat([norm_img, norm_txt], dim=1)
            consistency_dims = self.cross_consistency(concat_feat)

            # Apply penalties for generated features
            if is_image_missing.any() or is_text_missing.any():
                # More severe penalty if both are generated
                both_generated = is_image_missing & is_text_missing
                one_generated = (is_image_missing | is_text_missing) & ~both_generated

                # Create confidence mask
                conf_mask = torch.ones(batch_size, 1, device=device)
                conf_mask[one_generated] = 0.7  # One generated
                conf_mask[both_generated] = 0.5  # Both generated

                consistency_dims = consistency_dims * conf_mask

            # Calculate final consistency score
            consistency_weights = F.softmax(self.consistency_weights, dim=0)
            consistency = torch.sigmoid(torch.sum(consistency_dims * consistency_weights, dim=1, keepdim=True))

            results['cross_consistency'] = consistency
        else:
            results['cross_consistency'] = torch.full((batch_size, 1), 0.5, device=device)

        # Adjust modality scores based on consistency
        if self.training:
            # During training, gradually increase the importance of consistency
            w = torch.sigmoid(self.fusion_weight)

            if norm_img is not None and norm_txt is not None:
                # Adjust scores based on consistency
                results['image']['final_score'] = (1 - w) * results['image']['final_score'] + w * results[
                    'cross_consistency']
                results['text']['final_score'] = (1 - w) * results['text']['final_score'] + w * results[
                    'cross_consistency']

        # print("\nmisstype:",missing_type,"\nfinal_score:",results['image']['final_score'],results['text']['final_score']
        #       ,"\ncross_consistency",results['cross_consistency'])

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

    # def forward(self, image_feat, text_feat, quality_scores=None):
    #     """
    #     基于质量分数的前向传播函数
    #
    #     Args:
    #         image_feat: 图像特征 [B, D_img]
    #         text_feat: 文本特征 [B, D_txt]
    #         quality_scores: 质量评估模块的输出
    #
    #     Returns:
    #         融合特征和权重
    #     """
    #
    #     if self.training or (random.random() < 0.01):  # 训练模式或随机采样1%
    #         self._log_fusion_inputs(image_feat, text_feat, quality_scores)
    #
    #
    #     batch_size = image_feat.size(0)
    #     device = image_feat.device
    #
    #     # 投影特征到融合空间
    #     img_proj = self.image_proj(image_feat)  # [B, fusion_dim]
    #     txt_proj = self.text_proj(text_feat)  # [B, fusion_dim]
    #
    #     # 准备质量向量用于注意力调制
    #     if quality_scores is None or 'image' not in quality_scores or 'text' not in quality_scores:
    #         # 如果没有质量分数，使用均匀权重
    #         quality_vector = torch.ones(batch_size, 2, device=device) * 0.5
    #     else:
    #         # 使用质量分数
    #         quality_vector = torch.cat([
    #             quality_scores['image']['final_score'],
    #             quality_scores['text']['final_score']
    #         ], dim=1)  # [B, 2]
    #
    #     # 生成注意力权重
    #     attn_weights = self.quality_attn_weights(quality_vector)  # [B, num_heads]
    #
    #     # 堆叠特征作为序列 [img, txt]
    #     features = torch.stack([img_proj, txt_proj], dim=1)  # [B, 2, fusion_dim]
    #
    #     # 应用质量感知的跨模态注意力
    #     # 我们现在不需要修改原生的attention机制，而是在后处理中应用质量权重
    #     attn_output, _ = self.cross_attn(
    #         query=features,
    #         key=features,
    #         value=features
    #     )  # [B, 2, fusion_dim]
    #
    #     # 提取各模态的表示
    #     img_attn = attn_output[:, 0]  # [B, fusion_dim]
    #     txt_attn = attn_output[:, 1]  # [B, fusion_dim]
    #
    #     # 计算质量门控值
    #     quality_gate = self.quality_gate(quality_vector)  # [B, 1]
    #
    #     # 基于质量的加权平均
    #     weighted_img = img_proj * quality_scores['image']['final_score']
    #     weighted_txt = txt_proj * quality_scores['text']['final_score']
    #
    #     # 计算质量加权的特征
    #     quality_weighted = (weighted_img + weighted_txt) / (
    #             quality_scores['image']['final_score'] + quality_scores['text']['final_score'] + 1e-8
    #     )
    #
    #     # 注意力输出和质量加权之间的插值
    #     attn_weighted = (img_attn + txt_attn) / 2
    #     fused_features = quality_gate * quality_weighted + (1 - quality_gate) * attn_weighted
    #
    #     # 应用最终投影
    #     output_features = self.output_proj(torch.cat([fused_features, attn_weighted], dim=1))
    #
    #     # 为可视化目的返回简单权重
    #     simple_weights = torch.cat([
    #         quality_scores['image']['final_score'],
    #         quality_scores['text']['final_score']
    #     ], dim=1)
    #
    #     # 记录输出融合权重
    #     if simple_weights is not None and (self.training or (random.random() < 0.01)):
    #         self._log_fusion_weights(simple_weights)
    #
    #     return output_features, simple_weights

    # 在quality_aware_prompting.py文件中的QualityAwareFeatureFusion类中添加
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

        if log_details:
            print(f"Attention output: img_attn={img_attn.mean().item():.6f}, txt_attn={txt_attn.mean().item():.6f}")

        # 计算质量门控值
        quality_gate = self.quality_gate(quality_vector)

        if log_details:
            print(f"Quality gate: {quality_gate.mean().item():.6f}")

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