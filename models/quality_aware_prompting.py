# Quality Aware Prompting
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

class EnhancedModalityQualityEstimator(nn.Module):
    def __init__(self, image_dim, text_dim):
        super().__init__()
        self.image_quality = nn.Sequential(
            nn.Linear(image_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 输出3个质量维度
        )

        self.text_quality = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 3)  # 输出3个质量维度
        )

        self.cross_consistency = nn.Sequential(
            nn.Linear(image_dim + text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, image_feat, text_feat, missing_type=None):
        """评估模态质量和跨模态一致性

        Args:
            image_feat: 图像特征
            text_feat: 文本特征
            missing_type: 缺失类型张量 (none=0, image=1, text=2, both=3)
        """
        batch_size = max(image_feat.size(0) if image_feat is not None else 0,
                         text_feat.size(0) if text_feat is not None else 0)
        device = image_feat.device if image_feat is not None else text_feat.device

        # 初始化结果
        results = {
            'image': {'quality': None, 'final_score': None},
            'text': {'quality': None, 'final_score': None},
            'cross_consistency': None
        }

        # 检测零填充的张量
        is_zero_image = torch.zeros(batch_size, dtype=torch.bool, device=device)
        is_zero_text = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if image_feat is not None:
            is_zero_image = torch.all(torch.abs(image_feat) < 1e-6, dim=1)
        else:
            is_zero_image = torch.ones(batch_size, dtype=torch.bool, device=device)

        if text_feat is not None:
            is_zero_text = torch.all(torch.abs(text_feat) < 1e-6, dim=1)
        else:
            is_zero_text = torch.ones(batch_size, dtype=torch.bool, device=device)

        # 计算图像质量评分
        if not is_zero_image.all():
            # 只对非零图像计算质量
            valid_image_feat = image_feat[~is_zero_image]
            valid_image_quality = self.image_quality(valid_image_feat)
            valid_image_final = torch.sigmoid(valid_image_quality.mean(dim=1, keepdim=True))

            # 将结果分配回原始批次
            image_quality = torch.zeros(batch_size, 3, device=device)
            image_final = torch.zeros(batch_size, 1, device=device)

            image_quality[~is_zero_image] = valid_image_quality
            image_final[~is_zero_image] = valid_image_final

            # 对于零图像，分配低质量分数
            image_quality[is_zero_image] = torch.tensor([-2.0, -2.0, -2.0], device=device)
            image_final[is_zero_image] = torch.tensor([0.1], device=device)  # 低但不为零

            results['image']['quality'] = image_quality
            results['image']['final_score'] = image_final
        else:
            # 所有图像都缺失
            results['image']['quality'] = torch.full((batch_size, 3), -2.0, device=device)
            results['image']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # 计算文本质量评分（类似逻辑）
        if not is_zero_text.all():
            valid_text_feat = text_feat[~is_zero_text]
            valid_text_quality = self.text_quality(valid_text_feat)
            valid_text_final = torch.sigmoid(valid_text_quality.mean(dim=1, keepdim=True))

            text_quality = torch.zeros(batch_size, 3, device=device)
            text_final = torch.zeros(batch_size, 1, device=device)

            text_quality[~is_zero_text] = valid_text_quality
            text_final[~is_zero_text] = valid_text_final

            text_quality[is_zero_text] = torch.tensor([-2.0, -2.0, -2.0], device=device)
            text_final[is_zero_text] = torch.tensor([0.1], device=device)

            results['text']['quality'] = text_quality
            results['text']['final_score'] = text_final
        else:
            results['text']['quality'] = torch.full((batch_size, 3), -2.0, device=device)
            results['text']['final_score'] = torch.full((batch_size, 1), 0.1, device=device)

        # 计算跨模态一致性
        cross_consistency = torch.zeros(batch_size, 1, device=device)

        # 只对既有图像又有文本的样本计算一致性
        both_valid = ~(is_zero_image | is_zero_text)
        if both_valid.any():
            valid_image = image_feat[both_valid]
            valid_text = text_feat[both_valid]

            # 计算两个模态特征的一致性
            concat_feat = torch.cat([valid_image, valid_text], dim=1)
            valid_consistency = torch.sigmoid(self.cross_consistency(concat_feat))

            # 重新分配回原始批次
            cross_consistency[both_valid] = valid_consistency

        # 对于缺失模态的样本，分配默认一致性分数
        cross_consistency[is_zero_image | is_zero_text] = 0.5  # 中等一致性分数

        results['cross_consistency'] = cross_consistency

        return results


class QualityGuidedFeatureFusion(nn.Module):
    """
    质量引导的特征融合模块 - 根据质量评分调整各模态的权重
    """

    def __init__(self, image_dim, text_dim, fusion_dim):
        super().__init__()

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, fusion_dim)
        )

        # 注意力网络 - 基于质量评分生成注意力系数
        self.attention_net = nn.Sequential(
            nn.Linear(2, 64),  # 输入两个模态的质量分数
            nn.ReLU(),
            nn.Linear(64, 2),  # 输出两个模态的权重
            nn.Softmax(dim=1)  # 确保权重归一化
        )

    def forward(self, image_feat, text_feat, quality_scores):
        """
        前向传播

        Args:
            image_feat: 图像特征 [B, D_img]
            text_feat: 文本特征 [B, D_txt]
            quality_scores: 字典，包含图像和文本的质量分数 {'image': [B, 1], 'text': [B, 1]}

        Returns:
            融合后的特征 [B, fusion_dim]
        """
        batch_size = image_feat.size(0)

        # 收集质量分数
        quality_tensor = torch.cat([
            quality_scores['image']['final_score'],
            quality_scores['text']['final_score']
        ], dim=1)  # [B, 2]

        # 生成注意力权重
        attention_weights = self.attention_net(quality_tensor)  # [B, 2]

        # 应用注意力权重
        weighted_image = image_feat * attention_weights[:, 0:1]
        weighted_text = text_feat * attention_weights[:, 1:2]

        # 连接加权特征
        concat_features = torch.cat([weighted_image, weighted_text], dim=1)

        # 融合
        fused_features = self.fusion(concat_features)

        return fused_features, attention_weights