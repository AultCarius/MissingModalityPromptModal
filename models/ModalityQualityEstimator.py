import torch
import torch.nn as nn
import torch.nn.functional as F


class ModalityQualityEstimator(nn.Module):
    """模态质量评估器

    评估各模态的质量、完整性和一致性
    """

    def __init__(self, image_dim, text_dim, hidden_dim=256):
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim

        # 图像质量评估
        self.image_quality_estimator = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 文本质量评估
        self.text_quality_estimator = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # 跨模态一致性评估
        self.image_projection = nn.Linear(image_dim, hidden_dim)
        self.text_projection = nn.Linear(text_dim, hidden_dim)
        self.consistency_estimator = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image_features, text_features):
        """
        评估模态质量和一致性

        Args:
            image_features (torch.Tensor): 图像特征 [B, D_img]
            text_features (torch.Tensor): 文本特征 [B, D_txt]

        Returns:
            tuple: (image_quality, text_quality, consistency)
                  每个元素都是 [B, 1] 的张量，值范围为 [0, 1]
        """
        batch_size = image_features.shape[0]

        # 单模态质量评估
        image_quality = self.image_quality_estimator(image_features)
        text_quality = self.text_quality_estimator(text_features)

        # 跨模态一致性评估
        image_proj = self.image_projection(image_features)
        text_proj = self.text_projection(text_features)

        # 计算两个模态的余弦相似度作为一致性指标的一部分
        normalized_img = F.normalize(image_proj, p=2, dim=1)
        normalized_txt = F.normalize(text_proj, p=2, dim=1)
        cosine_similarity = torch.sum(normalized_img * normalized_txt, dim=1, keepdim=True)

        # 拼接投影特征并评估一致性
        concat_features = torch.cat([image_proj, text_proj], dim=1)
        consistency = self.consistency_estimator(concat_features)

        # 结合余弦相似度和学习到的一致性
        final_consistency = (consistency + cosine_similarity.clamp(0, 1)) / 2

        # 综合质量 = 单模态质量 * 一致性
        image_final_quality = image_quality * final_consistency
        text_final_quality = text_quality * final_consistency

        return image_final_quality, text_final_quality, final_consistency


class DetailedQualityEstimator(nn.Module):
    """详细的模态质量评估器

    提供更详细的质量指标：清晰度、相关性、信息量、冗余度等
    """

    def __init__(self, image_dim, text_dim, hidden_dim=256, output_dim=4):
        """
        Args:
            image_dim (int): 图像特征维度
            text_dim (int): 文本特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出质量向量的维度 (默认4: 清晰度,相关性,信息量,冗余度)
        """
        super().__init__()
        self.image_dim = image_dim
        self.text_dim = text_dim
        self.output_dim = output_dim

        # 图像质量向量评估
        self.image_quality_estimator = nn.Sequential(
            nn.LayerNorm(image_dim),
            nn.Linear(image_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

        # 文本质量向量评估
        self.text_quality_estimator = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )

        # 跨模态一致性评估
        self.consistency_estimator = nn.Sequential(
            nn.Linear(image_dim + text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, image_features, text_features):
        """
        评估详细的模态质量

        Args:
            image_features (torch.Tensor): 图像特征 [B, D_img]
            text_features (torch.Tensor): 文本特征 [B, D_txt]

        Returns:
            tuple: (image_quality_vector, text_quality_vector, consistency)
                  image_quality_vector: [B, output_dim]，详细的图像质量向量
                  text_quality_vector: [B, output_dim]，详细的文本质量向量
                  consistency: [B, 1]，模态一致性
        """
        # 计算详细质量向量
        image_quality_vector = self.image_quality_estimator(image_features)
        text_quality_vector = self.text_quality_estimator(text_features)

        # 计算一致性
        concat_features = torch.cat([image_features, text_features], dim=1)
        consistency = self.consistency_estimator(concat_features)

        # 加权详细质量向量
        image_quality_weighted = image_quality_vector * consistency
        text_quality_weighted = text_quality_vector * consistency

        # 计算总体质量分数（可选）
        image_quality_score = torch.mean(image_quality_weighted, dim=1, keepdim=True)
        text_quality_score = torch.mean(text_quality_weighted, dim=1, keepdim=True)

        return {
            'image_quality_vector': image_quality_vector,
            'text_quality_vector': text_quality_vector,
            'consistency': consistency,
            'image_quality_score': image_quality_score,
            'text_quality_score': text_quality_score
        }