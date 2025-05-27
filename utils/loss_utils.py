import torch
import torch.nn.functional as F
# 同时使用MSE和分布匹配损失
def combined_reconstruction_loss(original, reconstructed, distribution_weight=0.5):
    """
    结合点对点重建损失和分布匹配损失

    Args:
        original: 原始特征 [batch_size, feature_dim]
        reconstructed: 重建特征 [batch_size, feature_dim]
        distribution_weight: 分布损失的权重系数
    """
    # 1. 传统的MSE重建损失
    mse_loss = F.mse_loss(reconstructed, original)

    # 2. 分布匹配损失
    # 匹配均值
    mean_orig = original.mean(dim=0)
    mean_recon = reconstructed.mean(dim=0)
    mean_loss = F.mse_loss(mean_recon, mean_orig)

    # 匹配方差
    var_orig = torch.var(original, dim=0,unbiased=False)
    var_recon = torch.var(reconstructed, dim=0,unbiased=False)
    var_loss = F.mse_loss(var_recon, var_orig)

    # 总分布损失
    distribution_loss = mean_loss + var_loss

    # 组合损失
    total_loss = mse_loss + distribution_weight * distribution_loss

    return total_loss, {
        'mse_loss': mse_loss.item(),
        'mean_loss': mean_loss.item(),
        'var_loss': var_loss.item(),
        'distribution_loss': distribution_loss.item()
    }