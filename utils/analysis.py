import torch

def analyze_feature_stats(features: torch.Tensor, name: str = "Feature"):
    """
    打印特征张量的统计信息。

    Args:
        features (torch.Tensor): 特征张量，形状为 [B, D] 或 [B, T, D]
        name (str): 特征名称，用于打印标识
    """
    if features.dim() == 3:
        B, T, D = features.shape
        features_flat = features.view(B * T, D)
    elif features.dim() == 2:
        B, D = features.shape
        features_flat = features
    else:
        raise ValueError(f"Unsupported feature shape: {features.shape}")

    # 计算统计值
    mean = features_flat.mean().item()
    std = features_flat.std().item()
    abs_mean = features_flat.abs().mean().item()
    min_val = features_flat.min().item()
    max_val = features_flat.max().item()
    l2_norm = torch.norm(features_flat, p=2).item()

    print(f"===== Feature Analysis: {name} =====")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
    print(f"  Min: {min_val:.6f}, Max: {max_val:.6f}")
    print(f"  Abs Mean: {abs_mean:.6f}, L2 Norm: {l2_norm:.6f}")
    print("====================================\n")


def compare_features(
    feat_a: torch.Tensor,
    feat_b: torch.Tensor,
    names=("Feature A", "Feature B"),
    verbose=True
):
    """
    比较两个特征张量的分布特性。

    Args:
        feat_a (torch.Tensor): 特征A，形状为 [B, D] 或 [B, T, D]
        feat_b (torch.Tensor): 特征B，形状为 [B, D] 或 [B, T, D]
        names (tuple): 特征名称 (name_a, name_b)
        verbose (bool): 是否打印详细统计信息
    """

    def flatten(x):
        return x.view(-1, x.shape[-1]) if x.dim() == 3 else x

    def get_stats(x):
        return {
            'shape': x.shape,
            'mean': x.mean().item(),
            'std': x.std().item(),
            'abs_mean': x.abs().mean().item(),
            'min': x.min().item(),
            'max': x.max().item(),
            'l2': torch.norm(x, p=2).item(),
            'range': (x.max() - x.min()).item()
        }

    a_flat, b_flat = flatten(feat_a), flatten(feat_b)
    stats_a, stats_b = get_stats(a_flat), get_stats(b_flat)

    if verbose:
        print(f"===== Feature Analysis: {names[0]} =====")
        print(f"  Shape: {feat_a.shape}")
        print(f"  Mean: {stats_a['mean']:.6f}, Std: {stats_a['std']:.6f}")
        print(f"  Min: {stats_a['min']:.6f}, Max: {stats_a['max']:.6f}")
        print(f"  Abs Mean: {stats_a['abs_mean']:.6f}, L2 Norm: {stats_a['l2']:.6f}")
        print("========================================")

        print(f"===== Feature Analysis: {names[1]} =====")
        print(f"  Shape: {feat_b.shape}")
        print(f"  Mean: {stats_b['mean']:.6f}, Std: {stats_b['std']:.6f}")
        print(f"  Min: {stats_b['min']:.6f}, Max: {stats_b['max']:.6f}")
        print(f"  Abs Mean: {stats_b['abs_mean']:.6f}, L2 Norm: {stats_b['l2']:.6f}")
        print("========================================")

    # 比值分析（避免除以零）
    def safe_div(a, b):
        return a / b if abs(b) > 1e-8 else float('inf')

    std_ratio = safe_div(stats_a['std'], stats_b['std'])
    abs_mean_ratio = safe_div(stats_a['abs_mean'], stats_b['abs_mean'])
    l2_ratio = safe_div(stats_a['l2'], stats_b['l2'])
    range_ratio = safe_div(stats_a['range'], stats_b['range'])

    print(f"===== Comparison ({names[0]} vs {names[1]}) =====")
    print(f"  Std ratio: {std_ratio:.4f}x")
    print(f"  Abs Mean ratio: {abs_mean_ratio:.4f}x")
    print(f"  L2 Norm ratio: {l2_ratio:.4f}x")
    print(f"  Range ratio: {range_ratio:.4f}x")
    print("========================================\n")