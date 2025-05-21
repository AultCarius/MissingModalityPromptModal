import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
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





class FusionAnalyzer:
    """
    分析融合特征的工具类，用于研究base_hidden和quality_guided_feat的差异
    """

    def __init__(self, save_dir='analysis_results'):
        """
        初始化分析器

        Args:
            save_dir: 保存分析结果的目录
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.sample_count = 0

        # 存储累积统计信息
        self.base_stats = {
            'means': [],
            'stds': [],
            'norms': [],
            'abs_means': []
        }
        self.quality_stats = {
            'means': [],
            'stds': [],
            'norms': [],
            'abs_means': []
        }
        self.diff_stats = {
            'cosine_sims': [],
            'l2_dists': [],
            'missing_types': []
        }

        # 为了PCA/TSNE可视化存储一些特征向量
        self.stored_features = {
            'base': [],
            'quality': [],
            'missing_types': []
        }
        self.max_stored = 500  # 最多存储多少样本

    def set_plotdir(self,dir):
        self.save_dir = dir
        self._create_analysis_directories()

    def _create_analysis_directories(self):
        """创建分析所需的子文件夹并保存路径到类属性"""
        subdirs = {
            'fusion_analysis': 'fusion_analysis',
            'tsne_vis': 'tsne_visualization',
            'pca_vis': 'pca_visualization',
            'missing_type': 'analysis_by_missing_type',
            'feature_stats': 'feature_statistics_summary',
            'dim_dist': 'dim_distribution'
        }

        # 创建主目录（如果不存在）
        os.makedirs(self.save_dir, exist_ok=True)

        # 创建子目录并保存路径
        for attr, subdir_name in subdirs.items():
            full_path = os.path.join(self.save_dir, subdir_name)
            os.makedirs(full_path, exist_ok=True)
            setattr(self, attr, full_path)


    def analyze_fusion_features(self, base_hidden, quality_guided_feat, missing_type=None,
                                alpha=None, batch_idx=None, save_current=False):
        """
        分析融合特征的差异

        Args:
            base_hidden: 基础融合特征 [batch_size, hidden_dim]
            quality_guided_feat: 质量引导的融合特征 [batch_size, hidden_dim]
            missing_type: 缺失类型张量 [batch_size]
            alpha: 融合权重
            batch_idx: 当前批次索引（用于文件名）
            save_current: 是否保存当前批次的分析结果
        """
        if base_hidden is None or quality_guided_feat is None:
            print("Warning: One of the input features is None. Skipping analysis.")
            return

        if batch_idx is None:
            batch_idx = self.sample_count

        # 确保张量在CPU上
        base = base_hidden.detach().cpu()
        quality = quality_guided_feat.detach().cpu()

        if missing_type is not None:
            missing_type = missing_type.detach().cpu()

        batch_size, hidden_dim = base.shape
        self.sample_count += batch_size

        # 1. 计算基本统计量
        base_mean = base.mean(dim=1)
        base_std = base.std(dim=1)
        base_norm = torch.norm(base, dim=1)
        base_abs_mean = base.abs().mean(dim=1)

        quality_mean = quality.mean(dim=1)
        quality_std = quality.std(dim=1)
        quality_norm = torch.norm(quality, dim=1)
        quality_abs_mean = quality.abs().mean(dim=1)

        # 2. 计算差异度量
        # 余弦相似度
        base_normalized = F.normalize(base, p=2, dim=1)
        quality_normalized = F.normalize(quality, p=2, dim=1)
        cosine_sim = torch.sum(base_normalized * quality_normalized, dim=1)

        # L2距离
        l2_dist = torch.norm(base - quality, dim=1)

        # 3. 存储统计信息
        self.base_stats['means'].append(base_mean.numpy())
        self.base_stats['stds'].append(base_std.numpy())
        self.base_stats['norms'].append(base_norm.numpy())
        self.base_stats['abs_means'].append(base_abs_mean.numpy())

        self.quality_stats['means'].append(quality_mean.numpy())
        self.quality_stats['stds'].append(quality_std.numpy())
        self.quality_stats['norms'].append(quality_norm.numpy())
        self.quality_stats['abs_means'].append(quality_abs_mean.numpy())

        self.diff_stats['cosine_sims'].append(cosine_sim.numpy())
        self.diff_stats['l2_dists'].append(l2_dist.numpy())

        if missing_type is not None:
            self.diff_stats['missing_types'].append(missing_type.numpy())

        # 4. 存储部分样本用于降维可视化
        if len(self.stored_features['base']) < self.max_stored:
            samples_to_store = min(batch_size, self.max_stored - len(self.stored_features['base']))
            self.stored_features['base'].append(base[:samples_to_store].numpy())
            self.stored_features['quality'].append(quality[:samples_to_store].numpy())
            if missing_type is not None:
                self.stored_features['missing_types'].append(missing_type[:samples_to_store].numpy())

        # 5. 如果需要，保存当前批次的详细分析
        if save_current:
            self._analyze_and_save_current_batch(base, quality, cosine_sim, l2_dist, missing_type, alpha, batch_idx)

        # 返回一个简洁的摘要
        return {
            'mean_cosine_sim': cosine_sim.mean().item(),
            'mean_l2_dist': l2_dist.mean().item(),
            'base_stats': {
                'mean': base_mean.mean().item(),
                'std': base_std.mean().item(),
                'norm': base_norm.mean().item()
            },
            'quality_stats': {
                'mean': quality_mean.mean().item(),
                'std': quality_std.mean().item(),
                'norm': quality_norm.mean().item()
            }
        }

    def _analyze_and_save_current_batch(self, base, quality, cosine_sim, l2_dist, missing_type, alpha, batch_idx):
        """分析并保存当前批次的详细结果"""
        batch_size, hidden_dim = base.shape

        # 创建一个模式图，显示特征激活模式
        plt.figure(figsize=(15, 10))

        # 1. 创建热力图显示前几个样本的特征
        n_samples = min(8, batch_size)
        plt.subplot(2, 2, 1)
        sns.heatmap(base[:n_samples, :min(100, hidden_dim)].numpy(),
                    cmap='coolwarm', center=0, vmin=-2, vmax=2)
        plt.title('Base Hidden Features (First 100 dims)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Sample Index')

        plt.subplot(2, 2, 2)
        sns.heatmap(quality[:n_samples, :min(100, hidden_dim)].numpy(),
                    cmap='coolwarm', center=0, vmin=-2, vmax=2)
        plt.title('Quality Guided Features (First 100 dims)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Sample Index')

        # 2. 特征差异热力图
        plt.subplot(2, 2, 3)
        diff = (base - quality)[:n_samples, :min(100, hidden_dim)].numpy()
        sns.heatmap(diff, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        plt.title('Feature Difference (Base - Quality)')
        plt.xlabel('Feature Dimension')
        plt.ylabel('Sample Index')

        # 3. 按照缺失类型绘制余弦相似度分布
        plt.subplot(2, 2, 4)
        if missing_type is not None:
            missing_types = ['none', 'image', 'text', 'both']
            for i, mt_name in enumerate(missing_types):
                mt_mask = (missing_type == i)
                if mt_mask.sum() > 0:
                    sns.histplot(cosine_sim[mt_mask].numpy(),
                                 kde=True, label=f'{mt_name} (n={mt_mask.sum().item()})')
            plt.legend()
        else:
            sns.histplot(cosine_sim.numpy(), kde=True)

        plt.title(f'Cosine Similarity Distribution (α={"N/A" if alpha is None else f"{alpha:.2f}"})')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.xlim(-1, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.fusion_analysis, f'fusion_analysis_batch_{batch_idx}.png'))
        plt.close()

        # 4. 绘制单个维度的分布比较
        plt.figure(figsize=(15, 5))

        # 选取几个有代表性的维度
        selected_dims = [0, hidden_dim // 4, hidden_dim // 2, hidden_dim * 3 // 4, hidden_dim - 1]
        selected_dims = [d for d in selected_dims if d < hidden_dim]

        for i, dim in enumerate(selected_dims[:5]):  # 最多显示5个维度
            plt.subplot(1, len(selected_dims), i + 1)

            sns.histplot(base[:, dim].numpy(), color='blue', alpha=0.5,
                         label='Base', kde=True, stat='density')
            sns.histplot(quality[:, dim].numpy(), color='red', alpha=0.5,
                         label='Quality', kde=True, stat='density')

            plt.title(f'Dimension {dim}')
            plt.xlabel('Value')
            plt.ylabel('Density')
            if i == 0:
                plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.dim_dist, f'dim_distribution_batch_{batch_idx}.png'))
        plt.close()

    def generate_summary_report(self,epoch=0):
        """生成汇总报告并可视化累积结果"""
        if self.sample_count == 0:
            print("No samples have been analyzed yet.")
            return

        # 合并存储的统计信息
        base_means = np.concatenate(self.base_stats['means'])
        base_stds = np.concatenate(self.base_stats['stds'])
        base_norms = np.concatenate(self.base_stats['norms'])
        base_abs_means = np.concatenate(self.base_stats['abs_means'])

        quality_means = np.concatenate(self.quality_stats['means'])
        quality_stds = np.concatenate(self.quality_stats['stds'])
        quality_norms = np.concatenate(self.quality_stats['norms'])
        quality_abs_means = np.concatenate(self.quality_stats['abs_means'])

        cosine_sims = np.concatenate(self.diff_stats['cosine_sims'])
        l2_dists = np.concatenate(self.diff_stats['l2_dists'])

        # 1. 创建统计摘要图
        plt.figure(figsize=(15, 10))

        # 绘制均值分布比较
        plt.subplot(2, 2, 1)
        sns.histplot(base_means, color='blue', alpha=0.5, label='Base', kde=True)
        sns.histplot(quality_means, color='red', alpha=0.5, label='Quality', kde=True)
        plt.title('Distribution of Feature Means')
        plt.xlabel('Mean Value')
        plt.ylabel('Count')
        plt.legend()

        # 绘制标准差分布比较
        plt.subplot(2, 2, 2)
        sns.histplot(base_stds, color='blue', alpha=0.5, label='Base', kde=True)
        sns.histplot(quality_stds, color='red', alpha=0.5, label='Quality', kde=True)
        plt.title('Distribution of Feature Standard Deviations')
        plt.xlabel('Standard Deviation')
        plt.ylabel('Count')
        plt.legend()

        # 绘制模范数分布比较
        plt.subplot(2, 2, 3)
        sns.histplot(base_norms, color='blue', alpha=0.5, label='Base', kde=True)
        sns.histplot(quality_norms, color='red', alpha=0.5, label='Quality', kde=True)
        plt.title('Distribution of Feature L2 Norms')
        plt.xlabel('L2 Norm')
        plt.ylabel('Count')
        plt.legend()

        # 绘制绝对均值分布比较
        plt.subplot(2, 2, 4)
        sns.histplot(base_abs_means, color='blue', alpha=0.5, label='Base', kde=True)
        sns.histplot(quality_abs_means, color='red', alpha=0.5, label='Quality', kde=True)
        plt.title('Distribution of Feature Absolute Means')
        plt.xlabel('Absolute Mean')
        plt.ylabel('Count')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.feature_stats, f'feature_statistics_summary_epoch{epoch}.png'))
        plt.close()

        # 2. 创建差异分析图
        plt.figure(figsize=(12, 5))

        # 绘制余弦相似度分布
        plt.subplot(1, 2, 1)
        sns.histplot(cosine_sims, kde=True)
        plt.title('Distribution of Cosine Similarities')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.xlim(-1, 1)

        # 绘制L2距离分布
        plt.subplot(1, 2, 2)
        sns.histplot(l2_dists, kde=True)
        plt.title('Distribution of L2 Distances')
        plt.xlabel('L2 Distance')
        plt.ylabel('Count')

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'feature_difference_summary_epoch{epoch}'))
        plt.close()

        # 3. 如果有缺失类型信息，按缺失类型分析
        if self.diff_stats['missing_types']:
            missing_types = np.concatenate(self.diff_stats['missing_types'])
            self._analyze_by_missing_type(
                base_means, quality_means, base_norms, quality_norms,
                cosine_sims, l2_dists, missing_types,epoch
            )

        # 4. 降维可视化
        if self.stored_features['base']:
            self._visualize_with_dimensionality_reduction(epoch)

        # 5. 打印统计摘要
        print("\n===== Fusion Features Analysis Summary =====")
        print(f"Total samples analyzed: {self.sample_count}")
        print("\nBase Hidden Features:")
        print(f"  Mean: {np.mean(base_means):.6f} ± {np.std(base_means):.6f}")
        print(f"  Std: {np.mean(base_stds):.6f} ± {np.std(base_stds):.6f}")
        print(f"  L2 Norm: {np.mean(base_norms):.6f} ± {np.std(base_norms):.6f}")
        print(f"  Abs Mean: {np.mean(base_abs_means):.6f} ± {np.std(base_abs_means):.6f}")

        print("\nQuality Guided Features:")
        print(f"  Mean: {np.mean(quality_means):.6f} ± {np.std(quality_means):.6f}")
        print(f"  Std: {np.mean(quality_stds):.6f} ± {np.std(quality_stds):.6f}")
        print(f"  L2 Norm: {np.mean(quality_norms):.6f} ± {np.std(quality_norms):.6f}")
        print(f"  Abs Mean: {np.mean(quality_abs_means):.6f} ± {np.std(quality_abs_means):.6f}")

        print("\nFeature Differences:")
        print(f"  Average Cosine Similarity: {np.mean(cosine_sims):.6f} ± {np.std(cosine_sims):.6f}")
        print(f"  Average L2 Distance: {np.mean(l2_dists):.6f} ± {np.std(l2_dists):.6f}")
        print("==========================================")

        # 6. 保存数值结果为CSV
        np.savetxt(os.path.join(self.save_dir, f'feature_stats_epoch{epoch}.csv'),
                   np.column_stack([
                       base_means, base_stds, base_norms, base_abs_means,
                       quality_means, quality_stds, quality_norms, quality_abs_means,
                       cosine_sims, l2_dists
                   ]),
                   delimiter=',',
                   header='base_mean,base_std,base_norm,base_abs_mean,'
                          'quality_mean,quality_std,quality_norm,quality_abs_mean,'
                          'cosine_sim,l2_dist')

    def _analyze_by_missing_type(self, base_means, quality_means, base_norms, quality_norms,
                                 cosine_sims, l2_dists, missing_types,epoch):
        """按缺失类型分析特征差异"""
        missing_type_names = ['none', 'image', 'text', 'both']

        # 创建按缺失类型的图表
        plt.figure(figsize=(15, 10))

        # 1. 按缺失类型的余弦相似度
        plt.subplot(2, 2, 1)
        for i, mt_name in enumerate(missing_type_names):
            mt_mask = (missing_types == i)
            if np.sum(mt_mask) > 0:
                sns.histplot(cosine_sims[mt_mask], kde=True, label=f'{mt_name} (n={np.sum(mt_mask)})')
        plt.title('Cosine Similarity by Missing Type')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Count')
        plt.legend()
        plt.xlim(-1, 1)

        # 2. 按缺失类型的L2距离
        plt.subplot(2, 2, 2)
        for i, mt_name in enumerate(missing_type_names):
            mt_mask = (missing_types == i)
            if np.sum(mt_mask) > 0:
                sns.histplot(l2_dists[mt_mask], kde=True, label=f'{mt_name} (n={np.sum(mt_mask)})')
        plt.title('L2 Distance by Missing Type')
        plt.xlabel('L2 Distance')
        plt.ylabel('Count')
        plt.legend()

        # 3. 按缺失类型的特征均值差异
        plt.subplot(2, 2, 3)
        mean_diffs = base_means - quality_means

        for i, mt_name in enumerate(missing_type_names):
            mt_mask = (missing_types == i)
            if np.sum(mt_mask) > 0:
                sns.histplot(mean_diffs[mt_mask], kde=True, label=f'{mt_name} (n={np.sum(mt_mask)})')
        plt.title('Mean Difference (Base - Quality) by Missing Type')
        plt.xlabel('Mean Difference')
        plt.ylabel('Count')
        plt.legend()

        # 4. 按缺失类型的特征范数比率
        plt.subplot(2, 2, 4)
        norm_ratios = base_norms / (quality_norms + 1e-6)  # 避免除零

        for i, mt_name in enumerate(missing_type_names):
            mt_mask = (missing_types == i)
            if np.sum(mt_mask) > 0:
                sns.histplot(norm_ratios[mt_mask], kde=True, label=f'{mt_name} (n={np.sum(mt_mask)})')
        plt.title('Norm Ratio (Base / Quality) by Missing Type')
        plt.xlabel('Norm Ratio')
        plt.ylabel('Count')
        plt.legend()
        plt.xscale('log')  # 使用对数刻度更好地显示比率

        plt.tight_layout()
        plt.savefig(os.path.join(self.missing_type, f'analysis_by_missing_type_epoch{epoch}.png'))
        plt.close()

        # 统计每种缺失类型的关键指标均值
        print("\n===== Analysis by Missing Type =====")
        for i, mt_name in enumerate(missing_type_names):
            mt_mask = (missing_types == i)
            n_samples = np.sum(mt_mask)

            if n_samples > 0:
                print(f"\nMissing Type: {mt_name} (n={n_samples})")
                print(f"  Cosine Similarity: {np.mean(cosine_sims[mt_mask]):.6f} ± {np.std(cosine_sims[mt_mask]):.6f}")
                print(f"  L2 Distance: {np.mean(l2_dists[mt_mask]):.6f} ± {np.std(l2_dists[mt_mask]):.6f}")
                print(f"  Mean Difference: {np.mean(mean_diffs[mt_mask]):.6f} ± {np.std(mean_diffs[mt_mask]):.6f}")
                print(f"  Norm Ratio: {np.mean(norm_ratios[mt_mask]):.6f} ± {np.std(norm_ratios[mt_mask]):.6f}")

    def _visualize_with_dimensionality_reduction(self,epoch):
        """使用降维技术可视化特征空间"""
        # 合并所有存储的特征
        base_features = np.vstack(self.stored_features['base'])
        quality_features = np.vstack(self.stored_features['quality'])

        # 获取缺失类型（如果有）
        if self.stored_features['missing_types']:
            missing_types = np.concatenate(self.stored_features['missing_types'])
        else:
            missing_types = None

        # 将特征拼接起来用于降维
        all_features = np.vstack([base_features, quality_features])
        n_samples = base_features.shape[0]

        # 应用PCA降维
        try:
            # 1. PCA可视化
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(all_features)

            pca_base = pca_result[:n_samples]
            pca_quality = pca_result[n_samples:]

            plt.figure(figsize=(12, 10))
            plt.subplot(2, 1, 1)

            # 按照特征类型绘制
            plt.scatter(pca_base[:, 0], pca_base[:, 1], c='blue', label='Base', alpha=0.5)
            plt.scatter(pca_quality[:, 0], pca_quality[:, 1], c='red', label='Quality', alpha=0.5)

            # 绘制连接线，表示同一样本的两种特征
            for i in range(n_samples):
                plt.plot([pca_base[i, 0], pca_quality[i, 0]],
                         [pca_base[i, 1], pca_quality[i, 1]],
                         'gray', alpha=0.1)

            plt.title(f'PCA Visualization of Base vs Quality Features (n={n_samples})')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend()

            # 如果有缺失类型，按缺失类型绘制
            if missing_types is not None:
                plt.subplot(2, 1, 2)

                missing_type_names = ['none', 'image', 'text', 'both']
                colors = ['green', 'red', 'blue', 'purple']

                for i, (mt_name, color) in enumerate(zip(missing_type_names, colors)):
                    mt_mask = (missing_types == i)
                    if np.sum(mt_mask) > 0:
                        plt.scatter(pca_base[mt_mask, 0], pca_base[mt_mask, 1],
                                    c=color, marker='o', label=f'{mt_name} (Base)', alpha=0.7)
                        plt.scatter(pca_quality[mt_mask, 0], pca_quality[mt_mask, 1],
                                    c=color, marker='+', label=f'{mt_name} (Quality)', alpha=0.7)

                plt.title('PCA Visualization by Missing Type')
                plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
                plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
                plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.pca_vis, f'pca_visualization_epoch{epoch}.png'))
            plt.close()

            # 2. t-SNE可视化
            # 限制样本数量以加快t-SNE计算
            max_tsne_samples = min(1000, n_samples)
            if n_samples > max_tsne_samples:
                indices = np.random.choice(n_samples, max_tsne_samples, replace=False)
                tsne_base = base_features[indices]
                tsne_quality = quality_features[indices]
                tsne_missing_types = missing_types[indices] if missing_types is not None else None
            else:
                tsne_base = base_features
                tsne_quality = quality_features
                tsne_missing_types = missing_types

            tsne_all = np.vstack([tsne_base, tsne_quality])

            tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
            tsne_result = tsne.fit_transform(tsne_all)

            tsne_base_result = tsne_result[:max_tsne_samples]
            tsne_quality_result = tsne_result[max_tsne_samples:]

            plt.figure(figsize=(12, 10))
            plt.subplot(2, 1, 1)

            # 按照特征类型绘制
            plt.scatter(tsne_base_result[:, 0], tsne_base_result[:, 1],
                        c='blue', label='Base', alpha=0.5)
            plt.scatter(tsne_quality_result[:, 0], tsne_quality_result[:, 1],
                        c='red', label='Quality', alpha=0.5)

            # 绘制连接线
            for i in range(max_tsne_samples):
                plt.plot([tsne_base_result[i, 0], tsne_quality_result[i, 0]],
                         [tsne_base_result[i, 1], tsne_quality_result[i, 1]],
                         'gray', alpha=0.1)

            plt.title(f't-SNE Visualization of Base vs Quality Features (n={max_tsne_samples})')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()

            # 如果有缺失类型，按缺失类型绘制
            if tsne_missing_types is not None:
                plt.subplot(2, 1, 2)

                missing_type_names = ['none', 'image', 'text', 'both']
                colors = ['green', 'red', 'blue', 'purple']

                for i, (mt_name, color) in enumerate(zip(missing_type_names, colors)):
                    mt_mask = (tsne_missing_types == i)
                    if np.sum(mt_mask) > 0:
                        plt.scatter(tsne_base_result[mt_mask, 0], tsne_base_result[mt_mask, 1],
                                    c=color, marker='o', label=f'{mt_name} (Base)', alpha=0.7)
                        plt.scatter(tsne_quality_result[mt_mask, 0], tsne_quality_result[mt_mask, 1],
                                    c=color, marker='+', label=f'{mt_name} (Quality)', alpha=0.7)

                plt.title('t-SNE Visualization by Missing Type')
                plt.xlabel('t-SNE 1')
                plt.ylabel('t-SNE 2')
                plt.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.tsne_vis, f'tsne_visualization_epoch{epoch}.png'))
            plt.close()

        except Exception as e:
            print(f"Error during dimensionality reduction: {str(e)}")
            # 如果降维失败，则跳过这一部分
            pass