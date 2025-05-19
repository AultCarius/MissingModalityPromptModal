import torch
import yaml
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from trainer import Trainer
from datamodules.MmimdbDataModule import mmimdbDataModule
from models.multimodal_model import create_multimodal_prompt_model

# GENRE_CLASS 用于MMIMDB数据集的类别名称
GENRE_CLASS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure',
    'Horror', 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family',
    'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western',
    'Sport', 'Short', 'Film-Noir'
]


def load_model_and_data(config_path, checkpoint_path):
    """加载模型和数据模块"""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 设置数据模块
    print("Setting up data module...")
    datamodule = mmimdbDataModule(
        data_dir=config.get("data_dir", "./data/mmimdb"),
        batch_size=16,  # 使用小批量方便处理
        num_workers=4,
        missing_strategy=config.get("missing_strategy", "none"),
        missing_prob=config.get("missing_prob", 0.7),
        val_missing_strategy="both",  # 确保验证集有各种缺失类型
        val_missing_prob=0.7,
        test_missing_strategy="both",
        test_missing_prob=0.7,
        max_length=config.get("max_length", 77),
        image_size=config.get("image_size", 224),
        patch_size=config.get("patch_size", 16),
        seed=config.get("seed", 42)
    )
    datamodule.setup()

    # 创建模型
    print("Creating model...")
    model = create_multimodal_prompt_model(
        image_model_name=config.get("image_model_name", "vit_base_patch16_224"),
        text_model_name=config.get("text_model_name", "roberta-base"),
        image_prompt_len=config.get("image_prompt_len", 5),
        text_prompt_len=config.get("text_prompt_len", 5),
        prompt_depth=config.get("prompt_depth", 6),
        fusion_dim=config.get("fusion_dim", 512),
        num_classes=datamodule.get_num_classes(),
        freeze_image_encoder=config.get("freeze_image_encoder", False),
        freeze_text_encoder=config.get("freeze_text_encoder", False),
        use_quality_prompt=config.get("use_quality_prompt", False),
        use_cross_modal_prompt=config.get("use_cross_modal_prompt", False),
        max_length=config.get("max_length", 512),
        encoder_type=config.get("encoder_type", "clip")
    )

    # 创建Trainer
    print("Setting up trainer...")
    trainer = Trainer(model, datamodule.train_dataloader(), datamodule.val_dataloader(), config=config)

    # 加载检查点
    print(f"Loading checkpoint from {checkpoint_path}")
    device = trainer.device
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    return model, datamodule, trainer


def test_classifier_threshold(model, datamodule, trainer):
    """测试分类器对不同阈值的反应"""
    print("\n=== Testing classifier thresholds ===")
    device = trainer.device
    test_loader = datamodule.test_dataloader()

    # 收集所有图像缺失样本的logits和标签
    all_logits = []
    all_labels = []
    print("Collecting image-missing samples...")

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader):
            image, input_ids, attention_mask, label, missing_type = [x.to(device) for x in batch]
            image_missing = (missing_type == 1)
            if not image_missing.any():
                continue

            # 只处理图像缺失样本
            img_miss_image = image[image_missing]
            img_miss_input_ids = input_ids[image_missing]
            img_miss_attention_mask = attention_mask[image_missing]
            img_miss_label = label[image_missing]
            img_miss_missing_type = missing_type[image_missing]

            # 前向传播
            output = model(img_miss_image, img_miss_input_ids, img_miss_attention_mask, img_miss_missing_type)

            if isinstance(output, tuple):
                logits, _ = output
            else:
                logits = output

            all_logits.append(logits.cpu())
            all_labels.append(img_miss_label.cpu())

    # 合并所有logits和标签
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    print(f"Collected {len(all_logits)} image-missing samples")

    # 获取每个类别的平均logits
    class_avg_logits = all_logits.mean(dim=0)
    class_std_logits = all_logits.std(dim=0)

    # 找出平均logits最高的前5个类别
    top_classes = torch.topk(class_avg_logits, k=5)
    print("\nTop 5 classes by average logits:")
    for i, cls_idx in enumerate(top_classes.indices):
        print(
            f"  {GENRE_CLASS[cls_idx]}: {class_avg_logits[cls_idx].item():.4f} ± {class_std_logits[cls_idx].item():.4f}")

    # 找出平均logits最低的前5个类别
    bottom_classes = torch.topk(class_avg_logits, k=5, largest=False)
    print("\nBottom 5 classes by average logits:")
    for i, cls_idx in enumerate(bottom_classes.indices):
        print(
            f"  {GENRE_CLASS[cls_idx]}: {class_avg_logits[cls_idx].item():.4f} ± {class_std_logits[cls_idx].item():.4f}")

    # 测试不同阈值
    thresholds = [-1.0, -0.8, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    print("\nTesting different thresholds:")

    # 为结果创建表格标题
    print(f"{'Threshold':^10} | {'Zero pred%':^10} | {'Avg pred/sample':^15} | {'Accuracy':^10} | {'Macro F1':^10}")
    print("-" * 65)

    threshold_results = []

    for threshold in thresholds:
        preds = (all_logits > threshold).float()
        pred_counts = preds.sum(dim=1)

        # 计算每个阈值下的性能指标
        zero_pred_samples = (pred_counts == 0).sum().item()
        zero_pred_percent = zero_pred_samples / preds.size(0) * 100
        avg_preds_per_sample = pred_counts.mean().item()

        # 计算准确率
        correct = ((preds > 0.5) == (all_labels > 0.5)).float()
        accuracy = correct.mean().item()

        # 计算每个类别的F1分数
        f1_scores = []
        for i in range(preds.size(1)):
            true_pos = ((preds[:, i] > 0.5) & (all_labels[:, i] > 0.5)).sum().item()
            false_pos = ((preds[:, i] > 0.5) & (all_labels[:, i] <= 0.5)).sum().item()
            false_neg = ((preds[:, i] <= 0.5) & (all_labels[:, i] > 0.5)).sum().item()

            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)

        macro_f1 = np.mean(f1_scores)

        # 打印当前阈值的结果
        print(
            f"{threshold:^10.2f} | {zero_pred_percent:^10.2f} | {avg_preds_per_sample:^15.2f} | {accuracy:^10.4f} | {macro_f1:^10.4f}")

        threshold_results.append({
            'threshold': threshold,
            'zero_pred_percent': zero_pred_percent,
            'avg_preds_per_sample': avg_preds_per_sample,
            'accuracy': accuracy,
            'macro_f1': macro_f1
        })

    # 保存可视化结果
    os.makedirs('debug_plots', exist_ok=True)

    # 绘制不同阈值下的指标变化
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot([r['threshold'] for r in threshold_results], [r['zero_pred_percent'] for r in threshold_results], 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Zero Prediction %')
    plt.title('% of Samples with Zero Predictions')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot([r['threshold'] for r in threshold_results], [r['avg_preds_per_sample'] for r in threshold_results], 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Avg Predictions/Sample')
    plt.title('Average Predictions per Sample')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot([r['threshold'] for r in threshold_results], [r['accuracy'] for r in threshold_results], 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Threshold')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot([r['threshold'] for r in threshold_results], [r['macro_f1'] for r in threshold_results], 'o-')
    plt.xlabel('Threshold')
    plt.ylabel('Macro F1')
    plt.title('Macro F1 vs Threshold')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('debug_plots/threshold_analysis.png')
    plt.close()

    # 选择最佳阈值（基于Macro F1）
    best_idx = np.argmax([r['macro_f1'] for r in threshold_results])
    best_threshold = threshold_results[best_idx]['threshold']
    print(f"\nBest threshold based on Macro F1: {best_threshold} "
          f"(Macro F1: {threshold_results[best_idx]['macro_f1']:.4f})")


def check_classifier_bias(model):
    """检查分类器权重和偏置"""
    print("\n=== Checking Classifier Bias ===")

    # 获取分类器
    classifier = model.classifier

    # 分析权重
    weight = classifier.weight.data

    print(f"Classifier weight shape: {weight.shape}")
    print(f"Weight statistics:")
    print(f"  Mean: {weight.mean().item():.4f}")
    print(f"  Std: {weight.std().item():.4f}")
    print(f"  Min: {weight.min().item():.4f}")
    print(f"  Max: {weight.max().item():.4f}")
    print(f"  Positive%: {(weight > 0).float().mean().item() * 100:.2f}%")

    # 检查每个类别的权重统计
    print("\nClass weight statistics:")
    print(f"{'Class':^20} | {'Mean':^8} | {'Std':^8} | {'Min':^8} | {'Max':^8} | {'Pos%':^8}")
    print("-" * 70)

    for i in range(min(23, weight.size(0))):  # 限制为前23个类别（MMIMDB的类别数）
        cls_weight = weight[i]
        mean = cls_weight.mean().item()
        std = cls_weight.std().item()
        min_val = cls_weight.min().item()
        max_val = cls_weight.max().item()
        pos_percent = (cls_weight > 0).float().mean().item() * 100

        print(
            f"{GENRE_CLASS[i]:^20} | {mean:^8.4f} | {std:^8.4f} | {min_val:^8.4f} | {max_val:^8.4f} | {pos_percent:^8.2f}")

    # 检查偏置（如果存在）
    if hasattr(classifier, 'bias') and classifier.bias is not None:
        bias = classifier.bias.data
        print("\nClassifier bias statistics:")
        print(f"  Mean: {bias.mean().item():.4f}")
        print(f"  Std: {bias.std().item():.4f}")
        print(f"  Min: {bias.min().item():.4f}")
        print(f"  Max: {bias.max().item():.4f}")
        print(f"  Positive%: {(bias > 0).float().mean().item() * 100:.2f}%")

        # 显示每个类别的偏置值
        print("\nClass bias values:")
        print(f"{'Class':^20} | {'Bias':^8}")
        print("-" * 30)

        for i in range(min(23, bias.size(0))):
            print(f"{GENRE_CLASS[i]:^20} | {bias[i].item():^8.4f}")

        # 特别关注那些可能在小批量测试中产生正向预测的类别
        positive_classes = [0, 5, 6, 14, 19]  # 从之前评估日志观察到的
        print("\nBias for potentially positive classes:")
        for cls in positive_classes:
            if cls < bias.size(0):
                print(f"  Class {cls} ({GENRE_CLASS[cls]}): {bias[cls].item():.4f}")

    # 计算权重和偏置的总体影响
    if hasattr(classifier, 'bias') and classifier.bias is not None:
        # 假设输入特征均值为之前观察到的-0.015671
        avg_input = -0.015671
        # 计算每个类别的预激活值 (W*x + b)
        pre_activations = avg_input * weight.mean(dim=1) + bias

        print("\nEstimated pre-activations with average input:")
        print(f"{'Class':^20} | {'Pre-activation':^15}")
        print("-" * 40)

        for i in range(min(23, weight.size(0))):
            print(f"{GENRE_CLASS[i]:^20} | {pre_activations[i].item():^15.4f}")

        # 按预激活值排序的前5和后5个类别
        top_classes = torch.topk(pre_activations, k=5)
        bottom_classes = torch.topk(pre_activations, k=5, largest=False)

        print("\nTop 5 classes by pre-activation:")
        for i, cls_idx in enumerate(top_classes.indices):
            print(f"  {i + 1}. {GENRE_CLASS[cls_idx]}: {pre_activations[cls_idx].item():.4f}")

        print("\nBottom 5 classes by pre-activation:")
        for i, cls_idx in enumerate(bottom_classes.indices):
            print(f"  {i + 1}. {GENRE_CLASS[cls_idx]}: {pre_activations[cls_idx].item():.4f}")

    # 可视化分类器权重
    try:
        plt.figure(figsize=(12, 8))
        plt.imshow(weight.cpu().numpy(), cmap='coolwarm', aspect='auto')
        plt.colorbar(label='Weight Value')
        plt.xlabel('Input Feature Dimension')
        plt.ylabel('Class')
        plt.title('Classifier Weight Matrix')
        plt.savefig('debug_plots/classifier_weights.png')
        plt.close()
        print("\nClassifier weight matrix visualization saved to debug_plots/classifier_weights.png")
    except Exception as e:
        print(f"Error creating classifier weight visualization: {str(e)}")


def test_batch_dependency(model, datamodule, trainer):
    """测试模型对批量大小的依赖性"""
    print("\n=== Testing Batch Size Dependency ===")
    device = trainer.device
    test_loader = datamodule.test_dataloader()

    # 收集所有图像缺失样本
    all_samples = []
    print("Collecting image-missing samples...")

    for batch in tqdm(test_loader):
        image, input_ids, attention_mask, label, missing_type = batch
        image_missing = (missing_type == 1)
        if not image_missing.any():
            continue

        # 只收集图像缺失样本
        for i in range(image_missing.size(0)):
            if image_missing[i]:
                all_samples.append((
                    image[i:i + 1],
                    input_ids[i:i + 1],
                    attention_mask[i:i + 1],
                    label[i:i + 1],
                    missing_type[i:i + 1]
                ))

    print(f"Collected {len(all_samples)} image-missing samples")

    # 测试不同批量大小
    batch_sizes = [1, 4, 8, 16, 32, 64]

    print("\nTesting different batch sizes:")
    print(f"{'Batch Size':^12} | {'Positive%':^12} | {'Samples w/ pred%':^20} | {'Avg pred/sample':^18}")
    print("-" * 68)

    batch_results = []

    for bs in batch_sizes:
        # 限制总样本数，以保持测试高效
        max_samples = min(len(all_samples), 256)
        test_samples = all_samples[:max_samples]

        # 构建具有不同批量大小的批次
        batches = []
        for i in range(0, max_samples, bs):
            end = min(i + bs, max_samples)
            batch_samples = test_samples[i:end]

            # 合并批次
            batch_image = torch.cat([s[0] for s in batch_samples])
            batch_input_ids = torch.cat([s[1] for s in batch_samples])
            batch_attention_mask = torch.cat([s[2] for s in batch_samples])
            batch_label = torch.cat([s[3] for s in batch_samples])
            batch_missing_type = torch.cat([s[4] for s in batch_samples])

            batches.append((batch_image, batch_input_ids, batch_attention_mask, batch_label, batch_missing_type))

        # 测试每个批次
        all_batch_logits = []
        all_batch_preds = []

        model.eval()
        with torch.no_grad():
            for batch in batches:
                image, input_ids, attention_mask, label, missing_type = [x.to(device) for x in batch]

                # 前向传播
                output = model(image, input_ids, attention_mask, missing_type)

                if isinstance(output, tuple):
                    logits, _ = output
                else:
                    logits = output

                # 保存logits
                all_batch_logits.append(logits.cpu())

                # 计算预测 (使用阈值0，标准阈值)
                preds = (logits > 0).float()
                all_batch_preds.append(preds.cpu())

        # 合并所有批次结果
        all_batch_logits = torch.cat(all_batch_logits, dim=0)
        all_batch_preds = torch.cat(all_batch_preds, dim=0)

        # 计算指标
        positive_percent = all_batch_preds.mean().item() * 100
        samples_with_preds = (all_batch_preds.sum(dim=1) > 0).float().mean().item() * 100
        avg_preds_per_sample = all_batch_preds.sum(dim=1).mean().item()

        print(f"{bs:^12d} | {positive_percent:^12.2f} | {samples_with_preds:^20.2f} | {avg_preds_per_sample:^18.2f}")

        batch_results.append({
            'batch_size': bs,
            'positive_percent': positive_percent,
            'samples_with_preds': samples_with_preds,
            'avg_preds_per_sample': avg_preds_per_sample,
            'logits_mean': all_batch_logits.mean().item(),
            'logits_std': all_batch_logits.std().item()
        })

    # 可视化批量大小对预测的影响
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot([r['batch_size'] for r in batch_results], [r['positive_percent'] for r in batch_results], 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Positive Predictions %')
    plt.title('Percentage of Positive Predictions vs Batch Size')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot([r['batch_size'] for r in batch_results], [r['samples_with_preds'] for r in batch_results], 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Samples with Predictions %')
    plt.title('Percentage of Samples with Any Prediction vs Batch Size')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot([r['batch_size'] for r in batch_results], [r['avg_preds_per_sample'] for r in batch_results], 'o-')
    plt.xlabel('Batch Size')
    plt.ylabel('Avg Predictions/Sample')
    plt.title('Average Predictions per Sample vs Batch Size')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot([r['batch_size'] for r in batch_results], [r['logits_mean'] for r in batch_results], 'o-', label='Mean')
    plt.plot([r['batch_size'] for r in batch_results], [r['logits_std'] for r in batch_results], 'o-', label='Std')
    plt.xlabel('Batch Size')
    plt.ylabel('Logits Statistics')
    plt.title('Logits Mean and Std vs Batch Size')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('debug_plots/batch_dependency.png')
    plt.close()
    print("\nBatch dependency visualization saved to debug_plots/batch_dependency.png")


if __name__ == "__main__":
    # 修改路径以匹配你的设置
    config_path = "configs/mmimdb.yaml"  # 你的配置文件路径
    checkpoint_path = "E:\\DL\\exp\\self\\mmimdb_both_0.7_clip\\checkpoints\\best_model.pt"  # 你的检查点路径

    # 创建输出目录
    os.makedirs('debug_plots', exist_ok=True)

    # 加载模型和数据
    model, datamodule, trainer = load_model_and_data(config_path, checkpoint_path)

    # 运行所有测试
    test_classifier_threshold(model, datamodule, trainer)
    check_classifier_bias(model)
    test_batch_dependency(model, datamodule, trainer)

    print("\nAll tests completed. Check the debug_plots directory for visualizations.")