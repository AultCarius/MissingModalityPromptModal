import torch
import yaml
from trainer import Trainer
from datamodules.MmimdbDataModule import mmimdbDataModule
from models.multimodal_model import create_multimodal_prompt_model

# 1. 加载配置
config_path = "configs/mmimdb.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 2. 设置数据模块
datamodule = mmimdbDataModule(
            data_dir=config.get("data_dir", "./data/mmimdb"),
            batch_size=config.get("batch_size", 32),
            num_workers=config.get("num_workers", 4),
            missing_strategy=config.get("missing_strategy", "none"),  # 直接读缺失策略
            missing_prob=config.get("initial_missing_prob", config.get("missing_prob", 0.7)),  # Use initial value if available
            val_missing_strategy=config.get("val_missing_strategy", "none"),  # 验证集策略
            val_missing_prob=config.get("val_missing_prob", 0.0),  # 验证集缺失率
            test_missing_strategy=config.get("test_missing_strategy", "none"),  # 测试集策略
            test_missing_prob=config.get("test_missing_prob", 0.0),             # 测试集缺失率
            max_length=config.get("max_length", 77),
            image_size=config.get("image_size", 224),
            patch_size=config.get("patch_size", 16),
            seed=config.get("seed",42)
        )
num_classes = 23
datamodule.setup()
test_loader = datamodule.test_dataloader()  # 获取测试数据加载器
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
# 3. 创建模型 - 必须与训练时使用的模型结构完全一致
model = create_multimodal_prompt_model(
    image_model_name=config.get("image_model_name", "vit_base_patch16_224"),
    text_model_name=config.get("text_model_name", "roberta-base"),  # Use roberta-base as default
    image_prompt_len=config.get("image_prompt_len", 5),
    text_prompt_len=config.get("text_prompt_len", 5),
    prompt_depth=config.get("prompt_depth", 6),
    fusion_dim=config.get("fusion_dim", 512),
    num_classes=num_classes,
    freeze_image_encoder=config.get("freeze_image_encoder", False),
    freeze_text_encoder=config.get("freeze_text_encoder", False),
    use_quality_prompt=config.get("use_quality_prompt", False),
    use_cross_modal_prompt=config.get("use_cross_modal_prompt", False),
    max_length=config.get("max_length", 512)
)

# 4. 初始化Trainer
trainer = Trainer(model, train_loader, val_loader, config=config)  # 测试时可以不需要训练和验证数据加载器

# 5. 加载最佳模型并运行测试
best_model_path = "/kaggle/input/best_model/transformers/default/1/best_model.pt"  # 替换为你的模型路径
test_results = trainer.test(test_loader=test_loader, model_path=best_model_path)

# 6. 打印测试结果
print("Test Results:")
for k, v in test_results['overall'].items():
    print(f"  {k}: {v:.4f}")

# 7. 按缺失类型打印详细结果
print("\nResults by Missing Type:")
for mt_name, metrics in test_results['by_missing_type'].items():
    print(f"  {mt_name}:")
    for k, v in metrics.items():
        print(f"    {k}: {v:.4f}")