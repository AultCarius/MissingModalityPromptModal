import torch
import yaml
import os
from trainer import Trainer
from datamodules.Food101DataModule import Food101DataModule
from datamodules.MmimdbDataModule import mmimdbDataModule
from models.multimodal_model import create_multimodal_prompt_model

# Load configuration
if __name__ == '__main__':

    config_path = "configs/mmimdb.yaml"
    # config_path = "/kaggle/working/MissingModalityPromptModal/configs/mmimdb.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 设置实验名称，如果配置文件中没有指定
    if "experiment_name" not in config:
        # 可以根据配置参数自动生成有意义的实验名称
        model_type = config.get("image_model_name", "").split('_')[0]  # 提取模型名称前缀
        missing_prob = config.get("missing_prob", 0.0)
        fusion_dim = config.get("fusion_dim", 512)

        # 创建包含关键参数的实验名称
        config["experiment_name"] = f"{model_type}_fusion{fusion_dim}_miss{missing_prob}"
    print(f"Starting experiment: {config['experiment_name']}")

    # Set up data module
    if config.get("dataset", "mmimdb") == "food101":
        datamodule = Food101DataModule(
            data_dir=config.get("data_dir", "./data/food101"),
            batch_size=config.get("batch_size", 32),
            num_workers=config.get("num_workers", 4),
            image_size=config.get("image_size", 224)
        )
        num_classes = 101
    else:
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

    # 根据数据集类型设置评估指标
    if isinstance(datamodule, mmimdbDataModule):
        config["metrics"] = config.get("metrics")
        config["primary_metric"] = config.get("primary_metric")  # MMIMDB主要使用micro_f1作为评估指标
    elif isinstance(datamodule, Food101DataModule):
        config["metrics"] = config.get("metrics")
        config["primary_metric"] = config.get("primary_metric")  # Food101使用accuracy
    else:
        # 默认添加通用的指标
        config["metrics"] = config.get("metrics")
        config["primary_metric"] = config.get("primary_metric")  # HatefulMemes使用auroc

    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Create model
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
        max_length=config.get("max_length",512)
    )

    # model = torch.nn.DataParallel(model)

    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config=config)

    trainer.class_weights = datamodule.get_class_weights().to(trainer.device)

    # Add this right after creating the trainer instance (around line 95)
    # After this line: trainer = Trainer(model, train_loader, val_loader, config=config)

    # Set up curriculum learning if the config has the required parameters
    if config.get("initial_missing_prob") is not None and config.get("final_missing_prob") is not None:
        trainer.setup_curriculum_learning(
            initial_missing_prob=config.get("initial_missing_prob"),
            final_missing_prob=config.get("final_missing_prob"),
            ramp_epochs=config.get("missing_prob_ramp_epochs", 10)
        )

    # Start training
    trainer.train()

