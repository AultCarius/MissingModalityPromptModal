import json
import os
import yaml
import torch
import random
import logging
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F

from models.improved_modality_generator import Discriminator
from scripts.emailsender import (
    setup_email_config,
    parse_log_file,
    create_training_plots,
    send_email_with_results
)
import torch.nn as nn
from tqdm import tqdm
from scipy.special import softmax

# from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torchmetrics.functional import f1_score, auroc, accuracy

from utils.loss_utils import combined_reconstruction_loss


# torch.autograd.set_detect_anomaly(True)
GENRE_CLASS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure',
    'Horror', 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family',
    'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western',
    'Sport', 'Short', 'Film-Noir'
]
GENRE_CLASS_DICT = {genre: idx for idx, genre in enumerate(GENRE_CLASS)}


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_weights = None
        if isinstance(config, str):
            with open(config, 'r') as f:
                config = yaml.safe_load(f)
        self.config = config
        self.config = setup_email_config(self.config)
        # 设置实验名称
        self.experiment_name = config.get("experiment_name", f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        # 创建实验目录结构
        self._setup_experiment_directories()

        # 保存当前配置和代码文件
        self._save_experiment_config_and_code()

        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self._set_seed(config.get("seed", 42))
        self._setup_logger()
        self._setup_tensorboard()
        self._setup_optimizer()
        self._freeze_modules()

        # 设置评估指标
        self.metrics = config.get("metrics", ["accuracy"])  # 默认使用准确率
        self.best_metrics = {metric: 0.0 for metric in self.metrics}
        self.primary_metric = config.get("primary_metric", "accuracy")  # 用于保存最佳模型的主要指标

        self.start_epoch = 0

        if config.get("resume_path"):
            self._load_checkpoint(config["resume_path"])

        self.batch_size = config.get("batch_size",8)

        self.is_single_label = False

        # Initialize loss weights for different components
        self.loss_weights = {
            'classification': 1.0,
            'reconstruction': 0.5,
            'cycle': 0.5,
            'contrastive': 0.2,
            'quality': 0.1
        }

        # Override with config values if provided
        if 'loss_weights' in config:
            for k, v in config.get('loss_weights', {}).items():
                if k in self.loss_weights:
                    self.loss_weights[k] = v

        # Setup curriculum learning for generator
        self.generator_curriculum = {
            'enabled': True,
            'initial_weight': 0.1,
            'final_weight': 1.0,
            'ramp_epochs': 5
        }

        # Override with config values if provided
        if 'generator_curriculum' in config:
            for k, v in config.get('generator_curriculum', {}).items():
                if k in self.generator_curriculum:
                    self.generator_curriculum[k] = v

        self.logger.info(f"Initialized loss weights: {self.loss_weights}")
        self.logger.info(f"Initialized generator curriculum: {self.generator_curriculum}")


    def send_training_results_email(self):
        """Send training results and logs via email"""
        if not self.config.get("email", {}).get("send_email", False):
            self.logger.info("Email notifications disabled in config")
            return

        self.logger.info("Preparing to send training results via email...")

        # Get paths to logs
        log_dir = self.log_dir
        log_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir)
                     if f.endswith('.log')]

        # Get the most recent log file
        if log_files:
            log_files.sort(key=os.path.getmtime, reverse=True)
            latest_log = log_files[0]
        else:
            latest_log = None

        # Parse log file to extract metrics
        if latest_log:
            metrics = parse_log_file(latest_log)
        else:
            metrics = None

        # Create output directory for plots
        plots_dir = os.path.join(self.base_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Create plots from metrics
        plot_paths = []
        if metrics:
            plot_paths = create_training_plots(metrics, plots_dir)

        # Create summary of best metrics
        best_metrics_summary = {
            metric: value for metric, value in self.best_metrics.items()
        }

        # Send email with results
        success = send_email_with_results(
            self.config,
            [latest_log] if latest_log else [],
            plot_paths,
            best_metrics_summary
        )

        if success:
            self.logger.info("Training results email sent successfully")
        else:
            self.logger.warning("Failed to send training results email")

    def _setup_experiment_directories(self):
        """创建实验相关的目录结构，支持在 Kaggle 上持久化保存"""
        base_root = "experiments"
        # 基础目录
        original_experiment_name = self.experiment_name
        base_dir = os.path.join(base_root, self.experiment_name)

        if os.path.exists(base_dir):
            # 获取当前时间戳并添加到实验名
            from datetime import datetime
            timestamp = datetime.now().strftime("%m%d_%H%M")
            self.experiment_name = f"{original_experiment_name}_{timestamp}"
            base_dir = os.path.join(base_root, self.experiment_name)
            print(f"实验目录 '{original_experiment_name}' 已存在，自动重命名为 '{self.experiment_name}'")

        # 基础目录
        self.base_dir = base_dir

        # 各子目录
        self.save_path = os.path.join(self.base_dir, "checkpoints")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.tb_dir = os.path.join(self.log_dir, "tb")
        self.code_dir = os.path.join(self.base_dir, "code_snapshot")
        self.plot_dir = os.path.join(self.base_dir, "debug_plots")

        # 创建所有目录
        for directory in [self.save_path, self.log_dir, self.tb_dir, self.code_dir,self.plot_dir]:
            os.makedirs(directory, exist_ok=True)

        self.logger_initialized = False

    def _save_experiment_config_and_code(self):
        """保存当前配置文件和关键代码文件"""
        # 保存配置
        config_path = os.path.join(self.code_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        # 保存关键代码文件
        files_to_copy = [
            "trainer.py",
            "test.py",
            "train.py",
            "datamodules/UPMCFood101DataModule.py",
            "datamodules/MmimdbDataModule.py",
            "models/multimodal_model.py",
            "models/quality_aware_prompting.py",
            "models/modality_generator.py"
        ]

        for file_path in files_to_copy:
            try:
                # 确保目标目录存在
                target_dir = os.path.join(self.code_dir, os.path.dirname(file_path))
                os.makedirs(target_dir, exist_ok=True)

                # 复制文件
                if os.path.exists(file_path):
                    shutil.copy2(file_path, os.path.join(self.code_dir, file_path))
                    if self.logger_initialized:
                        self.logger.info(f"Copied file: {file_path}")
            except Exception as e:
                if self.logger_initialized:
                    self.logger.warning(f"Failed to copy file {file_path}: {str(e)}")

    def _set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _setup_logger(self):
        self.logger = logging.getLogger(f"Trainer_{self.experiment_name}")
        self.logger.setLevel(logging.INFO)

        # 清除之前的处理程序（如果有的话）
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 文件处理程序
        log_file = os.path.join(self.log_dir, f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        fh = logging.FileHandler(log_file)
        self.logfilename = log_file

        # 控制台处理程序
        ch = logging.StreamHandler()

        formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger_initialized = True

        # 记录实验开始信息
        self.logger.info(f"Experiment '{self.experiment_name}' started")
        self.logger.info(f"Configuration: {self.config}")
        self.logger.info(f"Model: {type(self.model).__name__}")
        self.logger.info(f"Train dataset size: {len(self.train_loader.dataset)}")
        self.logger.info(f"Validation dataset size: {len(self.val_loader.dataset)}")

    def _setup_tensorboard(self):
        self.writer = None
        if self.config.get("use_tensorboard", False):
            if os.path.exists(self.tb_dir):
                shutil.rmtree(self.tb_dir)
            self.writer = SummaryWriter(self.tb_dir)

    def _freeze_modules(self):
        if self.config.get("freeze_image_encoder", False):
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
            self.logger.info("Image encoder frozen.")
        if self.config.get("freeze_text_encoder", False):
            for param in self.model.text_encoder.parameters():
                param.requires_grad = False
            self.logger.info("Text encoder frozen.")

    def _setup_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 0)
        )

        # 计算总训练步数和预热步数
        num_epochs = self.config.get("epochs", 10)
        steps_per_epoch = len(self.train_loader)
        total_steps = num_epochs * steps_per_epoch

        # 获取预热百分比，默认为10%
        warmup_percent = self.config.get("warmup_percent", 0.1)
        warmup_steps = int(total_steps * warmup_percent)

        # 设置学习率调度器
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.logger.info(f"Using AdamW optimizer with warmup. Total steps: {total_steps}, "
                         f"Warmup steps: {warmup_steps} ({warmup_percent:.1%})")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.start_epoch = checkpoint["epoch"] + 1

        # 加载之前保存的最佳指标值
        if "metrics" in checkpoint:
            self.best_metrics = checkpoint["metrics"]
        else:
            # 兼容旧格式，只有accuracy
            self.best_metrics[self.primary_metric] = checkpoint.get("best_acc", 0.0)

        self.logger.info(f"Resumed from checkpoint: {path}")

    def _save_checkpoint(self, epoch, metrics=None, is_best=False):
        os.makedirs(self.save_path, exist_ok=True)

        # 常规检查点
        filename = f"model_epoch{epoch}.pt"
        path = os.path.join(self.save_path, filename)

        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics or self.best_metrics,
            "config": self.config  # 保存配置信息
        }

        # 如果是最佳模型，另存一份
        if is_best:
            best_path = os.path.join(self.save_path, "best_model.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model to {best_path}")
        else:
            torch.save(checkpoint, path)
            self.logger.info(f"Saved checkpoint to {path}")

    # Modifications to initialize_model in trainer.py

    def _setup_adversarial_optimizers(self):
        """Initialize optimizers for adversarial training"""
        if hasattr(self.model, 'modality_generator'):
            # Split parameters for generator and discriminator
            generator_params = []
            discriminator_params = []

            for name, param in self.model.modality_generator.named_parameters():
                if 'discriminator' in name:
                    discriminator_params.append(param)
                else:
                    generator_params.append(param)

        # Create optimizers
        from torch.optim import Adam

        self.generator_optimizer = Adam(
            generator_params,
            lr=self.config.get("generator_lr", 0.0001),
            betas=(0.5, 0.999),
            weight_decay=self.config.get("generator_weight_decay", 0.0001)
        )

        self.discriminator_optimizer = Adam(
            discriminator_params,
            lr=self.config.get("discriminator_lr", 0.0004),
            betas=(0.5, 0.999),
            weight_decay=self.config.get("discriminator_weight_decay", 0.0001)
        )

        # Setup learning rate schedulers
        from torch.optim.lr_scheduler import CosineAnnealingLR

        self.generator_scheduler = CosineAnnealingLR(
            self.generator_optimizer,
            T_max=self.config.get("epochs", 30),
            eta_min=self.config.get("min_generator_lr", 1e-6)
        )

        self.discriminator_scheduler = CosineAnnealingLR(
            self.discriminator_optimizer,
            T_max=self.config.get("epochs", 30),
            eta_min=self.config.get("min_discriminator_lr", 1e-6)
        )

        self.logger.info("Initialized adversarial training optimizers and schedulers")

    def train(self):
        """
        主训练函数，添加质量评估器的训练和对比损失训练
        """
        # ===== 初始化训练参数 =====
        num_epochs = self.config.get("epochs", 10)
        max_epochs = num_epochs

        # 配置缺失概率课程学习
        if hasattr(self, 'use_curriculum') and self.use_curriculum:
            self.logger.info(
                f"Using curriculum learning for missing probability: {self.initial_missing_prob} → {self.final_missing_prob}")

        # Get dataset type
        dataset_type = self.config.get("dataset", "mmimdb")
        self.is_single_label = dataset_type == "food101"

        # Focal Loss配置
        focal_start_epoch = self.config.get("focal_start_epoch", 3)
        use_focal_loss = self.config.get("use_focal_loss", True)
        focal_alpha = self.config.get("focal_alpha", 0.5)
        focal_max_gamma = self.config.get("focal_gamma", 2.0)
        gamma_ramp_epochs = self.config.get("gamma_ramp_epochs", 5)
        focal_weight = self.config.get("focal_weight", 0.3)

        # 非对称损失(ASL)配置
        use_asymmetric_loss = self.config.get("use_asymmetric_loss", True)
        asl_start_epoch = self.config.get("asl_start_epoch", 3)
        asl_gamma_pos = self.config.get("asl_gamma_pos", 0.0)
        asl_gamma_neg = self.config.get("asl_gamma_neg", 4.0)
        asl_ramp_epochs = self.config.get("asl_ramp_epochs", 3)
        asl_clip = self.config.get("asl_clip", 0.05)

        # 重建损失权重配置
        initial_recon_weight = self.config.get("initial_recon_weight", 0.1)
        final_recon_weight = self.config.get("final_recon_weight", 0.01)


        # ===== Begin training loop =====
        for epoch in range(self.start_epoch, num_epochs):
            self.logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")

            # 应用课程学习调整缺失概率（如果启用）
            if hasattr(self, 'use_curriculum') and self.use_curriculum:
                progress = min(1.0, epoch / self.missing_prob_ramp_epochs)
                current_missing_prob = self.initial_missing_prob + progress * (
                            self.final_missing_prob - self.initial_missing_prob)

                if self.update_missing_probability(current_missing_prob):
                    self.logger.info(f"Epoch {epoch}: Updated missing probability to {current_missing_prob:.3f}")

            # Switch to training mode
            self.model.train()

            # ===== 初始化指标跟踪器 =====
            # Initialize metrics tracking
            total_loss = 0
            cls_loss_sum = 0
            recon_loss_sum = 0
            cycle_loss_sum = 0
            contra_loss_sum = 0
            quality_loss_sum = 0
            distribution_loss_sum =0

            # Prediction tracking
            all_preds, all_labels = [], []

            # Quality assessment and fusion weight statistics
            quality_stats = {'image': [], 'text': [], 'consistency': []}
            fusion_weights_stats = []

            # Modality generation performance statistics
            gen_stats = {'image': {'mse': [], 'count': 0}, 'text': {'mse': [], 'count': 0}}

            # Feature tracking
            all_features = {
                'real_image': [], 'real_text': [],
                'gen_image': [], 'gen_text': [],
                'missing_types': []
            }

            # 为每个模态的分类器单独跟踪性能
            modality_preds = {
                'image': [], 'text': [], 'combined': [], 'original': []
            }
            modality_labels = []

            # Create progress bar
            batch_pbar = tqdm(total=len(self.train_loader),
                              desc=f"Epoch {epoch + 1}/{num_epochs}",
                              dynamic_ncols=True,
                              leave=False)

            # ===== Batch training loop =====
            for batch_idx, batch in enumerate(self.train_loader):
                # Load batch data
                image, input_ids, attention_mask, label, missing_type = [x.to(self.device) for x in batch]

                # Track missing modality types
                is_image_missing = (missing_type == 1) | (missing_type == 3)
                is_text_missing = (missing_type == 2) | (missing_type == 3)
                no_missing = (~is_image_missing) & (~is_text_missing)

                # ===== Forward pass =====
                output = self.model(image, input_ids, attention_mask, missing_type)

                # Process model output (logits and additional info)
                if isinstance(output, tuple):
                    logits, additional_info = output

                    # ===== 提取特征和质量分数 =====
                    # 获取原始特征和生成特征
                    original_features = additional_info.get('original_features', {})
                    generated_features = additional_info.get('generated_features', {})
                    reconstructed_features = additional_info.get('reconstructed_features', {})
                    quality_scores = additional_info.get('quality_scores', None)




                    # ===== 1.分类损失 =====
                    if self.is_single_label:
                        # Single-label classification - use cross-entropy loss
                        targets = label.argmax(dim=1)  # Convert to class indices
                        classification_loss = F.cross_entropy(logits, targets)
                    else:
                        # Multi-label classification - can use different loss types
                        if use_asymmetric_loss and epoch >= asl_start_epoch:
                            # Asymmetric Loss (ASL) - better for imbalanced multi-label classification
                            progress = min(1.0, (epoch - asl_start_epoch + 1) / asl_ramp_epochs)
                            gamma_pos = asl_gamma_pos * progress
                            gamma_neg = asl_gamma_neg * progress
                            classification_loss = self.asymmetric_loss_with_logits(
                                logits, label,
                                gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=asl_clip
                            )
                        elif use_focal_loss and epoch >= focal_start_epoch:
                            # Focal Loss - focus on hard examples
                            progress = min(1.0, (epoch - focal_start_epoch + 1) / gamma_ramp_epochs)
                            gamma = focal_max_gamma * progress
                            bce_loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=self.class_weights)
                            focal = self.focal_loss(logits, label, alpha=focal_alpha, gamma=gamma)
                            classification_loss = bce_loss + focal_weight * focal
                        else:
                            # Standard binary cross-entropy loss
                            classification_loss = F.binary_cross_entropy_with_logits(
                                logits, label, pos_weight=self.class_weights
                            )

                    # ===== 2. Reconstruction and Cycle Consistency Losses =====
                    reconstruction_loss = 0.0
                    cycle_loss = 0.0
                    generation_quality_loss = 0.0

                    # Extract information from additional_info
                    if additional_info:
                        orig_features = additional_info.get('original_features', {})
                        gen_features = additional_info.get('generated_features', {})
                        recon_features = additional_info.get('reconstructed_features', {})
                        cycle_features = additional_info.get('cycle_features', {})

                        # Create masks for different sample types
                        both_modalities = ~is_image_missing & ~is_text_missing  # Complete samples
                        image_only = ~is_image_missing & is_text_missing  # Only image present
                        text_only = is_image_missing & ~is_text_missing  # Only text present
                        both_missing = is_image_missing & is_text_missing  # Both modalities missing

                        # Track reconstruction and cycle loss components
                        recon_components = []
                        cycle_components = []

                        # === 2.1 Process samples with only text present (image is missing) ===
                        if text_only.any() and 'text' in orig_features:
                            text_orig = orig_features['text'][text_only]  # Original text features

                            # Check if we have reconstructed text features
                            text_recon_key = None
                            for key in recon_features.keys():
                                if key.startswith('text_from_') and 'image' in key:
                                    text_recon_key = key
                                    break

                            if text_recon_key and text_recon_key in recon_features:
                                # Get text that was reconstructed from generated image
                                text_recon = recon_features[text_recon_key]
                                if text_only.sum() <= text_recon.size(0):
                                    text_recon = text_recon[text_only]

                                    # Calculate cycle consistency loss (text -> gen image -> recon text)
                                    if text_recon.shape == text_orig.shape:
                                        text_cycle_loss,_ = combined_reconstruction_loss(text_recon, text_orig)
                                        cycle_components.append(text_cycle_loss)

                                        # Log metrics
                                        cycle_loss_sum += text_cycle_loss.item()
                                        gen_stats['image']['mse'].append(text_cycle_loss.item())
                                        gen_stats['image']['count'] += text_only.sum().item()

                        # === 2.2 Process samples with only image present (text is missing) ===
                        if image_only.any() and 'image' in orig_features:
                            image_orig = orig_features['image'][image_only]  # Original image features

                            # Check if we have reconstructed image features
                            image_recon_key = None
                            for key in recon_features.keys():
                                if key.startswith('image_from_') and 'text' in key:
                                    image_recon_key = key
                                    break

                            if image_recon_key and image_recon_key in recon_features:
                                # Get image that was reconstructed from generated text
                                image_recon = recon_features[image_recon_key]
                                if image_only.sum() <= image_recon.size(0):
                                    image_recon = image_recon[image_only]

                                    # Calculate cycle consistency loss (image -> gen text -> recon image)
                                    if image_recon.shape == image_orig.shape:
                                        image_cycle_loss,_ = combined_reconstruction_loss(image_recon, image_orig)
                                        cycle_components.append(image_cycle_loss)

                                        # Log metrics
                                        cycle_loss_sum += image_cycle_loss.item()
                                        gen_stats['text']['mse'].append(image_cycle_loss.item())
                                        gen_stats['text']['count'] += image_only.sum().item()

                        # === 2.3 Process complete samples (both modalities present) ===
                        # For complete samples, we can train the generator with direct reconstruction
                        if both_modalities.any() and recon_features:
                            # Extract original features for complete samples
                            if 'image' in orig_features and 'text' in orig_features:
                                img_orig = orig_features['image'][both_modalities]
                                txt_orig = orig_features['text'][both_modalities]

                                # Check which cycle features we have
                                for key, features in recon_features.items():
                                    if key == 'text_from_image' and features is not None:
                                        # Get generated text from image
                                        txt_from_img = features
                                        if both_modalities.sum() <= txt_from_img.size(0):
                                            txt_from_img = txt_from_img[both_modalities]

                                            # Calculate reconstruction loss (how well image->text works)
                                            if txt_from_img.shape == txt_orig.shape:
                                                txt_gen_loss,_ = combined_reconstruction_loss(txt_from_img, txt_orig)
                                                recon_components.append(txt_gen_loss*2)
                                                recon_loss_sum += txt_gen_loss.item()

                                    elif key == 'image_from_text' and features is not None:
                                        # Get generated image from text
                                        img_from_txt = features
                                        if both_modalities.sum() <= img_from_txt.size(0):
                                            img_from_txt = img_from_txt[both_modalities]

                                            # Calculate reconstruction loss (how well text->image works)
                                            if img_from_txt.shape == img_orig.shape:
                                                img_gen_loss,_ = combined_reconstruction_loss(img_from_txt, img_orig)
                                                recon_components.append(img_gen_loss*2)
                                                recon_loss_sum += img_gen_loss.item()

                        # === 2.4 Calculate combined reconstruction and cycle losses ===
                        if recon_components:
                            reconstruction_loss = sum(recon_components)

                        if cycle_components:
                            cycle_loss = sum(cycle_components) / len(cycle_components)

                        # ===== 3. Contrastive Loss =====
                        contrastive_loss = 0.0
                        contrastive_components = []

                        # Only calculate contrastive loss for complete samples
                        if both_modalities.any() and recon_features:
                            # Original features
                            img_orig = orig_features['image'][
                                both_modalities] if 'image' in orig_features else None
                            txt_orig = orig_features['text'][
                                both_modalities] if 'text' in orig_features else None

                            # Generated features
                            img_gen = recon_features.get('image_from_text')
                            txt_gen = recon_features.get('text_from_image')

                            if img_orig is not None and txt_orig is not None and img_gen is not None and txt_gen is not None:
                                # Prepare features for contrastive loss (flatten multi-token features)
                                if img_orig.dim() > 2:
                                    img_orig = img_orig.mean(dim=1)  # Average over tokens
                                if txt_orig.dim() > 2:
                                    txt_orig = txt_orig.mean(dim=1)
                                if img_gen.dim() > 2:
                                    img_gen = img_gen.mean(dim=1)
                                if txt_gen.dim() > 2:
                                    txt_gen = txt_gen.mean(dim=1)

                                # Ensure all tensors have the same batch size
                                min_batch = min(img_orig.size(0), txt_orig.size(0), img_gen.size(0),
                                                txt_gen.size(0))
                                img_orig = img_orig[:min_batch]
                                txt_orig = txt_orig[:min_batch]
                                img_gen = img_gen[:min_batch]
                                txt_gen = txt_gen[:min_batch]

                                # 1. Image-to-Text alignment (original image features should be close to original text)
                                img_txt_contra = self.modality_contrastive_loss(img_orig, txt_orig)
                                contrastive_components.append(img_txt_contra)

                                # 2. Original-to-Generated alignment
                                img_gen_contra = self.contrastive_loss(img_orig, img_gen)
                                txt_gen_contra = self.contrastive_loss(txt_orig, txt_gen)
                                contrastive_components.extend([img_gen_contra, txt_gen_contra])

                                # 3. Generated-to-Generated alignment (cross-modal)
                                gen_contra = self.modality_contrastive_loss(img_gen, txt_gen)
                                contrastive_components.append(gen_contra)

                        # Calculate average contrastive loss
                        if contrastive_components:
                            contrastive_loss = sum(contrastive_components) / len(contrastive_components)
                            contra_loss_sum += contrastive_loss.item()

                    # ===== 质量评估器训练 =====
                    # ===== 4. Quality Assessment Loss =====
                    quality_loss = 0.0

                    if quality_scores is not None and 'image' in quality_scores and 'text' in quality_scores:
                        # 创建基于缺失类型的质量目标
                        image_quality_target = torch.ones_like(quality_scores['image']['final_score'])
                        text_quality_target = torch.ones_like(quality_scores['text']['final_score'])
                        consistency_target = torch.ones_like(quality_scores['cross_consistency'])

                        # 根据缺失类型调整目标质量分数
                        # 1. 图像缺失样本
                        img_missing = is_image_missing & (~is_text_missing)
                        if img_missing.any():
                            # 图像缺失时：图像质量应低，文本质量应高，一致性适中
                            image_quality_target[img_missing] = 0.5  # 生成的图像质量低
                            text_quality_target[img_missing] = 0.9  # 真实文本质量高
                            consistency_target[img_missing] = 0.5  # 一致性适中

                        # 2. 文本缺失样本
                        txt_missing = (~is_image_missing) & is_text_missing
                        if txt_missing.any():
                            # 文本缺失时：图像质量应高，文本质量应低，一致性适中
                            image_quality_target[txt_missing] = 0.9  # 真实图像质量高
                            text_quality_target[txt_missing] = 0.5  # 生成的文本质量低
                            consistency_target[txt_missing] = 0.5  # 一致性适中

                        # 3. 两个模态都缺失
                        both_missing = is_image_missing & is_text_missing
                        if both_missing.any():
                            # 两个模态都缺失时：两个质量都应低，一致性低
                            image_quality_target[both_missing] = 0.2
                            text_quality_target[both_missing] = 0.2
                            consistency_target[both_missing] = 0.3

                        # 4. 完整样本
                        complete = (~is_image_missing) & (~is_text_missing)
                        if complete.any():
                            # 完整样本：两个质量都应高，一致性高
                            image_quality_target[complete] = 0.9
                            text_quality_target[complete] = 0.9
                            consistency_target[complete] = 0.9

                        # 计算质量评估损失
                        img_quality_loss = F.mse_loss(
                            quality_scores['image']['final_score'],
                            image_quality_target
                        )

                        txt_quality_loss = F.mse_loss(
                            quality_scores['text']['final_score'],
                            text_quality_target
                        )

                        consistency_loss = F.mse_loss(
                            quality_scores['cross_consistency'],
                            consistency_target
                        )

                        # 组合质量损失
                        quality_loss = img_quality_loss + txt_quality_loss + consistency_loss
                        quality_loss_sum += quality_loss.item()

                        # 添加分布一致性约束 - 确保质量分数分布合理
                        # 这将鼓励质量评估器对相似质量的特征给出相似的分数
                        if epoch > 5:  # 在训练初期阶段跳过此损失
                            # 使用相同缺失类型样本的质量分数方差作为正则化项
                            reg_loss = 0.0

                            # 对每种缺失类型计算质量分数分布约束
                            for mask in [img_missing, txt_missing, complete, both_missing]:
                                if mask.sum() > 1:  # 至少需要两个样本
                                    # 图像质量分数方差
                                    img_quality_var = torch.var(quality_scores['image']['final_score'][mask])
                                    # 文本质量分数方差
                                    txt_quality_var = torch.var(quality_scores['text']['final_score'][mask])
                                    reg_weight = 0.05 + 0.15 * min(1.0, epoch / 5)  # 随着训练进行逐渐增加权重
                                    reg_loss += reg_weight * (img_quality_var + txt_quality_var)
                                    # 加权平均方差 - 鼓励同类样本质量分数一致
                                    reg_loss += 0.2 * (img_quality_var + txt_quality_var)

                            # 添加到质量损失
                            quality_loss = quality_loss + reg_loss

                    # ===== 5. Collect Quality Assessment Data =====
                    if 'quality_scores' in additional_info:
                        quality_scores = additional_info['quality_scores']
                        quality_stats['image'].append(quality_scores['image']['final_score'].mean().item())
                        quality_stats['text'].append(quality_scores['text']['final_score'].mean().item())
                        quality_stats['consistency'].append(
                            quality_scores['cross_consistency'].mean().item())

                    # Collect fusion weights data
                    # if 'fusion_weights' in additional_info and additional_info[
                    #     'fusion_weights'] is not None:
                    #     fusion_weights = additional_info['fusion_weights']
                    #     fusion_weights_stats.append(fusion_weights.mean(dim=0).cpu().detach().numpy())



                    # ===== Part 3: Feature consistency loss =====
                    # Only apply if not using adversarial training (which already has cycle consistency)
                    # 如果想要额外添加分布损失监督（在不使用对抗训练或作为补充）
                    if self.config.get("add_distribution_supervision", False):
                        distribution_loss = torch.tensor(0.0, device=self.device)

                        if 'generated_features' in additional_info:
                            gen_feats = additional_info['generated_features']

                            # 1. 图像特征分布匹配
                            if 'image' in gen_feats and gen_feats['image'] is not None and is_image_missing.any():
                                # 找到有真实图像的样本作为参考
                                real_img_samples = ~is_image_missing
                                if real_img_samples.any() and 'image' in original_features:
                                    real_img_feats = original_features['image'][real_img_samples]
                                    gen_img_feats = gen_feats['image'][is_image_missing]

                                    # 处理多token特征
                                    if real_img_feats.dim() > 2:
                                        real_img_feats = real_img_feats.mean(dim=1)
                                    if gen_img_feats.dim() > 2:
                                        gen_img_feats = gen_img_feats.mean(dim=1)

                                    # 计算生成图像特征与真实图像特征的分布距离
                                    if (real_img_feats.numel() > 0 and gen_img_feats.numel() > 0 and
                                            torch.sum(torch.abs(real_img_feats)) > 1e-6 and torch.sum(
                                                torch.abs(gen_img_feats)) > 1e-6):
                                        # 均值匹配
                                        real_mean = real_img_feats.mean(dim=0)
                                        gen_mean = gen_img_feats.mean(dim=0)
                                        mean_loss = F.mse_loss(gen_mean, real_mean)

                                        # 方差匹配
                                        real_var = torch.var(real_img_feats, dim=0,unbiased=False)
                                        gen_var = torch.var(gen_img_feats, dim=0,unbiased=False)
                                        var_loss = F.mse_loss(gen_var, real_var)

                                        # 添加到分布损失
                                        distribution_loss = distribution_loss + mean_loss + 0.5 * var_loss

                            # 2. 文本特征分布匹配
                            if 'text' in gen_feats and gen_feats['text'] is not None and is_text_missing.any():
                                # 找到有真实文本的样本作为参考
                                real_txt_samples = ~is_text_missing
                                if real_txt_samples.any() and 'text' in original_features:
                                    real_txt_feats = original_features['text'][real_txt_samples]
                                    gen_txt_feats = gen_feats['text'][is_text_missing]

                                    # 处理多token特征
                                    if real_txt_feats.dim() > 2:
                                        real_txt_feats = real_txt_feats.mean(dim=1)
                                    if gen_txt_feats.dim() > 2:
                                        gen_txt_feats = gen_txt_feats.mean(dim=1)

                                    # 计算生成文本特征与真实文本特征的分布距离
                                    if real_txt_feats.numel() > 0 and gen_txt_feats.numel() > 0:
                                        # 均值匹配
                                        real_mean = real_txt_feats.mean(dim=0)
                                        gen_mean = gen_txt_feats.mean(dim=0)
                                        mean_loss = F.mse_loss(gen_mean, real_mean)

                                        # 方差匹配
                                        real_var = torch.var(real_txt_feats, dim=0,unbiased=False)
                                        gen_var = torch.var(gen_txt_feats, dim=0,unbiased=False)
                                        var_loss = F.mse_loss(gen_var, real_var)

                                        # 添加到分布损失
                                        distribution_loss = distribution_loss + mean_loss + 0.5 * var_loss


                    # ===== 组合所有损失 =====
                    # ===== 6. Calculate Total Loss with Weighting =====
                    # Initialize with classification loss
                    total_batch_loss = self.loss_weights['classification'] * classification_loss

                    # 添加分布损失到总损失
                    if 'distribution_loss' in locals() and distribution_loss> 0:
                        distribution_loss_weight = 1.0  # 调整权重
                        total_batch_loss = total_batch_loss + distribution_loss_weight * distribution_loss
                        distribution_loss_sum += distribution_loss.item()

                    # Apply curriculum learning for generator components
                    if hasattr(self, 'generator_curriculum') and self.generator_curriculum['enabled']:
                        progress = min(1.0, epoch / self.generator_curriculum['ramp_epochs'])
                        current_weight = self.generator_curriculum['initial_weight'] + progress * (
                                self.generator_curriculum['final_weight'] - self.generator_curriculum[
                            'initial_weight']
                        )

                        # Apply weight to generator losses
                        if reconstruction_loss > 0:
                            total_batch_loss += current_weight * self.loss_weights[
                                'reconstruction'] * reconstruction_loss

                        if cycle_loss > 0:
                            total_batch_loss += current_weight * self.loss_weights['cycle'] * cycle_loss

                        if contrastive_loss > 0:
                            total_batch_loss += current_weight * self.loss_weights[
                                'contrastive'] * contrastive_loss

                        if quality_loss > 0:
                            total_batch_loss += current_weight * self.loss_weights['quality'] * quality_loss
                    else:
                        # Use fixed weights
                        if reconstruction_loss > 0:
                            total_batch_loss += self.loss_weights['reconstruction'] * reconstruction_loss

                        if cycle_loss > 0:
                            total_batch_loss += self.loss_weights['cycle'] * cycle_loss

                        if contrastive_loss > 0:
                            total_batch_loss += self.loss_weights['contrastive'] * contrastive_loss

                        if quality_loss > 0:
                            total_batch_loss += self.loss_weights['quality'] * quality_loss

                    # Track loss components
                    cls_loss_sum += classification_loss.item()


                else:
                    # Simple case: output is just logits without additional info
                    logits = output

                    # Calculate classification loss
                    if self.is_single_label:
                        targets = label.argmax(dim=1)
                        classification_loss = F.cross_entropy(logits, targets)
                    else:
                        classification_loss = F.binary_cross_entropy_with_logits(
                            logits, label, pos_weight=self.class_weights
                        )

                    total_batch_loss = classification_loss
                    cls_loss_sum += classification_loss.item()

                # ===== Optimization step =====
                self.optimizer.zero_grad()
                total_batch_loss.backward()

                self.optimizer.step()
                self.scheduler.step()


                # Track current learning rate
                current_lr = self.scheduler.get_last_lr()[0]

                # Track total loss
                total_loss += total_batch_loss.item()

                # ===== Update progress bar =====
                # Prepare display info for progress bar
                postfix_dict = {
                    "total": f"{total_batch_loss.item():.4f}",
                    "cls": f"{classification_loss.item():.4f}",
                    "recon" : f"{reconstruction_loss:.4f}",
                    "cycle_loss":f"{cycle_loss:.4f}",
                    "contra":f"{contrastive_loss:.4f}",
                    "dis" : f"{distribution_loss.item():.4f}"
                }

                # 如果有质量损失，添加到显示
                if 'quality_loss' in locals() and quality_loss > 0:
                    postfix_dict["qlty"] = f"{quality_loss.item():.4f}"

                # 如果有对比损失，添加到显示
                if 'contrastive_loss' in locals() and contrastive_loss > 0:
                    postfix_dict["cont"] = f"{contrastive_loss.item():.4f}"

                # Add feature generation metrics if available

                # Add quality scores if available
                if additional_info and 'quality_scores' in additional_info:
                    # Show average image quality score for image-missing samples
                    img_missing_mask = is_image_missing
                    if img_missing_mask.any():
                        img_quality = additional_info['quality_scores']['image']['final_score'][img_missing_mask]
                        if len(img_quality) > 0:
                            postfix_dict["img_q"] = f"{img_quality.mean().item():.3f}"

                # 更新进度条
                batch_pbar.set_postfix(postfix_dict)
                batch_pbar.update(1)

                # ===== 收集预测用于计算指标 =====
                # Process predictions based on dataset type
                if self.is_single_label:
                    # Single-label - get class with highest probability
                    pred_indices = logits.argmax(dim=1)
                    # Convert to one-hot for consistent processing
                    preds = torch.zeros_like(logits)
                    preds.scatter_(1, pred_indices.unsqueeze(1), 1.0)
                else:

                    preds = (logits > 0.5).float()  # Simple threshold at 0


                # Collect for later metric calculation
                all_preds.append(preds.cpu().detach())
                all_labels.append(label.cpu().detach())

                # ===== Collect feature statistics =====
                # Track quality assessment data if available
                if additional_info and 'quality_scores' in additional_info:
                    quality_scores = additional_info['quality_scores']
                    quality_stats['image'].append(quality_scores['image']['final_score'].mean().item())
                    quality_stats['text'].append(quality_scores['text']['final_score'].mean().item())
                    quality_stats['consistency'].append(quality_scores['cross_consistency'].mean().item())

                # Track fusion weights if available
                if additional_info and 'fusion_weights' in additional_info and additional_info[
                    'fusion_weights'] is not None:
                    fusion_weights = additional_info['fusion_weights']
                    # fusion_weights_stats.append(fusion_weights.mean(dim=0).cpu().detach().numpy())

                # Collect feature samples for analysis (limited number)
                if batch_idx % 10 == 0 and len(all_features['missing_types']) < 1000:
                    # Track missing types
                    all_features['missing_types'].append(missing_type.cpu().numpy())

                    # Track real features if available
                    if 'original_features' in additional_info:
                        if 'image' in additional_info['original_features']:
                            img_feat = additional_info['original_features']['image']
                            if img_feat is not None and img_feat.dim() > 2:
                                img_feat = img_feat.mean(dim=1)  # Average across tokens
                            all_features['real_image'].append(img_feat.cpu().detach().numpy())

                        if 'text' in additional_info['original_features']:
                            txt_feat = additional_info['original_features']['text']
                            if txt_feat is not None and txt_feat.dim() > 2:
                                txt_feat = txt_feat.mean(dim=1)  # Average across tokens
                            all_features['real_text'].append(txt_feat.cpu().detach().numpy())

                    # Track generated features if available
                    if 'generated_features' in additional_info:
                        if 'image' in additional_info['generated_features'] and additional_info['generated_features'][
                            'image'] is not None:
                            gen_img = additional_info['generated_features']['image']
                            if gen_img.dim() > 2:
                                gen_img = gen_img.mean(dim=1)  # Average across tokens
                            all_features['gen_image'].append(gen_img.cpu().detach().numpy())

                        if 'text' in additional_info['generated_features'] and additional_info['generated_features'][
                            'text'] is not None:
                            gen_txt = additional_info['generated_features']['text']
                            if gen_txt.dim() > 2:
                                gen_txt = gen_txt.mean(dim=1)  # Average across tokens
                            all_features['gen_text'].append(gen_txt.cpu().detach().numpy())

            # Close progress bar
            batch_pbar.close()

            # ===== 轮结束：分析融合数据 =====
            # Check for fusion analyzer and generate reports
            if hasattr(self.model, 'fusion_analyzer') and self.model.fusion_analyzer is not None:
                try:
                    # Generate summary report
                    if callable(getattr(self.model.fusion_analyzer, 'generate_summary_report', None)):
                        self.model.fusion_analyzer.generate_summary_report(epoch)

                    # Analyze modality feature distributions
                    if callable(getattr(self.model.fusion_analyzer, 'analyze_modality_features_distribution', None)):
                        self.model.fusion_analyzer.analyze_modality_features_distribution(epoch)
                except Exception as e:
                    self.logger.warning(f"Error generating fusion analysis: {e}")

            # ===== Compute training metrics =====
            # Merge predictions and labels from all batches
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Calculate metrics
            train_metrics = self._compute_metrics(all_preds, all_labels)

            # Calculate average losses
            num_batches = len(self.train_loader)
            avg_loss = total_loss / num_batches
            avg_cls_loss = cls_loss_sum / num_batches

            # Additional losses if applicable

            avg_quality_loss = quality_loss_sum / num_batches if num_batches > 0 else 0
            avg_recon_loss = recon_loss_sum / num_batches if num_batches > 0 else 0
            avg_cycle_loss = cycle_loss_sum / num_batches if num_batches > 0 else 0
            avg_distribution_loss = distribution_loss_sum / num_batches if num_batches > 0 else 0

            # ===== Log metrics =====
            # Create metrics string
            metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])

            # Log to console
            self.logger.info(
                f"Epoch {epoch} Train: loss={avg_loss:.4f} | cls={avg_cls_loss:.4f} | "
                f"recon={avg_recon_loss:.4f} | cycle={avg_cycle_loss:.4f} | "
                f"quality={avg_quality_loss:.4f} | {metrics_str}"
            )

            # Log to TensorBoard
            if self.writer:
                # Loss components
                self.writer.add_scalar("Loss/train_total", avg_loss, epoch)
                self.writer.add_scalar("Loss/train_cls", avg_cls_loss, epoch)
                self.writer.add_scalar("Loss/train_recon", avg_recon_loss, epoch)
                self.writer.add_scalar("Loss/train_cycle", avg_cycle_loss, epoch)
                self.writer.add_scalar("Loss/train_distribution", avg_distribution_loss, epoch)

                # Training metrics
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"{k}/train", v, epoch)

                # Quality statistics
                if quality_stats['image']:
                    self.writer.add_scalar("Quality/image", np.mean(quality_stats['image']), epoch)
                    self.writer.add_scalar("Quality/text", np.mean(quality_stats['text']), epoch)
                    self.writer.add_scalar("Quality/consistency", np.mean(quality_stats['consistency']), epoch)

                # Fusion weights statistics
                if fusion_weights_stats:
                    avg_weights = np.mean(fusion_weights_stats, axis=0)
                    for i, w in enumerate(avg_weights):
                        self.writer.add_scalar(f"Fusion/weight_{i}", w, epoch)

            # ===== Evaluate on validation set =====
            val_metrics = self.evaluate(epoch)

            # ===== Update learning rate schedulers =====
            # Main model scheduler
            self.scheduler.step()

            # ===== Save checkpoints =====
            # Regular epoch checkpoint
            if epoch != 0 and epoch % self.config.get("save_every_epochs", 5) == 0:
                self._save_checkpoint(epoch, metrics=val_metrics)

            # Best model checkpoint
            is_best = False
            if val_metrics[self.primary_metric] > self.best_metrics[self.primary_metric]:
                self.best_metrics = val_metrics.copy()
                is_best = True
                self._save_checkpoint(epoch, metrics=val_metrics, is_best=True)
                self.logger.info(
                    f"New best model saved: {self.primary_metric} = {val_metrics[self.primary_metric]:.4f}")



        # ===== End of training =====
        self.logger.info(
            f"Training completed. Best {self.primary_metric}: {self.best_metrics[self.primary_metric]:.4f}")

        # Print all best metrics
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in self.best_metrics.items()])
        self.logger.info(f"Best metrics: {metrics_str}")

        # Send training results email if configured
        self.send_training_results_email()

        # Close TensorBoard writer
        if self.writer:
            self.writer.close()


    def _compute_metrics(self, preds, labels):
        """
            使用torchmetrics.functional计算评估指标，适用于单标签(Food101)和多标签(MMIMDB)分类任务。

            Args:
                preds: 预测值（logits或概率）
                       对于Food101: [batch_size, 101]
                       对于MMIMDB: [batch_size, 23]
                labels: 真实标签
                        对于Food101: [batch_size, 101]（one-hot编码）或 [batch_size]（类别索引）
                        对于MMIMDB: [batch_size, 23]（multi-hot编码）

            Returns:
                指标字典
            """
        # 确保输入是PyTorch张量并在同一设备上
        device = preds.device

        # 初始化结果字典
        results = {}

        if self.is_single_label:
            # 单标签分类 (Food101)

            # 获取预测的类别索引
            pred_classes = preds.argmax(dim=1)

            # 如果标签是one-hot编码，转换为类别索引
            if labels.dim() > 1 and labels.size(1) > 1:
                true_classes = labels.argmax(dim=1)
            else:
                true_classes = labels

            try:
                # 计算准确率
                results["accuracy"] = accuracy(
                    pred_classes,
                    true_classes,
                    task="multiclass",
                    num_classes=preds.size(1)
                ).item()

                # 计算F1分数
                results["macro_f1"] = f1_score(
                    pred_classes,
                    true_classes,
                    task="multiclass",
                    num_classes=preds.size(1),
                    average="macro"
                ).item()

                results["micro_f1"] = f1_score(
                    pred_classes,
                    true_classes,
                    task="multiclass",
                    num_classes=preds.size(1),
                    average="micro"
                ).item()

                # 如果需要计算AUROC
                if "auroc" in self.metrics:
                    # 转换logits为概率分布
                    probs = F.softmax(preds, dim=1)
                    try:
                        results["auroc"] = auroc(
                            probs,
                            true_classes,
                            task="multiclass",
                            num_classes=preds.size(1),
                            average="macro"
                        ).item()
                    except Exception as e:
                        self.logger.warning(f"计算AUROC时出错: {str(e)}")
                        results["auroc"] = 0.5  # 随机分类器的默认值
            except Exception as e:
                self.logger.warning(f"计算指标时出错: {str(e)}")
                results["accuracy"] = 0.0
                results["macro_f1"] = 0.0
                results["micro_f1"] = 0.0
                results["auroc"] = 0.5

        else:
            # 多标签分类 (MMIMDB)

            # 使用threshold将logits转换为二值预测
            threshold = 0.5  # 与您原始代码中相同的阈值
            preds_sigmoid = torch.sigmoid(preds)
            binary_preds = (preds_sigmoid > threshold).float()

            try:
                # 计算多标签准确率（元素级匹配）
                correct = (binary_preds == labels).sum().float()
                total = labels.numel()
                results["accuracy"] = (correct / total).item()

                # 计算多标签F1分数
                results["macro_f1"] = f1_score(
                    binary_preds,
                    labels,
                    task="multilabel",
                    num_labels=preds.size(1),
                    average="macro"
                ).item()

                results["micro_f1"] = f1_score(
                    binary_preds,
                    labels,
                    task="multilabel",
                    num_labels=preds.size(1),
                    average="micro"
                ).item()

                # 如果需要计算AUROC
                if "auroc" in self.metrics:
                    try:
                        results["auroc"] = auroc(
                            preds_sigmoid,
                            labels,
                            task="multilabel",
                            num_labels=preds.size(1),
                            average="macro"
                        ).item()
                    except Exception as e:
                        self.logger.warning(f"计算AUROC时出错: {str(e)}")
                        results["auroc"] = 0.5  # 随机分类器的默认值
            except Exception as e:
                self.logger.warning(f"计算指标时出错: {str(e)}")
                results["accuracy"] = 0.0
                results["macro_f1"] = 0.0
                results["micro_f1"] = 0.0
                results["auroc"] = 0.5

        return results




    # 在test或evaluate函数中添加以下代码
    def examine_logits(self, logits, missing_type, labels=None, prefix=""):
        """
        详细检查不同缺失类型下的logits分布

        Args:
            logits: 模型输出的logits [batch_size, num_classes]
            missing_type: 缺失类型标记 [batch_size]
            labels: 可选的真实标签 [batch_size, num_classes]
            prefix: 日志前缀
        """
        # 将张量转移到CPU并转为NumPy数组处理
        logits_np = logits.detach().cpu().numpy()
        missing_type_np = missing_type.detach().cpu().numpy()

        # 分离不同缺失类型的logits
        missing_types = {
            'none': (missing_type_np == 0),
            'image': (missing_type_np == 1),
            'text': (missing_type_np == 2),
            'both': (missing_type_np == 3)
        }

        self.logger.info(f"\n{prefix} Logits Analysis:")

        # 对每种缺失类型分析
        for mt_name, mask in missing_types.items():
            if not np.any(mask):
                continue  # 跳过没有样本的缺失类型

            mt_logits = logits_np[mask]
            sample_count = mt_logits.shape[0]

            # 基本统计信息
            stats = {
                'mean': np.mean(mt_logits),
                'std': np.std(mt_logits),
                'min': np.min(mt_logits),
                'max': np.max(mt_logits),
                'positive%': np.mean(mt_logits > 0) * 100,
                'samples': sample_count
            }

            self.logger.info(f"  {mt_name} ({sample_count} samples):")
            self.logger.info(f"    mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, "
                             f"max={stats['max']:.4f}, positive%={stats['positive%']:.2f}%")

            # 每个类别的详细统计

            if mt_name in ['image', 'none', 'text']:
                self.logger.info(f"    Per-class logits for {mt_name}:")

                # 计算每个类别的统计信息
                for i in range(mt_logits.shape[1]):
                    cls_logits = mt_logits[:, i]
                    cls_stats = {
                        'mean': np.mean(cls_logits),
                        'std': np.std(cls_logits),
                        'min': np.min(cls_logits),
                        'max': np.max(cls_logits),
                        'positive%': np.mean(cls_logits > 0) * 100
                    }

                    # 如果有标签，计算此类别的真实正例比例
                    if labels is not None:
                        labels_np = labels.detach().cpu().numpy()
                        cls_positive = np.mean(labels_np[mask, i]) * 100
                        cls_info = f"Class {i}: mean={cls_stats['mean']:.4f}, positive%={cls_stats['positive%']:.2f}%, true_positive%={cls_positive:.2f}%"
                    else:
                        cls_info = f"Class {i}: mean={cls_stats['mean']:.4f}, positive%={cls_stats['positive%']:.2f}%"

                    self.logger.info(f"      {cls_info}")

                # 计算logits分布
                hist, bins = np.histogram(mt_logits.flatten(), bins=10, range=(-5, 5))
                self.logger.info(f"    Logits histogram:")
                for i, (start, end) in enumerate(zip(bins[:-1], bins[1:])):
                    self.logger.info(f"      [{start:.2f}, {end:.2f}): {hist[i]}")

                # 计算每个样本的预测数量
                pred_counts = np.sum(mt_logits > 0, axis=1)
                unique_counts, count_freqs = np.unique(pred_counts, return_counts=True)
                self.logger.info(f"    Prediction counts per sample:")
                for count, freq in zip(unique_counts, count_freqs):
                    self.logger.info(f"      {count} predictions: {freq} samples ({freq / sample_count * 100:.2f}%)")

    def evaluate(self, epoch=None):
        """在验证集上评估模型，返回各种指标"""
        self.model.eval()
        all_preds, all_labels = [], []
        all_logits = []  # 存储原始logits

        # 跟踪不同模态缺失类型的性能
        missing_type_metrics = {
            'none': {'preds': [], 'labels': [], 'logits': []},
            'image': {'preds': [], 'labels': [], 'logits': []},
            'text': {'preds': [], 'labels': [], 'logits': []},
            'both': {'preds': [], 'labels': [], 'logits': []}
        }

        # 收集质量评分数据
        quality_data = {
            'image_quality': [],
            'text_quality': [],
            'image_dims': [],
            'text_dims': [],
            'consistency': [],
            'missing_types': [],
            'fusion_weights': []
        }

        # 跟踪不同模态缺失类型的性能
        missing_type_metrics = {
            'none': {'preds': [], 'labels': [], 'logits': []},
            'image': {'preds': [], 'labels': [], 'logits': []},
            'text': {'preds': [], 'labels': [], 'logits': []},
            'both': {'preds': [], 'labels': [], 'logits': []}
        }
        modality_preds = {
            'image': [], 'text': [], 'combined': [], 'original': []
        }

        with torch.no_grad():
            for batch in self.val_loader:
                image, input_ids, attention_mask, label, missing_type = [x.to(self.device) for x in batch]
                output = self.model(image, input_ids, attention_mask, missing_type)

                # 确保我们只使用logits而不是额外信息
                if isinstance(output, tuple):
                    logits, additional_info = output

                    # 收集质量评分数据
                    if additional_info and 'quality_scores' in additional_info:
                        quality_scores = additional_info['quality_scores']
                        quality_data['image_quality'].append(quality_scores['image']['final_score'].cpu())
                        quality_data['text_quality'].append(quality_scores['text']['final_score'].cpu())

                        # 收集详细质量维度（如果存在）
                        if 'quality' in quality_scores['image']:
                            quality_data['image_dims'].append(quality_scores['image']['quality'].cpu())
                        if 'quality' in quality_scores['text']:
                            quality_data['text_dims'].append(quality_scores['text']['quality'].cpu())

                        quality_data['consistency'].append(quality_scores['cross_consistency'].cpu())
                        quality_data['missing_types'].append(missing_type.cpu())

                    if 'modality_logits' in additional_info:
                        modality_logits = additional_info['modality_logits']

                        # 收集每种模态分类器的预测
                        for modality, mod_logits in modality_logits.items():
                            # 计算预测
                            if self.is_single_label:
                                # 单标签分类
                                mod_preds = torch.zeros_like(mod_logits)
                                mod_preds.scatter_(1, mod_logits.argmax(dim=1).unsqueeze(1), 1.0)
                            else:
                                # 多标签分类
                                mod_preds = (mod_logits > -0.2).float()

                            # 仅处理有效的预测（非零logits）
                            valid_samples = (mod_logits.abs().sum(dim=1) > 1e-6)

                            if valid_samples.any():
                                valid_preds = mod_preds[valid_samples]
                                valid_labels = label[valid_samples]

                                # 计算指标
                                mod_metrics = self._compute_metrics(valid_preds.cpu(), valid_labels.cpu())

                                # 记录日志
                                self.logger.info(
                                    f"{modality} classifier metrics ({valid_samples.sum().item()} samples):")
                                metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in mod_metrics.items()])
                                self.logger.info(f"  {metrics_str}")
                else:
                    logits = output

                # 根据数据集类型计算预测
                if self.is_single_label:
                    # 单标签分类 - 使用argmax
                    pred_indices = logits.argmax(dim=1)
                    # 转换回one-hot格式以保持一致
                    preds = torch.zeros_like(logits)
                    preds.scatter_(1, pred_indices.unsqueeze(1), 1.0)
                else:
                    # 多标签分类 - 使用阈值
                    preds = (logits > -0.2).float()

                # 收集总体预测和标签
                all_preds.append(preds.cpu())
                all_labels.append(label.cpu())
                all_logits.append(logits.cpu())

                # 按缺失类型分类收集预测和标签
                for b in range(missing_type.size(0)):
                    mt = missing_type[b].item()
                    mt_name = ['none', 'image', 'text', 'both'][mt]

                    missing_type_metrics[mt_name]['preds'].append(preds[b:b + 1].cpu())
                    missing_type_metrics[mt_name]['labels'].append(label[b:b + 1].cpu())
                    missing_type_metrics[mt_name]['logits'].append(logits[b:b + 1].cpu())

        # 合并所有批次的预测和标签
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        print("检查logits")
        self.examine_logits(logits, missing_type)

        # 计算总体指标
        metrics = self._compute_metrics(all_preds, all_labels)

        # 详细统计信息
        if self.is_single_label:
            # 使用Food101的详细统计函数
            self._detailed_statistics_food101(all_logits, all_labels, all_preds, "Overall")
        else:
            # 使用MMIMDB的详细统计函数
            self._detailed_statistics(all_logits, all_labels, all_preds, "Overall")

        # 计算每种缺失类型的指标
        missing_type_results = {}
        for mt_name, data in missing_type_metrics.items():
            if data['preds'] and data['labels']:
                mt_preds = torch.cat(data['preds'], dim=0)
                mt_labels = torch.cat(data['labels'], dim=0)
                mt_logits = torch.cat(data['logits'], dim=0)
                mt_metrics = self._compute_metrics(mt_preds, mt_labels)
                missing_type_results[mt_name] = mt_metrics
                if self.is_single_label:
                    # 为每种缺失类型记录详细统计信息
                    self._detailed_statistics_food101(mt_logits, mt_labels, mt_preds, f"Missing: {mt_name}")
                else:
                    self._detailed_statistics(mt_logits, mt_labels, mt_preds, f"Missing: {mt_name}")

        # 计算每种模态的指标
        modality_results = {}
        for modal_name, preds_list in modality_preds.items():
            if preds_list:
                modal_preds = torch.cat(preds_list, dim=0)
                modal_metrics = self._compute_metrics(modal_preds, all_labels)
                modality_results[modal_name] = modal_metrics

        # 记录每种模态的性能
        self.logger.info("\nModality-specific performance (Validation):")
        for modal_name, modal_metrics in modality_results.items():
            metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in modal_metrics.items()])
            self.logger.info(f"  {modal_name.capitalize()}: {metrics_str}")

            # 写入TensorBoard
            if self.writer and epoch is not None:
                for k, v in modal_metrics.items():
                    self.writer.add_scalar(f"{k}/val_{modal_name}", v, epoch)


        # 处理收集的质量评分数据
        if quality_data['image_quality']:
            # 合并所有批次的质量数据
            quality_data['image_quality'] = torch.cat(quality_data['image_quality'], dim=0)
            quality_data['text_quality'] = torch.cat(quality_data['text_quality'], dim=0)
            quality_data['consistency'] = torch.cat(quality_data['consistency'], dim=0)
            quality_data['missing_types'] = torch.cat(quality_data['missing_types'], dim=0)

            if quality_data['image_dims']:
                quality_data['image_dims'] = torch.cat(quality_data['image_dims'], dim=0)
                quality_data['text_dims'] = torch.cat(quality_data['text_dims'], dim=0)

            if quality_data['fusion_weights']:
                quality_data['fusion_weights'] = torch.cat(quality_data['fusion_weights'], dim=0)

            # 可视化质量评分
            if self.writer and epoch is not None:
                self.visualize_quality_scores(
                    epoch,
                    {
                        'image': {
                            'final_score': quality_data['image_quality'] if 'image_quality' in quality_data else None,
                            'quality': quality_data['image_dims'] if 'image_dims' in quality_data and quality_data[
                                'image_dims'] is not None else None
                        },
                        'text': {
                            'final_score': quality_data['text_quality'] if 'text_quality' in quality_data else None,
                            'quality': quality_data['text_dims'] if 'text_dims' in quality_data and quality_data[
                                'text_dims'] is not None else None
                        },
                        'cross_consistency': quality_data['consistency'] if 'consistency' in quality_data else None
                    },
                    quality_data['missing_types'] if 'missing_types' in quality_data else None
                )

                # Visualize fusion weights if available
                if 'fusion_weights' in quality_data and quality_data['fusion_weights'] is not None:
                    self.visualize_fusion_weights(epoch, quality_data['fusion_weights'],
                                                  quality_data[
                                                      'missing_types'] if 'missing_types' in quality_data else None)



            # 按质量分数分层分析性能
            img_quality = quality_data['image_quality'].squeeze()
            txt_quality = quality_data['text_quality'].squeeze()

            # 图像质量分组
            img_high_mask = img_quality > 0.7
            img_med_mask = (img_quality > 0.4) & (img_quality <= 0.7)
            img_low_mask = img_quality <= 0.4

            # 文本质量分组
            txt_high_mask = txt_quality > 0.7
            txt_med_mask = (txt_quality > 0.4) & (txt_quality <= 0.7)
            txt_low_mask = txt_quality <= 0.4

            # 计算各组性能
            quality_group_metrics = {}

            # 记录日志
            self.logger.info("\nPerformance by quality group:")

            # 图像质量分组性能
            for name, mask in [
                ('img_high', img_high_mask),
                ('img_med', img_med_mask),
                ('img_low', img_low_mask),
                ('txt_high', txt_high_mask),
                ('txt_med', txt_med_mask),
                ('txt_low', txt_low_mask)
            ]:
                if mask.sum() > 10:  # 至少10个样本才计算指标
                    group_preds = all_preds[mask]
                    group_labels = all_labels[mask]
                    group_metrics = self._compute_metrics(group_preds, group_labels)
                    quality_group_metrics[name] = group_metrics

                    # 记录日志
                    self.logger.info(f"  - {name} quality group ({mask.sum()} samples):")
                    metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in group_metrics.items()])
                    self.logger.info(f"    {metrics_str}")

                    # 写入TensorBoard
                    if self.writer and epoch is not None:
                        for k, v in group_metrics.items():
                            self.writer.add_scalar(f"{k}/{name}", v, epoch)

            # 分析质量评估器的校准情况
            self.logger.info("\nQuality calibration analysis:")

            # 按图像质量四等分，检验分数与性能的相关性
            try:
                from sklearn.model_selection import KFold
                import numpy as np

                # 转换为NumPy数组
                img_quality_np = img_quality.numpy()
                txt_quality_np = txt_quality.numpy()

                # 按质量分数排序的索引
                img_sorted_idx = np.argsort(img_quality_np)
                txt_sorted_idx = np.argsort(txt_quality_np)

                # 等分成4组
                img_quartiles = np.array_split(img_sorted_idx, 4)
                txt_quartiles = np.array_split(txt_sorted_idx, 4)

                # 图像质量校准
                self.logger.info("  Image quality calibration:")
                for i, indices in enumerate(img_quartiles):
                    group_preds = all_preds[indices]
                    group_labels = all_labels[indices]
                    group_metrics = self._compute_metrics(group_preds, group_labels)
                    group_quality = img_quality_np[indices].mean()

                    metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in group_metrics.items()])
                    self.logger.info(f"    Quartile {i + 1} - Avg Quality: {group_quality:.4f} | {metrics_str}")

                # 文本质量校准
                self.logger.info("  Text quality calibration:")
                for i, indices in enumerate(txt_quartiles):
                    group_preds = all_preds[indices]
                    group_labels = all_labels[indices]
                    group_metrics = self._compute_metrics(group_preds, group_labels)
                    group_quality = txt_quality_np[indices].mean()

                    metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in group_metrics.items()])
                    self.logger.info(f"    Quartile {i + 1} - Avg Quality: {group_quality:.4f} | {metrics_str}")

            except Exception as e:
                self.logger.warning(f"Failed to perform quality calibration analysis: {str(e)}")

            # 分析每种缺失类型的质量分数和性能
            missing_types = quality_data['missing_types']

            self.logger.info("\nQuality and performance by missing type:")
            for mt, mt_name in enumerate(['none', 'image', 'text', 'both']):
                mask = (missing_types == mt)
                if mask.sum() > 0:
                    # 获取该缺失类型的质量分数
                    mt_img_quality = quality_data['image_quality'][mask].mean().item()
                    mt_txt_quality = quality_data['text_quality'][mask].mean().item()
                    mt_consistency = quality_data['consistency'][mask].mean().item()

                    self.logger.info(f"  - {mt_name} is missing type ({mask.sum().item()} samples):")
                    self.logger.info(
                        f"    Quality - Image: {mt_img_quality:.4f} | Text: {mt_txt_quality:.4f} | Consistency: {mt_consistency:.4f}")

                    if mt_name in missing_type_results:
                        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in missing_type_results[mt_name].items()])
                        self.logger.info(f"    Performance - {metrics_str}")

        # 记录缺失类型的性能
        if self.writer and epoch is not None:
            for mt_name, mt_metrics in missing_type_results.items():
                for k, v in mt_metrics.items():
                    self.writer.add_scalar(f"{k}/{mt_name}", v, epoch)

        # 记录每种缺失类型的样本数
        for mt_name, data in missing_type_metrics.items():
            if data['preds']:
                sample_count = len(data['preds'])
                self.logger.info(f"  - {mt_name}: {sample_count} samples")
                if missing_type_results.get(mt_name):
                    metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in missing_type_results[mt_name].items()])
                    self.logger.info(f"    {metrics_str}")

        # 记录日志
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"[Val] Epoch {epoch}: {metrics_str}")

        # 写入TensorBoard
        if self.writer and epoch is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{k}/val", v, epoch)

            # 更新质量评估器的性能统计 - 新增
            if hasattr(self.model, 'quality_estimator') and hasattr(self.model.quality_estimator,
                                                                    'update_performance_stats'):
                # 获取不同缺失类型的性能
                img_missing_perf = missing_type_results.get('image', {}).get(self.primary_metric, 0.4)
                txt_missing_perf = missing_type_results.get('text', {}).get(self.primary_metric, 0.5)

                # 更新质量评估器
                self.model.quality_estimator.update_performance_stats(
                    torch.tensor(img_missing_perf),
                    torch.tensor(txt_missing_perf)
                )

                # 记录日志
                self.logger.info(
                    f"Updated modality performance stats - Image missing: {img_missing_perf:.4f}, Text missing: {txt_missing_perf:.4f}")

        return metrics

    def visualize_quality_scores(self, epoch, quality_scores, missing_types):
        """可视化不同缺失类型的质量分数"""
        if not self.writer:
            return

        # 将张量转换为NumPy数组
        img_quality = quality_scores['image']['final_score'].cpu().numpy()
        txt_quality = quality_scores['text']['final_score'].cpu().numpy()
        consistency = quality_scores['cross_consistency'].cpu().numpy()
        missing = missing_types.cpu().numpy()

        # 计算不同缺失类型的平均分数
        missing_labels = ['none', 'image', 'text', 'both']
        img_by_missing = [img_quality[missing == i].mean() if (missing == i).any() else 0 for i in range(4)]
        txt_by_missing = [txt_quality[missing == i].mean() if (missing == i).any() else 0 for i in range(4)]
        cons_by_missing = [consistency[missing == i].mean() if (missing == i).any() else 0 for i in range(4)]

        # 创建图表
        try:
            import matplotlib.pyplot as plt
            import numpy as np

            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(4)
            width = 0.25

            ax.bar(x - width, img_by_missing, width, label='Image Quality')
            ax.bar(x, txt_by_missing, width, label='Text Quality')
            ax.bar(x + width, cons_by_missing, width, label='Consistency')

            ax.set_xticks(x)
            ax.set_xticklabels(missing_labels)
            ax.set_ylabel('Quality Score')
            ax.set_title(f'Quality Scores by Missing Type (Epoch {epoch})')
            ax.legend()

            # 添加到TensorBoard
            self.writer.add_figure('quality/by_missing_type', fig, epoch)
            plt.close(fig)

            # 记录平均分数
            self.writer.add_scalars('quality/avg_scores', {
                'image': img_quality.mean(),
                'text': txt_quality.mean(),
                'consistency': consistency.mean()
            }, epoch)

            # 创建质量分数直方图
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))

            axs[0].hist(img_quality.flatten(), bins=20, range=(0, 1))
            axs[0].set_title('Image Quality Distribution')
            axs[0].set_xlabel('Quality Score')

            axs[1].hist(txt_quality.flatten(), bins=20, range=(0, 1))
            axs[1].set_title('Text Quality Distribution')
            axs[1].set_xlabel('Quality Score')

            axs[2].hist(consistency.flatten(), bins=20, range=(0, 1))
            axs[2].set_title('Consistency Distribution')
            axs[2].set_xlabel('Consistency Score')

            fig.tight_layout()
            self.writer.add_figure('quality/distributions', fig, epoch)
            plt.close(fig)

            # 如果有详细质量维度，也可视化它们
            if 'quality' in quality_scores['image'] and quality_scores['image']['quality'] is not None:
                img_dims = quality_scores['image']['quality'].cpu().numpy()
                txt_dims = quality_scores['text']['quality'].cpu().numpy()

                # 计算各维度的均值
                img_dim_means = img_dims.mean(axis=0)
                txt_dim_means = txt_dims.mean(axis=0)

                # 可视化质量维度
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

                dim_names_img = ['Clarity', 'Completeness', 'Informativeness', 'Confidence', 'Detail'][
                                :img_dims.shape[1]]
                dim_names_txt = ['Coherence', 'Relevance', 'Informativeness', 'Confidence', 'Detail'][
                                :txt_dims.shape[1]]

                ax1.bar(range(len(img_dim_means)), img_dim_means)
                ax1.set_xticks(range(len(img_dim_means)))
                ax1.set_xticklabels(dim_names_img, rotation=45)
                ax1.set_title('Image Quality Dimensions')

                ax2.bar(range(len(txt_dim_means)), txt_dim_means)
                ax2.set_xticks(range(len(txt_dim_means)))
                ax2.set_xticklabels(dim_names_txt, rotation=45)
                ax2.set_title('Text Quality Dimensions')

                fig.tight_layout()
                self.writer.add_figure('quality/dimensions', fig, epoch)
                plt.close(fig)
        except Exception as e:
            self.logger.warning(f"Failed to create quality visualization: {str(e)}")

    def visualize_fusion_weights(self, epoch, fusion_weights, missing_types):
        """可视化不同缺失类型的融合权重"""
        if not self.writer:
            return

        try:
            import matplotlib.pyplot as plt
            import numpy as np

            # 将张量转换为NumPy数组
            weights = fusion_weights.cpu().numpy()
            missing = missing_types.cpu().numpy()

            # 平均融合权重
            avg_weights = weights.mean(axis=0)

            # 按缺失类型计算平均权重
            missing_labels = ['none', 'image', 'text', 'both']
            weights_by_missing = [
                weights[missing == i].mean(axis=0) if (missing == i).any() else np.zeros_like(avg_weights) for i in
                range(4)]

            # 创建图表
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(avg_weights))
            width = 0.2

            # 绘制各缺失类型的权重
            for i, (mt_name, mt_weights) in enumerate(zip(missing_labels, weights_by_missing)):
                offset = (i - 1.5) * width
                ax.bar(x + offset, mt_weights, width, label=mt_name)

            ax.set_xticks(x)
            ax.set_xticklabels([f'Weight {i + 1}' for i in range(len(avg_weights))])
            ax.set_ylabel('Weight Value')
            ax.set_title(f'Fusion Weights by Missing Type (Epoch {epoch})')
            ax.legend()

            # 添加到TensorBoard
            self.writer.add_figure('fusion/weights_by_missing_type', fig, epoch)
            plt.close(fig)

            # 记录平均融合权重
            for i, w in enumerate(avg_weights):
                self.writer.add_scalar(f'fusion/weight_{i + 1}', w, epoch)
        except Exception as e:
            self.logger.warning(f"Failed to create fusion weights visualization: {str(e)}")

    def _detailed_statistics(self, logits, labels, preds, missing_type=None, prefix=""):
        """
        Compute and print detailed statistics, optionally by missing_type.

        Args:
            logits: Tensor [B, C]
            labels: Tensor [B, C]
            preds: Tensor [B, C]
            missing_type: Optional Tensor [B,] with values 0, 1, 2, 3
            prefix: Logging prefix
        """
        import numpy as np
        logits_np = logits.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()

        total_samples = labels_np.shape[0]
        positive_counts = labels_np.sum(axis=0)
        pred_counts = preds_np.sum(axis=0)

        class_metrics = []

        for i, genre in enumerate(GENRE_CLASS):
            true_pos = ((preds_np[:, i] > 0.5) & (labels_np[:, i] > 0.5)).sum()
            false_pos = ((preds_np[:, i] > 0.5) & (labels_np[:, i] <= 0.5)).sum()
            false_neg = ((preds_np[:, i] <= 0.5) & (labels_np[:, i] > 0.5)).sum()
            true_neg = ((preds_np[:, i] <= 0.5) & (labels_np[:, i] <= 0.5)).sum()

            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_pos + true_neg) / total_samples

            mean_logit = logits_np[:, i].mean()
            pos_logit = logits_np[labels_np[:, i] > 0.5, i].mean() if (labels_np[:, i] > 0.5).any() else float('nan')
            neg_logit = logits_np[labels_np[:, i] <= 0.5, i].mean() if (labels_np[:, i] <= 0.5).any() else float('nan')

            class_metrics.append({
                'genre': genre,
                'pos_count': int(positive_counts[i]),
                'pred_count': int(pred_counts[i]),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_logit': mean_logit,
                'pos_logit': pos_logit,
                'neg_logit': neg_logit
            })

        self.logger.info(f"\n{prefix} Detailed Statistics:")
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Average labels per sample: {labels_np.sum() / total_samples:.2f}")
        self.logger.info(f"Average predictions per sample: {preds_np.sum() / total_samples:.2f}")

        # 打印类别级别统计
        headers = ["Genre", "Pos%", "PredCnt", "Acc", "Prec", "Rec", "F1", "AvgLog", "PosLog", "NegLog"]
        format_row = "{:12} {:6} {:8} {:6} {:6} {:6} {:6} {:7} {:7} {:7}"
        self.logger.info(f"\nClass-level metrics [{prefix}]:")
        self.logger.info(format_row.format(*headers))
        for cm in sorted(class_metrics, key=lambda x: x['pos_count'], reverse=True):
            pos_percent = cm['pos_count'] / total_samples * 100
            self.logger.info(format_row.format(
                cm['genre'][:12],
                f"{pos_percent:.1f}%",
                cm['pred_count'],
                f"{cm['accuracy'] * 100:.2f}",
                f"{cm['precision'] * 100:.2f}",
                f"{cm['recall'] * 100:.2f}",
                f"{cm['f1'] * 100:.2f}",
                f"{cm['mean_logit']:.2f}",
                f"{0 if np.isnan(cm['pos_logit']) else cm['pos_logit']:.2f}",
                f"{0 if np.isnan(cm['neg_logit']) else cm['neg_logit']:.2f}",
            ))

        # 额外统计预测为0的样本数
        zero_pred = (preds_np.sum(axis=1) == 0).sum()
        if zero_pred > 0:
            self.logger.info(f"\n警告: {zero_pred}个样本({zero_pred / total_samples * 100:.2f}%)没有任何正向预测!")

        # 统计预测数量分布
        pred_counts_per_sample = preds_np.sum(axis=1)
        for i in range(min(5, int(pred_counts_per_sample.max())) + 1):
            count = (pred_counts_per_sample == i).sum()
            self.logger.info(f"样本有{i}个预测: {count}个 ({count / total_samples * 100:.2f}%)")

        # ========== 【新增】按缺失类型分组统计 ==========
        if missing_type is not None:
            mt = missing_type.cpu().numpy()
            for mtype in [0, 1, 2, 3]:
                idx = (mt == mtype)
                if idx.sum() == 0:
                    continue
                mean_logits_per_class = logits_np[idx].mean(axis=0)
                pred_rate = preds_np[idx].sum(axis=0) / idx.sum()
                self.logger.info(f"\n[{prefix}] MissingType={mtype} ({idx.sum()} samples):")
                for i, genre in enumerate(GENRE_CLASS):
                    self.logger.info(
                        f"  {genre:12s} | MeanLogit: {mean_logits_per_class[i]:.3f} | Pred%: {pred_rate[i] * 100:.2f}%")

    def _detailed_statistics(self, logits, labels, preds, prefix=""):
        """计算并打印详细的预测统计信息"""
        # 转为numpy方便处理
        logits_np = logits.numpy()
        labels_np = labels.numpy()
        preds_np = preds.numpy()

        # 1. 每个类别的正样本数量
        positive_counts = labels_np.sum(axis=0)
        total_samples = labels_np.shape[0]

        # 2. 每个类别的预测分布
        pred_counts = preds_np.sum(axis=0)

        # 3. 类别级别的准确率、精确率、召回率和F1
        class_metrics = []

        for i, genre in enumerate(GENRE_CLASS):
            # 真正例数量
            true_pos = ((preds_np[:, i] > 0.5) & (labels_np[:, i] > 0.5)).sum()
            # 假正例数量
            false_pos = ((preds_np[:, i] > 0.5) & (labels_np[:, i] <= 0.5)).sum()
            # 假负例数量
            false_neg = ((preds_np[:, i] <= 0.5) & (labels_np[:, i] > 0.5)).sum()
            # 真负例数量
            true_neg = ((preds_np[:, i] <= 0.5) & (labels_np[:, i] <= 0.5)).sum()

            # 计算指标
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (true_pos + true_neg) / total_samples

            # 计算平均logit值
            mean_logit = logits_np[:, i].mean()
            pos_logit = logits_np[labels_np[:, i] > 0.5, i].mean() if (labels_np[:, i] > 0.5).any() else float('nan')
            neg_logit = logits_np[labels_np[:, i] <= 0.5, i].mean() if (labels_np[:, i] <= 0.5).any() else float('nan')

            class_metrics.append({
                'genre': genre,
                'pos_count': int(positive_counts[i]),
                'pred_count': int(pred_counts[i]),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_logit': mean_logit,
                'pos_logit': pos_logit,
                'neg_logit': neg_logit
            })

        # 打印结果
        self.logger.info(f"\n{prefix} Detailed Statistics:")
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Average labels per sample: {labels_np.sum() / total_samples:.2f}")
        self.logger.info(f"Average predictions per sample: {preds_np.sum() / total_samples:.2f}")

        # 打印类别级别统计
        self.logger.info(f"\nClass-level metrics [{prefix}]:")
        headers = ["Genre", "Pos%", "PredCnt", "Acc", "Prec", "Rec", "F1", "AvgLog", "PosLog", "NegLog"]
        format_row = "{:12} {:6} {:8} {:6} {:6} {:6} {:6} {:7} {:7} {:7}"

        # 打印表头
        self.logger.info(format_row.format(*headers))

        # 打印每一行数据
        for cm in sorted(class_metrics, key=lambda x: x['pos_count'], reverse=True):
            pos_percent = cm['pos_count'] / total_samples * 100
            self.logger.info(format_row.format(
                cm['genre'][:12],
                f"{pos_percent:.1f}%",
                cm['pred_count'],
                f"{cm['accuracy'] * 100:.2f}",
                f"{cm['precision'] * 100:.2f}",
                f"{cm['recall'] * 100:.2f}",
                f"{cm['f1'] * 100:.2f}",
                f"{cm['mean_logit']:.2f}",
                f"{0 if np.isnan(cm['pos_logit']) else cm['pos_logit']:.2f}",
                f"{0 if np.isnan(cm['neg_logit']) else cm['neg_logit']:.2f}",
            ))

        # 检查模型预测全为0或全为1的情况
        zero_pred = (preds_np.sum(axis=1) == 0).sum()
        if zero_pred > 0:
            self.logger.info(f"\n警告: {zero_pred}个样本({zero_pred / total_samples * 100:.2f}%)没有任何正向预测!")

        # 计算预测数量分布
        pred_counts_per_sample = preds_np.sum(axis=1)
        for i in range(min(5, int(max(pred_counts_per_sample))) + 1):
            count = (pred_counts_per_sample == i).sum()
            self.logger.info(f"样本有{i}个预测: {count}个 ({count / total_samples * 100:.2f}%)")



    def _detailed_statistics_food101(self, logits, labels, preds, prefix=""):
        """计算并打印Food101数据集的详细预测统计信息"""
        # 转为numpy方便处理
        logits_np = logits.numpy()
        labels_np = labels.numpy()
        preds_np = preds.numpy()

        # 获取Food101类别名称
        food_classes = self._get_food101_classes()

        # 1. 每个类别的正样本数量
        positive_counts = labels_np.sum(axis=0)
        total_samples = labels_np.shape[0]

        # 2. 每个类别的预测分布
        pred_counts = preds_np.sum(axis=0)

        # 3. 类别级别的准确率、精确率、召回率和F1
        class_metrics = []

        for i, food_class in enumerate(food_classes):
            # 真正例数量
            true_pos = ((preds_np[:, i] > 0.5) & (labels_np[:, i] > 0.5)).sum()
            # 假正例数量
            false_pos = ((preds_np[:, i] > 0.5) & (labels_np[:, i] <= 0.5)).sum()
            # 假负例数量
            false_neg = ((preds_np[:, i] <= 0.5) & (labels_np[:, i] > 0.5)).sum()
            # 真负例数量
            true_neg = ((preds_np[:, i] <= 0.5) & (labels_np[:, i] <= 0.5)).sum()

            # 计算指标
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            # 平衡准确率 - 更有意义的单类准确率指标
            tpr = recall  # 真正率 = 召回率
            tnr = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0  # 真负率
            balanced_acc = (tpr + tnr) / 2

            # 计算平均logit值
            mean_logit = logits_np[:, i].mean()
            pos_logit = logits_np[labels_np[:, i] > 0.5, i].mean() if (labels_np[:, i] > 0.5).any() else float('nan')
            neg_logit = logits_np[labels_np[:, i] <= 0.5, i].mean() if (labels_np[:, i] <= 0.5).any() else float('nan')

            class_metrics.append({
                'food_class': food_class,
                'pos_count': int(positive_counts[i]),
                'pred_count': int(pred_counts[i]),
                'avg_accuracy': balanced_acc,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'mean_logit': mean_logit,
                'pos_logit': pos_logit,
                'neg_logit': neg_logit
            })

        # 打印结果
        self.logger.info(f"\n{prefix} Detailed Statistics:")
        self.logger.info(f"Total samples: {total_samples}")
        self.logger.info(f"Average labels per sample: {labels_np.sum() / total_samples:.2f}")
        self.logger.info(f"Average predictions per sample: {preds_np.sum() / total_samples:.2f}")

        # 打印类别级别统计 - 仅显示前20个类别以避免太长
        self.logger.info(f"\nClass-level metrics (top 20) [{prefix}]:")
        headers = ["Food Class", "Pos%", "PredCnt", "Acc", "Prec", "Rec", "F1", "AvgLog", "PosLog", "NegLog"]
        format_row = "{:20} {:6} {:8} {:6} {:6} {:6} {:6} {:7} {:7} {:7}"

        # 打印表头
        self.logger.info(format_row.format(*headers))

        # 按正样本数量排序并打印前20个
        sorted_metrics = sorted(class_metrics, key=lambda x: x['pos_count'], reverse=True)
        for cm in sorted_metrics[:20]:
            pos_percent = cm['pos_count'] / total_samples * 100
            self.logger.info(format_row.format(
                cm['food_class'][:20],  # 截断长类名
                f"{pos_percent:.1f}%",
                cm['pred_count'],
                f"{cm['avg_accuracy'] * 100:.2f}",
                f"{cm['precision'] * 100:.2f}",
                f"{cm['recall'] * 100:.2f}",
                f"{cm['f1'] * 100:.2f}",
                f"{cm['mean_logit']:.2f}",
                f"{0 if np.isnan(cm['pos_logit']) else cm['pos_logit']:.2f}",
                f"{0 if np.isnan(cm['neg_logit']) else cm['neg_logit']:.2f}",
            ))

        # 检查模型预测全为0或全为1的情况
        zero_pred = (preds_np.sum(axis=1) == 0).sum()
        if zero_pred > 0:
            self.logger.info(f"\n警告: {zero_pred}个样本({zero_pred / total_samples * 100:.2f}%)没有任何正向预测!")

        # 计算预测数量分布
        pred_counts_per_sample = preds_np.sum(axis=1)
        for i in range(min(5, int(max(pred_counts_per_sample))) + 1):
            count = (pred_counts_per_sample == i).sum()
            self.logger.info(f"样本有{i}个预测: {count}个 ({count / total_samples * 100:.2f}%)")

    def _get_food101_classes(self):
        """获取Food101数据集的类别名称"""
        # 你可以从配置或数据集中读取类别名称
        # 这里是一种简单的实现方式，实际使用时可能需要调整
        try:
            # 尝试从数据加载器获取类别名称
            if hasattr(self.train_loader.dataset, 'class_to_idx'):
                class_to_idx = self.train_loader.dataset.class_to_idx
                # 按索引排序类别
                classes = [None] * len(class_to_idx)
                for cls, idx in class_to_idx.items():
                    classes[idx] = cls
                return classes

            # 或者直接从数据集目录读取
            if hasattr(self, 'config') and 'data_dir' in self.config:
                class_idx_path = os.path.join(self.config['data_dir'], "class_idx.json")
                if os.path.exists(class_idx_path):
                    with open(class_idx_path, 'r') as f:
                        class_to_idx = json.load(f)
                    classes = [None] * len(class_to_idx)
                    for cls, idx in class_to_idx.items():
                        classes[idx] = cls
                    return classes
        except:
            pass

        # 如果无法获取实际类别名称，返回通用名称
        return [f"Class_{i}" for i in range(101)]

    def test(self, test_loader=None, model_path=None):
        """
        Test the model on test data or specified data loader

        Args:
            test_loader: Optional data loader for testing. If None, use self.test_loader
            model_path: Optional path to load model weights. If None, use current model

        Returns:
            Dictionary of test metrics
        """
        # 判断数据集类型 (单标签或多标签)
        dataset_type = self.config.get("dataset", "mmimdb")
        self.is_single_label = dataset_type == "food101"
        # Use provided test loader or default
        if test_loader is None:
            if not hasattr(self, 'test_loader') or self.test_loader is None:
                self.logger.error("No test loader provided or available")
                return {}
            test_loader = self.test_loader

        # Load model if path is provided
        if model_path is not None:
            self.logger.info(f"Loading model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])

        self.model.eval()
        all_preds, all_labels = [], []
        all_logits = []  # Store raw logits for detailed analysis

        # Track metrics by missing modality type
        missing_type_metrics = {
            'none': {'preds': [], 'labels': [], 'logits': []},
            'image': {'preds': [], 'labels': [], 'logits': []},
            'text': {'preds': [], 'labels': [], 'logits': []},
            'both': {'preds': [], 'labels': [], 'logits': []}
        }

        # Track generation quality
        gen_quality = {
            'image': {'mse': [], 'count': 0},
            'text': {'mse': [], 'count': 0}
        }

        # Track quality scores
        quality_scores_by_type = {
            'none': {'image': [], 'text': [], 'consistency': []},
            'image': {'image': [], 'text': [], 'consistency': []},
            'text': {'image': [], 'text': [], 'consistency': []},
            'both': {'image': [], 'text': [], 'consistency': []}
        }

        self.logger.info("Starting test evaluation")

        with torch.no_grad():
            for batch in test_loader:
                image, input_ids, attention_mask, label, missing_type = [x.to(self.device) for x in batch]
                output = self.model(image, input_ids, attention_mask, missing_type)

                # Process output and additional info
                if isinstance(output, tuple):
                    logits, additional_info = output

                    # Process quality scores
                    if additional_info and 'quality_scores' in additional_info:
                        quality_scores = additional_info['quality_scores']
                        for b in range(missing_type.size(0)):
                            mt = missing_type[b].item()
                            mt_name = ['none', 'image', 'text', 'both'][mt]

                            quality_scores_by_type[mt_name]['image'].append(
                                quality_scores['image']['final_score'][b].item())
                            quality_scores_by_type[mt_name]['text'].append(
                                quality_scores['text']['final_score'][b].item())
                            quality_scores_by_type[mt_name]['consistency'].append(
                                quality_scores['cross_consistency'][b].item())

                    # Process reconstruction quality (no changes needed here)
                    # ...existing code for reconstruction quality...
                else:
                    logits = output

                # Handle different classification types
                if self.is_single_label:
                    # Single-label classification (Food101) - use argmax
                    pred_indices = logits.argmax(dim=1)
                    # Convert to one-hot format for consistent processing
                    preds = torch.zeros_like(logits)
                    preds.scatter_(1, pred_indices.unsqueeze(1), 1.0)
                else:
                    # Multi-label classification (MMIMDB) - use threshold
                    preds = (logits > -0.2).float()

                # Collect overall predictions and labels
                all_preds.append(preds.cpu())
                all_labels.append(label.cpu())
                all_logits.append(logits.cpu())

                # Collect predictions and labels by missing type
                for b in range(missing_type.size(0)):
                    mt = missing_type[b].item()
                    mt_name = ['none', 'image', 'text', 'both'][mt]

                    missing_type_metrics[mt_name]['preds'].append(preds[b:b + 1].cpu())
                    missing_type_metrics[mt_name]['labels'].append(label[b:b + 1].cpu())
                    missing_type_metrics[mt_name]['logits'].append(logits[b:b + 1].cpu())

        # Combine all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_logits = torch.cat(all_logits, dim=0)

        # Calculate overall metrics
        metrics = self._compute_metrics(all_preds, all_labels)

        # Print detailed statistics based on dataset type
        if self.is_single_label:
            self._detailed_statistics_food101(all_logits, all_labels, all_preds, "Test Overall")
        else:
            self._detailed_statistics(all_logits, all_labels, all_preds, "Test Overall")

        # Calculate metrics by missing type
        missing_type_results = {}
        for mt_name, data in missing_type_metrics.items():
            if data['preds'] and data['labels']:
                mt_preds = torch.cat(data['preds'], dim=0)
                mt_labels = torch.cat(data['labels'], dim=0)
                mt_logits = torch.cat(data['logits'], dim=0)
                mt_metrics = self._compute_metrics(mt_preds, mt_labels)
                missing_type_results[mt_name] = mt_metrics

                # Log missing type metrics
                self.logger.info(f"Missing type {mt_name} ({len(data['preds'])} samples):")
                metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in mt_metrics.items()])
                self.logger.info(f"  {metrics_str}")

                # Print detailed statistics by missing type
                if self.is_single_label:
                    self._detailed_statistics_food101(mt_logits, mt_labels, mt_preds, f"Test {mt_name}")
                else:
                    self._detailed_statistics(mt_logits, mt_labels, mt_preds, f"Test {mt_name}")

                # Log quality scores for this missing type
                if quality_scores_by_type[mt_name]['image']:
                    avg_img_quality = np.mean(quality_scores_by_type[mt_name]['image'])
                    avg_txt_quality = np.mean(quality_scores_by_type[mt_name]['text'])
                    avg_consistency = np.mean(quality_scores_by_type[mt_name]['consistency'])

                    self.logger.info(
                        f"  Quality - Image: {avg_img_quality:.4f} | Text: {avg_txt_quality:.4f} | Consistency: {avg_consistency:.4f}")

        # Log overall metrics
        self.logger.info("Overall test results:")
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"  {metrics_str}")

        # Return detailed metrics
        detailed_metrics = {
            'overall': metrics,
            'by_missing_type': missing_type_results,
            'quality_scores': {
                k: {
                    'image': np.mean(v['image']) if v['image'] else 0,
                    'text': np.mean(v['text']) if v['text'] else 0,
                    'consistency': np.mean(v['consistency']) if v['consistency'] else 0
                }
                for k, v in quality_scores_by_type.items() if v['image'] or v['text'] or v['consistency']
            }
        }

        return detailed_metrics
    def contrastive_loss(self,x1, x2, temperature=0.1):
        """计算对比损失，用于提高特征表示质量"""
        # 确保输入是2D的 [batch_size, feature_dim]
        if x1.dim() > 2:
            x1 = x1.view(x1.size(0), -1)  # 展平多token特征
        if x2.dim() > 2:
            x2 = x2.view(x2.size(0), -1)  # 展平多token特征

        batch_size = x1.size(0)
        x1_norm = F.normalize(x1, p=2, dim=1)
        x2_norm = F.normalize(x2, p=2, dim=1)

        # 计算相似度矩阵
        logits = torch.matmul(x1_norm, x2_norm.transpose(0, 1)) / temperature
        # 正样本是对角线元素
        labels = torch.arange(batch_size, device=x1.device)

        # 双向对比损失
        loss_x1_x2 = F.cross_entropy(logits, labels)
        loss_x2_x1 = F.cross_entropy(logits.transpose(0, 1), labels)

        return loss_x1_x2 + loss_x2_x1

    # In trainer.py

    def modality_contrastive_loss(self, image_features, text_features, temperature=0.2):
        """
        Contrastive loss between modalities with dimension handling
        """
        # Handle multi-dimensional features (like multiple tokens)
        if image_features.dim() > 2:
            # If we have [batch, tokens, dim], average across tokens
            image_features = image_features.mean(dim=1)
        if text_features.dim() > 2:
            # If we have [batch, tokens, dim], average across tokens
            text_features = text_features.mean(dim=1)

        # Get dimensions
        img_dim = image_features.size(-1)
        txt_dim = text_features.size(-1)
        batch_size = image_features.size(0)

        # Project features to same dimension if needed
        if img_dim != txt_dim:
            # Instead of trying to reshape, create simple projection layers on the fly
            if not hasattr(self,
                           'img_projection') or self.img_projection.in_features != img_dim or self.img_projection.out_features != txt_dim:
                # Create a projection for image features
                common_dim = min(img_dim, txt_dim)
                self.img_projection = nn.Linear(img_dim, common_dim).to(image_features.device)
                self.txt_projection = nn.Linear(txt_dim, common_dim).to(text_features.device)

                # Initialize the weights to average the features
                with torch.no_grad():
                    self.img_projection.weight.fill_(1.0 / img_dim)
                    self.img_projection.bias.fill_(0.0)
                    self.txt_projection.weight.fill_(1.0 / txt_dim)
                    self.txt_projection.bias.fill_(0.0)

            # Apply projections
            image_features = self.img_projection(image_features)
            text_features = self.txt_projection(text_features)

        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        # Calculate cosine similarity
        logits = torch.matmul(image_features, text_features.t()) / temperature

        # Labels are the diagonal elements (matching pairs)
        labels = torch.arange(batch_size, device=image_features.device)

        # Calculate loss in both directions
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)

        return (loss_i2t + loss_t2i) / 2.0

    def focal_loss(self, logits, targets, alpha=0.5, gamma=2.0, eps=1e-8):
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        w = alpha * targets + (1 - alpha) * (1 - targets)
        loss = -w * ((1 - pt) ** gamma) * torch.log(pt + eps)
        return loss.mean()

    def asymmetric_loss_with_logits(
            self, logits, targets, gamma_pos=0.0, gamma_neg=4.0, clip=0.05, eps=1e-8
    ):
        """Asymmetric Loss (BCE-style) for Multi-label Classification."""
        targets = targets.type_as(logits)
        anti_targets = 1 - targets

        # probabilities
        probas = torch.sigmoid(logits)
        log_probs = torch.log(probas + eps)
        log_anti = torch.log(1.0 - probas + eps)

        # asymmetric clipping of negative predictions
        if clip is not None and clip > 0:
            log_anti = torch.clamp(log_anti, min=torch.log(torch.tensor(clip)).item())

        # positive loss
        pos_loss = targets * log_probs
        neg_loss = anti_targets * log_anti

        # asymmetric focusing (focal-style modulation)
        if gamma_pos > 0:
            pos_loss *= (1 - probas) ** gamma_pos
        if gamma_neg > 0:
            neg_loss *= probas ** gamma_neg

        loss = - (pos_loss + neg_loss)
        return loss.mean()

    # Add this method to the Trainer class
    def setup_curriculum_learning(self, initial_missing_prob, final_missing_prob, ramp_epochs):
        """
        Set up curriculum learning for modality missing rate.

        Args:
            initial_missing_prob: Starting missing probability
            final_missing_prob: Final missing probability
            ramp_epochs: Number of epochs to ramp up the missing probability
        """
        self.use_curriculum = True
        self.initial_missing_prob = initial_missing_prob
        self.final_missing_prob = final_missing_prob
        self.missing_prob_ramp_epochs = ramp_epochs
        self.logger.info(
            f"Curriculum learning enabled: missing prob from {initial_missing_prob} to {final_missing_prob} over {ramp_epochs} epochs")

    # Add a simple method to update the dataset missing probability
    def update_missing_probability(self, new_prob):
        """Update missing probability in the training dataset"""
        if hasattr(self.train_loader.dataset, 'missing_prob'):
            self.train_loader.dataset.missing_prob = new_prob
            return True
        return False

