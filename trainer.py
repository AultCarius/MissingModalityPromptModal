import os
import yaml
import torch
import random
import logging
import shutil
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from transformers import get_linear_schedule_with_warmup
import torch.nn.functional as F
from scripts.emailsender import (
    setup_email_config,
    parse_log_file,
    create_training_plots,
    send_email_with_results
)
import torch.nn as nn
from tqdm import tqdm


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
        # if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        #     # 在 Kaggle 上运行，持久化目录为 /kaggle/output
        #     base_root = "/kaggle/output"
        # else:
        #     # 本地或服务器上运行
        #     base_root = "experiments"
        base_root = "experiments"
        # 基础目录
        self.base_dir = os.path.join(base_root, self.experiment_name)

        # 各子目录
        self.save_path = os.path.join(self.base_dir, "checkpoints")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.tb_dir = os.path.join(self.log_dir, "tb")
        self.code_dir = os.path.join(self.base_dir, "code_snapshot")

        # 创建所有目录
        for directory in [self.save_path, self.log_dir, self.tb_dir, self.code_dir]:
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
            "datamodules/Food101DataModule.py",
            "datamodules/MmimdbDataModule.py",
            "models/multimodal_model.py",
            "models/quality_aware_prompting.py"
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
        # self.scheduler = WarmupLinearDecayLR(
        #     self.optimizer,
        #     warmup_steps=warmup_steps,
        #     total_steps=total_steps,
        #     min_lr=self.config.get("min_lr", 0)
        # )
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

    def train(self):
        num_epochs = self.config.get("epochs", 10)
        max_epochs = num_epochs
        focal_start_epoch = self.config.get("focal_start_epoch", 3)
        use_focal_loss = self.config.get("use_focal_loss", True)
        focal_alpha = self.config.get("focal_alpha", 0.5)
        focal_max_gamma = self.config.get("focal_gamma", 2.0)
        gamma_ramp_epochs = self.config.get("gamma_ramp_epochs", 5)
        focal_weight = self.config.get("focal_weight",0.3)

        use_asymmetric_loss = self.config.get("use_asymmetric_loss", True)
        asl_start_epoch = self.config.get("asl_start_epoch", 3)
        asl_gamma_pos = self.config.get("asl_gamma_pos", 0.0)
        asl_gamma_neg = self.config.get("asl_gamma_neg", 4.0)
        asl_ramp_epochs = self.config.get("asl_ramp_epochs", 3)
        asl_clip = self.config.get("asl_clip", 0.05)

        initial_recon_weight = self.config.get("initial_recon_weight", 0.1)
        final_recon_weight = self.config.get("final_recon_weight", 0.01)

        for epoch in range(self.start_epoch, num_epochs):
            current_epoch = epoch

            # In the train method, at the beginning of each epoch (around line 256)
            # Add right after the epoch loop starts:

            # Apply curriculum learning for missing probability if enabled
            # At the beginning of each epoch in the train method
            if hasattr(self, 'use_curriculum') and self.use_curriculum:
                # Calculate current missing probability
                progress = min(1.0, epoch / self.missing_prob_ramp_epochs)
                current_missing_prob = self.initial_missing_prob + progress * (
                            self.final_missing_prob - self.initial_missing_prob)

                # Simple direct update
                if self.update_missing_probability(current_missing_prob):
                    self.logger.info(f"Epoch {epoch}: Missing probability set to {current_missing_prob:.3f}")


            self.model.train()
            total_loss = 0
            cls_loss = 0
            recon_loss = 0
            all_preds, all_labels = [], []

            # Track quality assessment and fusion weights
            quality_stats = {'image': [], 'text': [], 'consistency': []}
            fusion_weights_stats = []

            # Track modality generation performance
            gen_stats = {'image': {'mse': [], 'count': 0}, 'text': {'mse': [], 'count': 0}}

            self.logger.info(f"Epoch {epoch} start")
            batch_pbar = tqdm(total=len(self.train_loader),
                              desc=f"Epoch {epoch + 1}/{num_epochs}",
                              dynamic_ncols=True,
                              leave=False)

            for batch_idx, batch in enumerate(self.train_loader):
                image, input_ids, attention_mask, label, missing_type = [x.to(self.device) for x in batch]
                is_image_missing = (missing_type == 1) | (missing_type == 3)
                is_text_missing = (missing_type == 2) | (missing_type == 3)
                output = self.model(image, input_ids, attention_mask, missing_type)

                if isinstance(output, tuple):
                    logits, additional_info = output

                    # Calculate classification loss
                    if use_asymmetric_loss and epoch >= asl_start_epoch:
                        progress = min(1.0, (epoch - asl_start_epoch + 1) / asl_ramp_epochs)
                        gamma_pos = asl_gamma_pos * progress
                        gamma_neg = asl_gamma_neg * progress
                        classification_loss = self.asymmetric_loss_with_logits(
                            logits, label,
                            gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=asl_clip
                        )
                    elif use_focal_loss and epoch >= focal_start_epoch:
                        progress = min(1.0, (epoch - focal_start_epoch + 1) / gamma_ramp_epochs)
                        gamma = focal_max_gamma * progress
                        bce_loss = F.binary_cross_entropy_with_logits(logits, label, pos_weight=self.class_weights)
                        focal = self.focal_loss(logits, label, alpha=focal_alpha, gamma=gamma)
                        classification_loss = bce_loss + focal_weight * focal
                    else:
                        classification_loss = F.binary_cross_entropy_with_logits(
                            logits, label, pos_weight=self.class_weights
                        )

                    # Initialize reconstruction loss
                    reconstruction_loss = 0.0

                    # Calculate reconstruction loss if modality generator is used
                    if additional_info and 'reconstructed_features' in additional_info and additional_info[
                        'reconstructed_features']:
                        generated_features = additional_info['generated_features']
                        recon_features = additional_info['reconstructed_features']
                        orig_features = additional_info['original_features']

                        # Image reconstruction loss (if applicable)
                        if 'image' in recon_features and orig_features['image'] is not None:
                            if is_image_missing.any():
                                missing_mask = is_image_missing
                                missing_img_recon = recon_features['image'][missing_mask]
                                missing_img_orig = orig_features['image'][missing_mask]

                                if len(missing_img_recon) > 0:
                                    # 调整形状以适应多token情况
                                    if missing_img_recon.dim() > 2:  # [batch, token_count, dim]
                                        # 计算每个token位置的MSE损失并平均
                                        img_recon_loss = F.mse_loss(missing_img_recon, missing_img_orig)
                                    else:  # 单token情况
                                        img_recon_loss = F.mse_loss(missing_img_recon, missing_img_orig)

                                    reconstruction_loss += img_recon_loss
                                    gen_stats['image']['mse'].append(img_recon_loss.item())
                                    gen_stats['image']['count'] += len(missing_img_recon)

                        # Text reconstruction loss (if applicable)
                        if 'text' in recon_features and orig_features['text'] is not None:
                            if is_text_missing.any():
                                missing_mask = is_text_missing
                                missing_txt_recon = recon_features['text'][missing_mask]
                                missing_txt_orig = orig_features['text'][missing_mask]

                                if len(missing_txt_recon) > 0:
                                    # 调整形状以适应多token情况
                                    if missing_txt_recon.dim() > 2:  # [batch, token_count, dim]
                                        # 计算每个token位置的MSE损失并平均
                                        txt_recon_loss = F.mse_loss(missing_txt_recon, missing_txt_orig)
                                    else:  # 单token情况
                                        txt_recon_loss = F.mse_loss(missing_txt_recon, missing_txt_orig)

                                    reconstruction_loss += txt_recon_loss
                                    gen_stats['text']['mse'].append(txt_recon_loss.item())
                                    gen_stats['text']['count'] += len(missing_txt_recon)

                    # Total loss with weighted reconstruction loss
                    recon_weight = initial_recon_weight * (1 - current_epoch / max_epochs) + final_recon_weight * (
                                current_epoch / max_epochs)
                    total_batch_loss = classification_loss + recon_weight * reconstruction_loss

                    # After calculating the classification loss
                    contrastive_weight = 0.1  # Start with a small weight

                    # Calculate contrastive loss between real modalities (only for samples with both modalities)
                    if 'quality_scores' in additional_info:
                        # Get samples with both modalities (missing_type == 0)
                        both_modalities = (missing_type == 0)
                        if both_modalities.any():
                            # Get features for samples with both modalities
                            if 'original_features' in additional_info:
                                orig_img_feat = additional_info['original_features']['image']
                                orig_txt_feat = additional_info['original_features']['text']

                                if orig_img_feat is not None and orig_txt_feat is not None:
                                    # Extract features only for samples with both modalities
                                    img_feat_both = orig_img_feat[both_modalities]
                                    txt_feat_both = orig_txt_feat[both_modalities]

                                    if len(img_feat_both) > 1:  # Need at least 2 samples for contrastive loss
                                        # Calculate contrastive loss
                                        contra_loss = self.modality_contrastive_loss(img_feat_both, txt_feat_both)
                                        # Add to total loss
                                        total_batch_loss = total_batch_loss + contrastive_weight * contra_loss
                                        # Track separately for logging
                                        contra_loss_value = contra_loss.item()
                                    else:
                                        contra_loss_value = 0.0
                                else:
                                    contra_loss_value = 0.0
                            else:
                                contra_loss_value = 0.0
                        else:
                            contra_loss_value = 0.0
                    else:
                        contra_loss_value = 0.0

                    # Quality prediction loss (when both modalities are available)
                    quality_pred_loss = 0.0
                    if both_modalities.any() and 'quality_scores' in additional_info:
                        # Extract features for samples with both real modalities
                        real_img_feat = orig_img_feat[both_modalities]
                        real_txt_feat = orig_txt_feat[both_modalities]

                        # For real samples, create pseudo ground truth quality scores
                        # We assume real samples have high quality (0.9) and high consistency (0.9)
                        target_img_quality = torch.full((len(real_img_feat), 1), 0.9, device=self.device)
                        target_txt_quality = torch.full((len(real_txt_feat), 1), 0.9, device=self.device)
                        target_consistency = torch.full((len(real_img_feat), 1), 0.9, device=self.device)

                        # Get actual quality predictions
                        real_quality_scores = {
                            'image': {'final_score': additional_info['quality_scores']['image']['final_score'][
                                both_modalities]},
                            'text': {'final_score': additional_info['quality_scores']['text']['final_score'][
                                both_modalities]},
                            'cross_consistency': additional_info['quality_scores']['cross_consistency'][both_modalities]
                        }

                        # Calculate quality prediction loss (MSE)
                        img_quality_loss = F.mse_loss(real_quality_scores['image']['final_score'], target_img_quality)
                        txt_quality_loss = F.mse_loss(real_quality_scores['text']['final_score'], target_txt_quality)
                        consistency_loss = F.mse_loss(real_quality_scores['cross_consistency'], target_consistency)

                        # Combined quality prediction loss
                        quality_pred_loss = img_quality_loss + txt_quality_loss + consistency_loss

                        # Add to total loss with small weight
                        quality_weight = 0.05
                        total_batch_loss = total_batch_loss + quality_weight * quality_pred_loss

                    if 'quality_scores' in additional_info:
                        # Get samples where only one modality is missing
                        only_img_missing = is_image_missing & ~is_text_missing
                        only_txt_missing = is_text_missing & ~is_image_missing

                        if only_img_missing.any() or only_txt_missing.any():
                            # Target: real features should get high scores, generated low scores
                            adv_targets = torch.zeros(self.batch_size, 2, device=self.device)  # [image_real, text_real]
                            adv_targets[~is_image_missing, 0] = 1.0  # Real image = 1
                            adv_targets[~is_text_missing, 1] = 1.0  # Real text = 1

                            # Actual quality scores
                            pred_quality = torch.zeros(self.batch_size, 2, device=self.device)
                            pred_quality[:, 0] = additional_info['quality_scores']['image']['final_score'].squeeze()
                            pred_quality[:, 1] = additional_info['quality_scores']['text']['final_score'].squeeze()

                            # Adversarial loss (binary cross entropy)
                            adv_loss = F.binary_cross_entropy(pred_quality, adv_targets)

                            # Add to total loss
                            adv_weight = 0.1
                            total_batch_loss = total_batch_loss + adv_weight * adv_loss

                    # Collect quality assessment data
                    if additional_info and 'quality_scores' in additional_info:
                        quality_scores = additional_info['quality_scores']
                        quality_stats['image'].append(quality_scores['image']['final_score'].mean().item())
                        quality_stats['text'].append(quality_scores['text']['final_score'].mean().item())
                        quality_stats['consistency'].append(quality_scores['cross_consistency'].mean().item())

                    # Collect fusion weights data
                    if additional_info and 'fusion_weights' in additional_info:
                        fusion_weights = additional_info['fusion_weights']
                        if fusion_weights is not None:
                            fusion_weights_stats.append(fusion_weights.mean(dim=0).cpu().detach().numpy())

                    # Track component losses
                    cls_loss += classification_loss.item()
                    recon_loss += reconstruction_loss.item() if isinstance(reconstruction_loss,
                                                                           torch.Tensor) else reconstruction_loss
                else:
                    logits = output
                    classification_loss = F.binary_cross_entropy_with_logits(
                        logits, label, pos_weight=self.class_weights
                    )
                    total_batch_loss = classification_loss
                    cls_loss += classification_loss.item()

                # Optimization step
                self.optimizer.zero_grad()
                total_batch_loss.backward()

                batch_pbar.set_postfix({"loss": f"{total_batch_loss.item():.4f}"})
                batch_pbar.update(1)

                # Optional gradient clipping
                if self.config.get("clip_grad_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("clip_grad_norm", 1.0)
                    )

                self.optimizer.step()

                # Update learning rate
                self.scheduler.step()

                # Record current learning rate
                current_lr = self.scheduler.get_last_lr()[0]
                if self.writer and batch_idx % 50 == 0:
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar("lr", current_lr, global_step)

                total_loss += total_batch_loss.item()
                preds = (logits > 0.5).float()

                # Collect predictions and true labels for metrics
                all_preds.append(preds.cpu().detach())
                all_labels.append(label.cpu().detach())

            batch_pbar.close()

            # Merge all batch predictions and labels
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # Calculate metrics on training set
            train_metrics = self._compute_metrics(all_preds, all_labels)

            # Record training loss and metrics
            avg_loss = total_loss / len(self.train_loader)
            avg_cls_loss = cls_loss / len(self.train_loader)
            avg_recon_loss = recon_loss / len(self.train_loader)

            metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
            self.logger.info(
                f"Epoch {epoch}: Train Loss={avg_loss:.4f} | Cls Loss={avg_cls_loss:.4f} | Recon Loss={avg_recon_loss:.4f} | {metrics_str}")
            # Track contrastive loss
            contra_loss_sum = 0.0  # Initialize at the beginning of the epoch
            contra_loss_sum += contra_loss_value  # Add in the batch loop

            # In the logging section after the epoch
            avg_contra_loss = contra_loss_sum / len(self.train_loader)
            self.logger.info(
                f"Epoch {epoch}: Train Loss={avg_loss:.4f} | Cls Loss={avg_cls_loss:.4f} | Recon Loss={avg_recon_loss:.4f} | Contra Loss={avg_contra_loss:.4f} | {metrics_str}")

            # Update tensorboard logging
            if self.writer:
                self.writer.add_scalar("Loss/train_contra", avg_contra_loss, epoch)
            # Log quality assessment and fusion weight statistics
            if quality_stats['image'] and self.writer:
                self.writer.add_scalar("Quality/image", np.mean(quality_stats['image']), epoch)
                self.writer.add_scalar("Quality/text", np.mean(quality_stats['text']), epoch)
                self.writer.add_scalar("Quality/consistency", np.mean(quality_stats['consistency']), epoch)

            if fusion_weights_stats and self.writer:
                avg_weights = np.mean(fusion_weights_stats, axis=0)
                for i, w in enumerate(avg_weights):
                    self.writer.add_scalar(f"Fusion/weight_{i}", w, epoch)

            # Log modality generation statistics
            if gen_stats['image']['mse'] and self.writer:
                self.writer.add_scalar("Generation/image_mse", np.mean(gen_stats['image']['mse']), epoch)
                self.writer.add_scalar("Generation/image_count", gen_stats['image']['count'], epoch)

            if gen_stats['text']['mse'] and self.writer:
                self.writer.add_scalar("Generation/text_mse", np.mean(gen_stats['text']['mse']), epoch)
                self.writer.add_scalar("Generation/text_count", gen_stats['text']['count'], epoch)

            if self.writer:
                self.writer.add_scalar("Loss/train_total", avg_loss, epoch)
                self.writer.add_scalar("Loss/train_cls", avg_cls_loss, epoch)
                self.writer.add_scalar("Loss/train_recon", avg_recon_loss, epoch)
                for k, v in train_metrics.items():
                    self.writer.add_scalar(f"{k}/train", v, epoch)

            # Evaluate on validation set
            val_metrics = self.evaluate(epoch)

            # Save checkpoint every few epochs
            if epoch % self.config.get("save_every_epochs", 5) == 0:
                self._save_checkpoint(epoch, metrics=val_metrics)

            # Save best model based on primary metric
            is_best = False
            if val_metrics[self.primary_metric] > self.best_metrics[self.primary_metric]:
                self.best_metrics = val_metrics.copy()
                is_best = True
                self._save_checkpoint(epoch, metrics=val_metrics, is_best=True)
                self.logger.info(
                    f"New best model saved with {self.primary_metric} = {val_metrics[self.primary_metric]:.4f}")

            with torch.no_grad():
                mean_logit = logits.mean().item()
                std_logit = logits.std().item()
                mean_pred = (torch.sigmoid(logits) > 0.5).float().sum(dim=1).mean().item()
                sigmoid_probs = torch.sigmoid(logits)
                predictions_per_sample = (sigmoid_probs > 0.5).float().sum(dim=1).mean().item()

            if self.writer:
                self.writer.add_scalar("Debug/mean_logit", mean_logit, epoch)
                self.writer.add_scalar("Debug/std_logit", std_logit, epoch)
                self.writer.add_scalar("Debug/predictions_per_sample", mean_pred, epoch)

            if self.writer:
                self.writer.add_scalar("Debug/predictions_per_sample", predictions_per_sample, epoch)
                self.writer.add_scalar("Debug/mean_logit", mean_logit, epoch)
                self.writer.add_scalar("Debug/std_logit", std_logit, epoch)
                if use_asymmetric_loss and epoch >= asl_start_epoch:
                    self.writer.add_scalar("Debug/asl_gamma_neg", gamma_neg, epoch)
                    self.writer.add_scalar("Debug/asl_gamma_pos", gamma_pos, epoch)

        # Training completed
        self.logger.info(
            f"Training completed. Best {self.primary_metric}: {self.best_metrics[self.primary_metric]:.4f}")

        # Print all best metrics
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in self.best_metrics.items()])
        self.logger.info(f"Best metrics: {metrics_str}")

        self.send_training_results_email()

        if self.writer:
            self.writer.close()

    def _compute_metrics(self, preds, labels):
        """计算多种评估指标"""
        results = {}

        # 将张量转换为NumPy数组
        preds_np = preds.numpy()
        labels_np = labels.numpy()

        # 计算要求的指标
        if "accuracy" in self.metrics:
            # 多标签情况下的准确率计算
            correct = ((preds_np > 0.5) == (labels_np > 0.5)).sum()
            total = labels_np.size
            results["accuracy"] = correct / total

        if "macro_f1" in self.metrics:
            # 多标签宏平均F1
            # 对每个样本的每个类别计算是否预测正确
            binary_preds = (preds_np > 0.5).astype(float)
            try:
                results["macro_f1"] = f1_score(labels_np, binary_preds, average='macro', zero_division=0)
            except:
                results["macro_f1"] = 0.0

        if "micro_f1" in self.metrics:
            # 多标签微平均F1
            binary_preds = (preds_np > 0.5).astype(float)
            try:
                results["micro_f1"] = f1_score(labels_np, binary_preds, average='micro', zero_division=0)
            except:
                results["micro_f1"] = 0.0

        if "auroc" in self.metrics:
            # 多标签AUROC
            try:
                # 当有类别全为正或全为负时，这会失败
                results["auroc"] = roc_auc_score(labels_np, preds_np, average='macro')
            except:
                results["auroc"] = 0.5  # 随机猜测的AUC值

        return results

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

                    # 收集融合权重
                    if additional_info and 'fusion_weights' in additional_info and additional_info[
                        'fusion_weights'] is not None:
                        quality_data['fusion_weights'].append(additional_info['fusion_weights'].cpu())
                else:
                    logits = output

                preds = (logits > 0.5).float()

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

        # 计算总体指标
        metrics = self._compute_metrics(all_preds, all_labels)

        # 详细统计信息
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

                # 为每种缺失类型记录详细统计信息
                self._detailed_statistics(mt_logits, mt_labels, mt_preds, f"Missing: {mt_name}")

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

                    self.logger.info(f"  - {mt_name} missing type ({mask.sum().item()} samples):")
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

    def test(self, test_loader=None, model_path=None):
        """
        Test the model on test data or specified data loader

        Args:
            test_loader: Optional data loader for testing. If None, use self.test_loader
            model_path: Optional path to load model weights. If None, use current model

        Returns:
            Dictionary of test metrics
        """
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

        # Track metrics by missing modality type
        missing_type_metrics = {
            'none': {'preds': [], 'labels': []},
            'image': {'preds': [], 'labels': []},
            'text': {'preds': [], 'labels': []},
            'both': {'preds': [], 'labels': []}
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

                    # Process reconstruction quality
                    if additional_info and 'reconstructed_features' in additional_info:
                        recon_features = additional_info['reconstructed_features']
                        orig_features = additional_info['original_features']

                        # Evaluate image reconstruction (for missing image samples)
                        if 'image' in recon_features and recon_features['image'] is not None:
                            for b in range(missing_type.size(0)):
                                if missing_type[b].item() == 1:  # Image missing
                                    if orig_features['image'] is not None:
                                        # This should not happen as image is missing, but check just in case
                                        continue
                                    # Skip if we can't evaluate reconstruction quality
                                    # (we need the ground truth to compare)
                                    continue

                        # Evaluate text reconstruction (for missing text samples)
                        if 'text' in recon_features and recon_features['text'] is not None:
                            for b in range(missing_type.size(0)):
                                if missing_type[b].item() == 2:  # Text missing
                                    if orig_features['text'] is not None:
                                        # This should not happen as text is missing, but check just in case
                                        continue
                                    # Skip if we can't evaluate reconstruction quality
                                    continue
                else:
                    logits = output

                preds = (logits > 0.5).float()

                # Collect overall predictions and labels
                all_preds.append(preds.cpu())
                all_labels.append(label.cpu())

                # Collect predictions and labels by missing type
                for b in range(missing_type.size(0)):
                    mt = missing_type[b].item()
                    mt_name = ['none', 'image', 'text', 'both'][mt]

                    missing_type_metrics[mt_name]['preds'].append(preds[b:b + 1].cpu())
                    missing_type_metrics[mt_name]['labels'].append(label[b:b + 1].cpu())

        # Combine all predictions and labels
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Calculate overall metrics
        metrics = self._compute_metrics(all_preds, all_labels)

        # Calculate metrics by missing type
        missing_type_results = {}
        for mt_name, data in missing_type_metrics.items():
            if data['preds'] and data['labels']:
                mt_preds = torch.cat(data['preds'], dim=0)
                mt_labels = torch.cat(data['labels'], dim=0)
                mt_metrics = self._compute_metrics(mt_preds, mt_labels)
                missing_type_results[mt_name] = mt_metrics

                # Log missing type metrics
                self.logger.info(f"Missing type {mt_name} ({len(data['preds'])} samples):")
                metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in mt_metrics.items()])
                self.logger.info(f"  {metrics_str}")

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

    def modality_contrastive_loss(self, image_features, text_features, temperature=0.1):
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

