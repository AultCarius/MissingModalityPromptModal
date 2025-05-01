import os
import smtplib
import matplotlib.pyplot as plt
import numpy as np
import re
import json
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication


def setup_email_config(config):
    """
    Add email configuration to trainer config

    Args:
        config (dict): Trainer configuration dictionary

    Returns:
        dict: Updated configuration with email settings
    """
    # Default email configuration for QQ email
    email_config = {
        "send_email": True,
        "smtp_server": "smtp.qq.com",
        "smtp_port": 465,  # QQ SMTP port for SSL
        "sender_email": config.get("sender_email", ""),
        "sender_password": config.get("sender_password", ""),
        "recipient_email": config.get("recipient_email", ""),
        "email_subject": f"Training Results: {config.get('experiment_name', 'Experiment')}"
    }

    # Update config with email settings
    if "email" not in config:
        config["email"] = email_config
    else:
        # Update with any missing default values
        for key, value in email_config.items():
            if key not in config["email"]:
                config["email"][key] = value

    return config


def parse_log_file(log_path):
    """
    Parse training log file to extract epoch metrics

    Args:
        log_path (str): Path to log file

    Returns:
        dict: Dictionary of training and validation metrics by epoch
    """
    if not os.path.exists(log_path):
        return None

    with open(log_path, 'r') as f:
        log_content = f.read()

    # Extract training metrics
    train_pattern = r'\[([^\]]+)\] Epoch (\d+): Train Loss=([\d\.]+) \| Cls Loss=([\d\.]+) \| Recon Loss=([\d\.]+) \| (.+)'
    train_matches = re.findall(train_pattern, log_content)

    # Extract validation metrics
    val_pattern = r'\[([^\]]+)\] \[Val\] Epoch (\d+): (.+)'
    val_matches = re.findall(val_pattern, log_content)

    # Organize metrics by epoch
    metrics = {
        'train': {},
        'val': {}
    }

    # Process training metrics
    for match in train_matches:
        timestamp, epoch, train_loss, cls_loss, recon_loss, other_metrics = match
        epoch = int(epoch)

        if epoch not in metrics['train']:
            metrics['train'][epoch] = {}

        metrics['train'][epoch]['train_loss'] = float(train_loss)
        metrics['train'][epoch]['cls_loss'] = float(cls_loss)
        metrics['train'][epoch]['recon_loss'] = float(recon_loss)

        # Extract other metrics
        other_metrics_dict = {}
        for metric_pair in other_metrics.split(' | '):
            key, value = metric_pair.split('=')
            other_metrics_dict[key] = float(value)

        metrics['train'][epoch].update(other_metrics_dict)

    # Process validation metrics
    for match in val_matches:
        timestamp, epoch, metrics_str = match
        epoch = int(epoch)

        if epoch not in metrics['val']:
            metrics['val'][epoch] = {}

        # Extract validation metrics
        for metric_pair in metrics_str.split(' | '):
            key, value = metric_pair.split('=')
            metrics['val'][epoch][key] = float(value)

    return metrics


def create_training_plots(metrics, output_dir):
    """
    Create plots from training metrics and save them to output directory

    Args:
        metrics (dict): Dictionary of training and validation metrics by epoch
        output_dir (str): Directory to save plots

    Returns:
        list: List of paths to saved plots
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    if not metrics:
        return plot_paths

    # Extract epochs
    epochs = sorted(metrics['train'].keys())

    # Plot 1: Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.title('Training and Validation Loss')

    # Training loss
    train_loss = [metrics['train'][e].get('train_loss', np.nan) for e in epochs]
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')

    # Validation loss (if available)
    if 'val' in metrics and metrics['val']:
        val_epochs = sorted(metrics['val'].keys())
        if 'accuracy' in metrics['val'][val_epochs[0]]:
            val_acc = [metrics['val'][e].get('accuracy', np.nan) for e in val_epochs]
            plt.plot(val_epochs, val_acc, 'r-', label='Validation Accuracy')

    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.grid(True)

    # Save plot
    loss_plot_path = os.path.join(output_dir, 'loss_plot.png')
    plt.savefig(loss_plot_path)
    plt.close()
    plot_paths.append(loss_plot_path)

    # Plot 2: Training Components (Classification and Reconstruction Loss)
    plt.figure(figsize=(10, 6))
    plt.title('Training Loss Components')

    # Classification loss
    cls_loss = [metrics['train'][e].get('cls_loss', np.nan) for e in epochs]
    plt.plot(epochs, cls_loss, 'g-', label='Classification Loss')

    # Reconstruction loss
    recon_loss = [metrics['train'][e].get('recon_loss', np.nan) for e in epochs]
    plt.plot(epochs, recon_loss, 'm-', label='Reconstruction Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Save plot
    components_plot_path = os.path.join(output_dir, 'loss_components_plot.png')
    plt.savefig(components_plot_path)
    plt.close()
    plot_paths.append(components_plot_path)

    # Plot 3: Performance Metrics
    metrics_to_plot = ['macro_f1', 'micro_f1', 'accuracy', 'auroc']
    available_metrics = []

    # Check which metrics are available in validation data
    if 'val' in metrics and metrics['val']:
        val_epochs = sorted(metrics['val'].keys())
        if val_epochs:
            first_epoch = val_epochs[0]
            available_metrics = [m for m in metrics_to_plot if m in metrics['val'][first_epoch]]

    if available_metrics:
        plt.figure(figsize=(10, 6))
        plt.title('Validation Performance Metrics')

        for metric in available_metrics:
            metric_values = [metrics['val'][e].get(metric, np.nan) for e in val_epochs]
            plt.plot(val_epochs, metric_values, label=metric)

        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True)

        # Save plot
        metrics_plot_path = os.path.join(output_dir, 'metrics_plot.png')
        plt.savefig(metrics_plot_path)
        plt.close()
        plot_paths.append(metrics_plot_path)

    return plot_paths


def send_email_with_results(config, log_paths, plot_paths, best_metrics):
    """
    Send email with training results, logs, and plots

    Args:
        config (dict): Configuration with email settings
        log_paths (list): Paths to log files
        plot_paths (list): Paths to plot images
        best_metrics (dict): Best validation metrics

    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    email_config = config.get("email", {})

    if not email_config.get("send_email", False):
        return False

    # Check required email fields
    required_fields = ["smtp_server", "smtp_port", "sender_email",
                       "sender_password", "recipient_email"]
    for field in required_fields:
        if not email_config.get(field):
            print(f"Warning: Email not sent. Missing required field: {field}")
            return False

    # Create message
    msg = MIMEMultipart()
    msg['From'] = email_config["sender_email"]
    msg['To'] = email_config["recipient_email"]
    msg['Subject'] = email_config.get("email_subject", "Training Results")

    # Prepare email body with HTML formatting
    body = f"""
    <html>
    <body>
    <h2>Training Results: {config.get('experiment_name', 'Experiment')}</h2>

    <h3>Best Validation Results:</h3>
    <ul>
    """

    # Add best metrics
    for metric, value in best_metrics.items():
        body += f"<li>{metric}: {value:.4f}</li>\n"

    body += "</ul>"

    # Add experiment details
    body += f"""
    <h3>Experiment Details:</h3>
    <ul>
        <li>Experiment Name: {config.get('experiment_name', 'Not specified')}</li>
        <li>Learning Rate: {config.get('lr', 'Not specified')}</li>
        <li>Epochs: {config.get('epochs', 'Not specified')}</li>
    </ul>
    """

    body += "</body></html>"

    # Attach text part to email
    msg.attach(MIMEText(body, 'html'))

    # Attach log files
    for log_path in log_paths:
        if os.path.exists(log_path):
            with open(log_path, 'rb') as f:
                log_attachment = MIMEApplication(f.read(), _subtype="txt")
                log_attachment.add_header('Content-Disposition', 'attachment',
                                          filename=os.path.basename(log_path))
                msg.attach(log_attachment)

    # Attach plot images
    for plot_path in plot_paths:
        if os.path.exists(plot_path):
            with open(plot_path, 'rb') as f:
                plot_attachment = MIMEApplication(f.read(), _subtype="png")
                plot_attachment.add_header('Content-Disposition', 'attachment',
                                           filename=os.path.basename(plot_path))
                msg.attach(plot_attachment)

    # Try to send email
    try:
        server = smtplib.SMTP(email_config["smtp_server"], email_config["smtp_port"])
        server.starttls()
        server.login(email_config["sender_email"], email_config["sender_password"])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False