import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


def parse_log_file(log_path):
    """解析日志文件，提取各Epoch的Macro-F1指标"""
    with open(log_path, 'r', encoding='gb2312') as f:
        log_content = f.read()

    # 按Epoch分割日志内容
    epoch_pattern = re.compile(
        r'Quality and performance by missing type:\s*'
        r'\[.*?\]\s*-\s*none is missing type\s*\((\d+)\s*samples\):\s*'
        r'\[.*?\]\s*Quality\s*-\s*Image:\s*([\d\.]+)\s*\|\s*Text:\s*([\d\.]+)\s*\|\s*Consistency:\s*([\d\.]+)\s*'
        r'\[.*?\]\s*Performance\s*-\s*accuracy=([\d\.]+)\s*\|\s*macro_f1=([\d\.]+)\s*\|\s*micro_f1=([\d\.]+)\s*'
        r'\[.*?\]\s*-\s*image is missing type\s*\((\d+)\s*samples\):\s*'
        r'\[.*?\]\s*Quality\s*-\s*Image:\s*([\d\.]+)\s*\|\s*Text:\s*([\d\.]+)\s*\|\s*Consistency:\s*([\d\.]+)\s*'
        r'\[.*?\]\s*Performance\s*-\s*accuracy=([\d\.]+)\s*\|\s*macro_f1=([\d\.]+)\s*\|\s*micro_f1=([\d\.]+)\s*'
        r'\[.*?\]\s*-\s*text is missing type\s*\((\d+)\s*samples\):\s*'
        r'\[.*?\]\s*Quality\s*-\s*Image:\s*([\d\.]+)\s*\|\s*Text:\s*([\d\.]+)\s*\|\s*Consistency:\s*([\d\.]+)\s*'
        r'\[.*?\]\s*Performance\s*-\s*accuracy=([\d\.]+)\s*\|\s*macro_f1=([\d\.]+)\s*\|\s*micro_f1=([\d\.]+)\s*'
        r'\[.*?\]\s*-\s*none:\s*(\d+)\s*samples\s*'
        r'\[.*?\]\s*accuracy=([\d\.]+)\s*\|\s*macro_f1=([\d\.]+)\s*\|\s*micro_f1=([\d\.]+)\s*'
        r'\[.*?\]\s*-\s*image:\s*(\d+)\s*samples\s*'
        r'\[.*?\]\s*accuracy=([\d\.]+)\s*\|\s*macro_f1=([\d\.]+)\s*\|\s*micro_f1=([\d\.]+)\s*'
        r'\[.*?\]\s*-\s*text:\s*(\d+)\s*samples\s*'
        r'\[.*?\]\s*accuracy=([\d\.]+)\s*\|\s*macro_f1=([\d\.]+)\s*\|\s*micro_f1=([\d\.]+)\s*'
        r'\[.*?\]\s*\[Val\]\s*Epoch\s*(\d+):\s*accuracy=([\d\.]+)\s*\|\s*macro_f1=([\d\.]+)\s*\|\s*micro_f1=([\d\.]+)'
    )

    epochs_data = []
    for match in epoch_pattern.finditer(log_content):
        groups = match.groups()
        epoch_data = {
            'epoch': int(groups[-4]),  # 总Epoch数
            'metrics': {
                'none': float(groups[5]),  # none类型的macro_f1
                'image': float(groups[12]),  # image类型的macro_f1
                'text': float(groups[19]),  # text类型的macro_f1
                'total': float(groups[-2])  # 总体macro_f1
            }
        }
        epochs_data.append(epoch_data)

    return epochs_data


def plot_macro_f1(epochs_data, output_path='macro_f1_trend.png'):
    """绘制Macro-F1指标随Epoch的变化趋势图"""
    if not epochs_data:
        print("没有找到匹配的Epoch数据！")
        return

    # 提取数据
    epochs = [data['epoch'] for data in epochs_data]
    none_f1 = [data['metrics']['none'] for data in epochs_data]
    image_f1 = [data['metrics']['image'] for data in epochs_data]
    text_f1 = [data['metrics']['text'] for data in epochs_data]
    total_f1 = [data['metrics']['total'] for data in epochs_data]

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制各类型的Macro-F1曲线
    plt.plot(epochs, none_f1, 'o-', color='blue', label='None缺失')
    plt.plot(epochs, image_f1, 's-', color='red', label='图像缺失')
    plt.plot(epochs, text_f1, 'd-', color='green', label='文本缺失')
    plt.plot(epochs, total_f1, '^-', color='purple', label='总体')

    # 添加标题和标签
    plt.title('不同缺失类型的Macro-F1指标随Epoch变化趋势', fontsize=16)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Macro-F1分数', fontsize=14)

    # 设置x轴为整数
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    # 添加网格和图例
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)

    # 保存图表
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"图表已保存至: {output_path}")

    # 显示图表
    plt.show()


if __name__ == "__main__":
    log_file_path = 'E:\DL\MissingModalityPromptModal\experiments\mmimdb_ceshi\logs\\train_20250520_064117.log'  # 请替换为实际的日志文件路径
    output_image_path = 'macro_f1_trend_ceshi.png'  # 输出图像路径

    # 解析日志
    epochs_data = parse_log_file(log_file_path)

    # 绘制并保存图表
    plot_macro_f1(epochs_data, output_image_path)