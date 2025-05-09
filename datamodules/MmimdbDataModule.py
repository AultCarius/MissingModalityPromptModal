import os
import json
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from .baseDataModule import BaseDataModule

from PIL import Image, ImageFile
import warnings

# 👇 忽略"大图像"警告
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
# 👇 允许无限大图片
Image.MAX_IMAGE_PIXELS = None
# 👇 忽略截断图像报错（防止损坏图像导致崩溃）
ImageFile.LOAD_TRUNCATED_IMAGES = True

GENRE_CLASS = [
    'Drama', 'Comedy', 'Romance', 'Thriller', 'Crime', 'Action', 'Adventure',
    'Horror', 'Documentary', 'Mystery', 'Sci-Fi', 'Fantasy', 'Family',
    'Biography', 'War', 'History', 'Music', 'Animation', 'Musical', 'Western',
    'Sport', 'Short', 'Film-Noir'
]
GENRE_CLASS_DICT = {genre: idx for idx, genre in enumerate(GENRE_CLASS)}


class MMIMDBDataset(Dataset):
    def __init__(self, sample_list, data_dir, image_transform=None, tokenizer=None,
                 max_length=128, missing_strategy="none"):
        self.sample_list = sample_list
        self.data_dir = data_dir
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])
        # # Replace CLIP tokenizer with RoBERTa tokenizer
        # from transformers import RobertaTokenizer
        # self.tokenizer = tokenizer or RobertaTokenizer.from_pretrained("roberta-base")

        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

        self.max_length = max_length
        self.missing_strategy = missing_strategy
        self.missing_prob = 0.7

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample_id = self.sample_list[idx]
        img_path = os.path.join(self.data_dir, "dataset", f"{sample_id}.jpeg")
        json_path = os.path.join(self.data_dir, "dataset", f"{sample_id}.json")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        # 1. 标签向量
        label = torch.zeros(len(GENRE_CLASS))
        for genre in metadata["genres"]:
            if genre in GENRE_CLASS_DICT:
                label[GENRE_CLASS_DICT[genre]] = 1.0

        # 2. 文本处理
        plot = metadata["plot"][0] if isinstance(metadata["plot"], list) else metadata["plot"]
        encoded = self.tokenizer(plot, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        # print("input_ids:",encoded["input_ids"].shape,"attention_mask",encoded["input_ids"].shape)
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # 3. 图像处理
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        # 4. 模态缺失模拟 - 存储原始数据并标记缺失
        missing_type = self._simulate_missing()

        # # 保存原始数据
        # original_image = image.clone()
        # original_input_ids = input_ids.clone()
        # original_attention_mask = attention_mask.clone()

        # 标记缺失模态（但不清零原始数据）
        # is_image_missing = missing_type in ["image", "both"]
        # is_text_missing = missing_type in ["text", "both"]

        missing_type_tensor = torch.tensor({
                                               "none": 0, "image": 1, "text": 2, "both": 3
                                           }[missing_type])

        return (
            image, input_ids, attention_mask,  # 可见模态数据
            label, missing_type_tensor
        )

    def _simulate_missing(self):
        """模拟模态缺失情况

        根据设置的缺失率η，模拟不同类型的模态缺失:
        - "none": 不模拟缺失
        - "both": 双模态缺失 (η/2图像缺失，η/2文本缺失)
        - "image": 仅图像缺失 (η图像缺失)
        - "text": 仅文本缺失 (η文本缺失)

        Returns:
            str: 缺失类型，可能为 "none", "image", "text"
        """
        if self.missing_strategy == "none":
            return "none"

        # 使用实例变量的缺失率
        eta = self.missing_prob

        # 如果是float类型，使用指定的缺失率
        if isinstance(self.missing_strategy, float):
            eta = self.missing_strategy

        # 根据缺失策略执行不同的缺失模式
        if self.missing_strategy == "both":
            # 按照设计：η/2的概率图像缺失，η/2的概率文本缺失，1-η的概率完整
            return random.choices(
                ["none", "image", "text"],
                weights=[1 - eta, eta / 2, eta / 2],
                k=1
            )[0]
        elif self.missing_strategy == "image":
            # 图像缺失：η的概率图像缺失，1-η的概率完整
            return random.choices(
                ["none", "image"],
                weights=[1 - eta, eta],
                k=1
            )[0]
        elif self.missing_strategy == "text":
            # 文本缺失：η的概率文本缺失，1-η的概率完整
            return random.choices(
                ["none", "text"],
                weights=[1 - eta, eta],
                k=1
            )[0]

        # 默认不缺失
        return "none"

    # Add this method to the MMIMDBDataset class
    def update_missing_prob(self, new_prob):
        """Update the missing probability for this dataset"""
        self.missing_prob = new_prob
        return self

class mmimdbDataModule(BaseDataModule):
    def __init__(self, data_dir="./data/mmimdb", batch_size=32, num_workers=4,
                 image_transform=None, tokenizer=None, max_length=128,
                 missing_strategy="none", missing_prob=0.7,  # 新增missing_prob
                 val_missing_strategy="none", val_missing_prob=0.0,  # 验证集也支持
                 test_missing_strategy="none", test_missing_prob=0.0,  # 新增测试集配置
                 image_size=224, patch_size=16,seed=42):
        super().__init__(batch_size, num_workers)

        self.seed = seed

        self.data_dir = data_dir
        self.image_size = image_size
        self.patch_size = patch_size

        if self.image_size % self.patch_size != 0:
            self.image_size = (self.image_size // self.patch_size) * self.patch_size
            print(f"调整图像尺寸为 patch_size 的整数倍: {self.image_size}")

        self.image_transform = image_transform or self.default_transform(self.image_size)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.missing_strategy = missing_strategy
        self.missing_prob = missing_prob
        self.val_missing_strategy = val_missing_strategy
        self.val_missing_prob = val_missing_prob
        self.test_missing_strategy = test_missing_strategy  # 新增的测试集缺失策略
        self.test_missing_prob = test_missing_prob  # 新增的测试集缺失率

    def setup(self, stage=None):
        split_path = os.path.join(self.data_dir, "split.json")
        with open(split_path, "r") as f:
            split_data = json.load(f)

        random.seed(self.seed)
        random.shuffle(split_data["train"])
        random.shuffle(split_data["dev"])
        random.shuffle(split_data["test"])

        # 训练集缺失策略
        train_missing = self._parse_missing_strategy(self.missing_strategy, self.missing_prob)

        self.train_dataset = MMIMDBDataset(
            split_data["train"], self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=train_missing
        )

        # 验证集缺失策略
        val_missing = self._parse_missing_strategy(self.val_missing_strategy, self.val_missing_prob)

        self.val_dataset = MMIMDBDataset(
            split_data["dev"], self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=val_missing
        )
        # 测试集缺失策略
        test_missing = self._parse_missing_strategy(self.test_missing_strategy, self.test_missing_prob)

        self.test_dataset = MMIMDBDataset(
            split_data["test"], self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=test_missing
        )


    def _parse_missing_strategy(self, strategy, prob):
        """解析缺失策略。"""
        if strategy == "random":
            return prob  # random就返回缺失率
        elif strategy in ["both", "image", "text"]:
            return strategy
        else:
            return "none"

    def _build_dataset(self, split):
        pass  # 未使用



    def default_transform(self, image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    def get_class_weights(self):
        """
        返回每个类别的加权 BCE 权重，值为 max_count / class_count。
        可用于 pos_weight 参数。
        """
        class_counts = torch.tensor([
            13967, 8592, 5364, 5192, 3838, 3550, 2710, 2703, 2082, 2057,
            1991, 1933, 1668, 1343, 1335, 1143, 1045, 997, 841, 705,
            634, 471, 338
        ], dtype=torch.float32)
        max_count = class_counts.max()
        weights = max_count / class_counts
        return weights

    def get_num_classes(self):
        return 23

