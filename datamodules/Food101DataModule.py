import os
from torchvision.datasets import Food101
from torchvision import transforms
from transformers import CLIPTokenizer
from PIL import Image
import torch
from torch.utils.data import Dataset
from .baseDataModule import BaseDataModule


class Food101DataModule(BaseDataModule):
    def __init__(self, data_dir, batch_size=32, num_workers=4, image_size=224):
        super().__init__(batch_size, num_workers)
        self.data_dir = data_dir
        self.image_size = image_size
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")

    def setup(self, stage=None):
        self.train_dataset = self._build_dataset(split="train")
        self.val_dataset = self._build_dataset(split="test")

    def _build_dataset(self, split):
        base_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        base_ds = Food101(root=self.data_dir, split=split, download=True)

        return Food101MultimodalWrapper(base_ds, transform=base_transform, tokenizer=self.tokenizer)


class Food101MultimodalWrapper(Dataset):
    def __init__(self, dataset, transform, tokenizer, max_length=16):
        self.dataset = dataset
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Food101 specific attributes
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = self.transform(image)

        # Get class name - in Food101, classes are stored as dataset.classes
        # and the integer label maps to the class name
        class_name = self.classes[label].replace('_', ' ')
        text = f"A photo of {class_name}"

        tokenized = self.tokenizer(text, return_tensors="pt", padding="max_length",
                                   truncation=True, max_length=self.max_length)

        return (
            image,
            tokenized["input_ids"].squeeze(0),
            tokenized["attention_mask"].squeeze(0),
            torch.tensor(label, dtype=torch.long)  # We can use the original label directly
        )