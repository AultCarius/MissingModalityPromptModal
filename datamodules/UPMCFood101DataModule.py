import os
import json
import torch
import random
from PIL import Image,ImageFile
import warnings
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from .baseDataModule import BaseDataModule

# Ignore warnings for large images
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
# Allow unlimited size images
Image.MAX_IMAGE_PIXELS = None
# Ignore truncated images error
ImageFile.LOAD_TRUNCATED_IMAGES = True


class UPMCFood101Dataset(Dataset):
    def __init__(self, sample_list, data_dir, image_transform=None, tokenizer=None,
                 max_length=128, missing_strategy="none",missing_prob=0.7):
        self.sample_list = sample_list
        self.data_dir = data_dir
        self.image_transform = image_transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

        self.tokenizer = tokenizer or CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
        self.max_length = max_length
        self.missing_strategy = missing_strategy
        self.missing_prob = missing_prob

        # Load class index mapping
        class_idx_path = os.path.join(data_dir, "class_idx.json")
        with open(class_idx_path, "r") as f:
            self.class_to_idx = json.load(f)

        # Load text descriptions
        text_path = os.path.join(data_dir, "text.json")
        with open(text_path, "r") as f:
            self.text_data = json.load(f)

        # Split data info to determine folder (train or test)
        split_path = os.path.join(data_dir, "split.json")
        with open(split_path, "r") as f:
            split_data = json.load(f)

        # Create sets for efficient lookup
        self.train_val_set = set(split_data["train"] + split_data["val"])
        self.test_set = set(split_data["test"])

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        # Get image filename
        img_filename = self.sample_list[idx]

        # Determine folder based on split
        folder = "train" if img_filename in self.train_val_set else "test"

        # Extract class name from filename (assuming format: class_name_id.jpg)
        class_name = "_".join(img_filename.split("_")[:-1])

        # Get label index
        label_idx = self.class_to_idx[class_name]

        # Create one-hot encoded label (101 classes)
        label = torch.zeros(len(self.class_to_idx))
        label[label_idx] = 1.0

        # Get image path
        img_path = os.path.join(self.data_dir, "images", folder, class_name, img_filename)

        # Get text description
        text = self.text_data.get(img_filename, "")  # Default to empty string if no text available

        # Process text
        encoded = self.tokenizer(text, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Process image
        # Process image - with error logging
        try:
            # Add a try-except block to catch warnings
            with warnings.catch_warnings(record=True) as caught_warnings:
                image = Image.open(img_path).convert("RGB")
                image = self.image_transform(image)
                # Check if any warnings were caught
                for warning in caught_warnings:
                    if "Truncated File Read" in str(warning.message):
                        print(f"WARNING: Truncated image detected - {img_path}")
        except Exception as e:
            print(f"ERROR loading image {img_path}: {e}")
            image = torch.zeros(3, 224, 224)

        # Simulate missing modality
        missing_type = self._simulate_missing()
        missing_type_tensor = torch.tensor({
                                               "none": 0, "image": 1, "text": 2, "both": 3
                                           }[missing_type])

        return (
            image, input_ids, attention_mask,
            label, missing_type_tensor
        )

    def _simulate_missing(self):
        """Simulate missing modalities based on strategy and probability"""
        if self.missing_strategy == "none":
            return "none"

        # Use instance missing rate
        eta = self.missing_prob

        # If missing_strategy is a float, use it as the missing rate
        if isinstance(self.missing_strategy, float):
            eta = self.missing_strategy

        # Apply different missing strategies
        if self.missing_strategy == "both":
            # η/2 probability of image missing, η/2 probability of text missing, 1-η probability intact
            return random.choices(
                ["none", "image", "text"],
                weights=[1 - eta, eta / 2, eta / 2],
                k=1
            )[0]
        elif self.missing_strategy == "image":
            # η probability of image missing, 1-η probability intact
            return random.choices(
                ["none", "image"],
                weights=[1 - eta, eta],
                k=1
            )[0]
        elif self.missing_strategy == "text":
            # η probability of text missing, 1-η probability intact
            return random.choices(
                ["none", "text"],
                weights=[1 - eta, eta],
                k=1
            )[0]

        # Default to none
        return "none"

    def update_missing_prob(self, new_prob):
        """Update the missing probability for this dataset"""
        self.missing_prob = new_prob
        return self


class UPMCFood101DataModule(BaseDataModule):
    def __init__(self, data_dir="./data/UCMFood101", batch_size=32, num_workers=4,
                 image_transform=None, tokenizer=None, max_length=128,
                 missing_strategy="none", missing_prob=0.7,
                 val_missing_strategy="none", val_missing_prob=0.0,
                 test_missing_strategy="none", test_missing_prob=0.0,
                 image_size=224, patch_size=16, seed=42):
        super().__init__(batch_size, num_workers)

        self.seed = seed
        self.data_dir = data_dir
        self.image_size = image_size
        self.patch_size = patch_size

        # Adjust image size to be multiple of patch size
        if self.image_size % self.patch_size != 0:
            self.image_size = (self.image_size // self.patch_size) * self.patch_size
            print(f"Adjusted image size to be multiple of patch_size: {self.image_size}")

        self.image_transform = image_transform or self.default_transform(self.image_size)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Missing modality settings
        self.missing_strategy = missing_strategy
        self.missing_prob = missing_prob
        self.val_missing_strategy = val_missing_strategy
        self.val_missing_prob = val_missing_prob
        self.test_missing_strategy = test_missing_strategy
        self.test_missing_prob = test_missing_prob

    def setup(self, stage=None):
        # Load split information
        split_path = os.path.join(self.data_dir, "split.json")
        with open(split_path, "r") as f:
            split_data = json.load(f)

        # Shuffle data (with fixed seed for reproducibility)
        random.seed(self.seed)
        random.shuffle(split_data["train"])
        random.shuffle(split_data["val"])
        random.shuffle(split_data["test"])

        # Parse missing strategies
        train_missing = self._parse_missing_strategy(self.missing_strategy, self.missing_prob)
        val_missing = self._parse_missing_strategy(self.val_missing_strategy, self.val_missing_prob)
        test_missing = self._parse_missing_strategy(self.test_missing_strategy, self.test_missing_prob)

        # Create datasets
        self.train_dataset = UPMCFood101Dataset(
            split_data["train"], self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=train_missing,
            missing_prob=self.missing_prob
        )

        self.val_dataset = UPMCFood101Dataset(
            split_data["val"], self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=val_missing,
            missing_prob=self.val_missing_prob
        )

        self.test_dataset = UPMCFood101Dataset(
            split_data["test"], self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=test_missing,
            missing_prob=self.test_missing_prob
        )

    def _parse_missing_strategy(self, strategy, prob):
        """Parse missing strategy."""
        if strategy == "random":
            return prob  # For random, return the probability
        elif strategy in ["both", "image", "text"]:
            return strategy
        else:
            return "none"

    def _build_dataset(self, split):
        # Not used but required by BaseDataModule
        pass

    def default_transform(self, image_size):
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    def get_class_weights(self):
        """
        Calculate class weights for weighted BCE loss.
        This is a simple implementation - ideally, you would count actual occurrences in the dataset.
        """
        # Load class index to get number of classes
        class_idx_path = os.path.join(self.data_dir, "class_idx.json")
        with open(class_idx_path, "r") as f:
            class_to_idx = json.load(f)

        # Create dummy counts (equal weights initially)
        class_counts = torch.ones(len(class_to_idx), dtype=torch.float32)

        # If you want to calculate actual counts from training data:
        # (This would require iterating through all samples which can be slow)
        # for sample in self.train_dataset:
        #     label = sample[3]  # Get label tensor
        #     class_counts += label

        # Inverse frequency weighting
        weights = 1.0 / (class_counts + 1e-5)  # Add small epsilon to avoid division by zero

        # Normalize weights
        weights = weights / weights.mean()

        return weights

    def get_num_classes(self):
        # Load class index to get number of classes
        class_idx_path = os.path.join(self.data_dir, "class_idx.json")
        with open(class_idx_path, "r") as f:
            class_to_idx = json.load(f)
        return len(class_to_idx)

