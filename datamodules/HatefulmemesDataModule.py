import os
import json
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPTokenizer
from .baseDataModule import BaseDataModule
import jsonlines
import warnings
from PIL import Image, ImageFile

# Ignore "large image" warnings
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
# Allow unlimited image size
Image.MAX_IMAGE_PIXELS = None
# Ignore truncated image errors (prevent crashes from corrupted images)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class HatefulMemesDataset(Dataset):
    def __init__(self, samples, data_dir, image_transform=None, tokenizer=None,
                 max_length=128, missing_strategy="none", missing_prob=0.7):
        """
        Initialize the Hateful Memes dataset.

        Args:
            samples: List of sample dictionaries from the jsonl files
            data_dir: Root directory containing the dataset
            image_transform: Optional transform for images
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length for tokenizer
            missing_strategy: Strategy for modality missing simulation
            missing_prob: Probability of missing modality
        """
        self.samples = samples
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load image
        img_path = os.path.join(self.data_dir, 'data', sample['img'])
        image = Image.open(img_path).convert("RGB")
        image = self.image_transform(image)

        # 2. Process text
        text = sample['text']
        encoded = self.tokenizer(text, padding="max_length", truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # 3. Get label
        # Binary classification (hateful=1, non-hateful=0)
        label = torch.zeros(2)  # One-hot encoding [non-hateful, hateful]
        label[sample['label']] = 1.0

        # 4. Simulate modality missing
        missing_type = self._simulate_missing()
        missing_type_tensor = torch.tensor({
                                               "none": 0, "image": 1, "text": 2, "both": 3
                                           }[missing_type])

        # 5. Zero out missing modalities
        if missing_type in ["image", "both"]:
            image = torch.zeros_like(image)

        if missing_type in ["text", "both"]:
            input_ids = torch.zeros_like(input_ids)
            attention_mask = torch.zeros_like(attention_mask)

        return (image, input_ids, attention_mask, label, missing_type_tensor)

    def _simulate_missing(self):
        """
        Simulate modality missing based on the configured strategy and probability.

        Returns:
            str: Missing type, one of "none", "image", "text", or "both"
        """
        if self.missing_strategy == "none":
            return "none"

        # Use instance variable missing probability
        eta = self.missing_prob

        # If a float is provided directly as strategy, use it as the missing rate
        if isinstance(self.missing_strategy, float):
            eta = self.missing_strategy

        # Apply different missing patterns based on strategy
        if self.missing_strategy == "both":
            # η/2 probability image missing, η/2 probability text missing, 1-η probability complete
            return random.choices(
                ["none", "image", "text"],
                weights=[1 - eta, eta / 2, eta / 2],
                k=1
            )[0]
        elif self.missing_strategy == "image":
            # η probability image missing, 1-η probability complete
            return random.choices(
                ["none", "image"],
                weights=[1 - eta, eta],
                k=1
            )[0]
        elif self.missing_strategy == "text":
            # η probability text missing, 1-η probability complete
            return random.choices(
                ["none", "text"],
                weights=[1 - eta, eta],
                k=1
            )[0]

        # Default to no missing
        return "none"

    def update_missing_prob(self, new_prob):
        """Update the missing probability for this dataset"""
        self.missing_prob = new_prob
        return self


class HatefulMemesDataModule(BaseDataModule):
    def __init__(self, data_dir="./data/hateful_memes", batch_size=32, num_workers=4,
                 image_transform=None, tokenizer=None, max_length=128,
                 missing_strategy="none", missing_prob=0.7,
                 val_missing_strategy="none", val_missing_prob=0.0,
                 test_missing_strategy="none", test_missing_prob=0.0,
                 image_size=224, patch_size=16, seed=42):
        """
        Initialize the Hateful Memes data module.

        Args:
            data_dir: Root directory for the dataset
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for dataloaders
            image_transform: Optional transform for images
            tokenizer: Tokenizer for text processing
            max_length: Maximum sequence length for tokenizer
            missing_strategy: Strategy for modality missing simulation in training
            missing_prob: Probability of missing modality in training
            val_missing_strategy: Strategy for modality missing simulation in validation
            val_missing_prob: Probability of missing modality in validation
            test_missing_strategy: Strategy for modality missing simulation in testing
            test_missing_prob: Probability of missing modality in testing
            image_size: Size of input images
            patch_size: Patch size for vision models
            seed: Random seed
        """
        super().__init__(batch_size, num_workers)

        self.seed = seed
        self.data_dir = data_dir

        # Ensure image_size is divisible by patch_size
        self.image_size = image_size
        self.patch_size = patch_size
        if self.image_size % self.patch_size != 0:
            self.image_size = (self.image_size // self.patch_size) * self.patch_size
            print(f"Adjusted image size to be divisible by patch_size: {self.image_size}")

        self.image_transform = image_transform or self.default_transform(self.image_size)
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Missing modality simulation settings
        self.missing_strategy = missing_strategy
        self.missing_prob = missing_prob
        self.val_missing_strategy = val_missing_strategy
        self.val_missing_prob = val_missing_prob
        self.test_missing_strategy = test_missing_strategy
        self.test_missing_prob = test_missing_prob

    def setup(self, stage=None):
        """
        Load and prepare datasets for all splits.
        """
        # Set random seed for reproducibility
        random.seed(self.seed)

        # Load data from jsonl files
        train_samples = self._load_jsonl_data('train')
        val_samples = self._load_jsonl_data('dev')
        test_samples = self._load_jsonl_data('test')

        # Shuffle samples
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)

        # Parse missing strategies
        train_missing = self._parse_missing_strategy(self.missing_strategy, self.missing_prob)
        val_missing = self._parse_missing_strategy(self.val_missing_strategy, self.val_missing_prob)
        test_missing = self._parse_missing_strategy(self.test_missing_strategy, self.test_missing_prob)

        # Create datasets
        self.train_dataset = HatefulMemesDataset(
            train_samples,
            self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=train_missing,
            missing_prob=self.missing_prob
        )

        self.val_dataset = HatefulMemesDataset(
            val_samples,
            self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=val_missing,
            missing_prob=self.val_missing_prob
        )

        self.test_dataset = HatefulMemesDataset(
            test_samples,
            self.data_dir,
            image_transform=self.image_transform,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            missing_strategy=test_missing,
            missing_prob=self.test_missing_prob
        )

    def _load_jsonl_data(self, split):
        """Load data from jsonl files for a specific split."""
        samples = []
        jsonl_path = os.path.join(self.data_dir, 'data', f'{split}.jsonl')

        try:
            with jsonlines.open(jsonl_path, 'r') as reader:
                for item in reader:
                    samples.append(item)
        except Exception as e:
            # Alternative method using regular json parsing
            try:
                with open(jsonl_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            samples.append(json.loads(line))
            except Exception as e2:
                raise ValueError(f"Failed to load {split} data: {e}, then tried alternate method: {e2}")

        return samples

    def _parse_missing_strategy(self, strategy, prob):
        """Parse the missing strategy configuration."""
        if strategy == "random":
            return prob  # For random strategy, return the probability
        elif strategy in ["both", "image", "text"]:
            return strategy
        else:
            return "none"

    def _build_dataset(self, split):
        """Not used - the setup method handles dataset creation."""
        pass

    def default_transform(self, image_size):
        """Default image transform."""
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
        ])

    def get_class_weights(self):
        """
        Return class weights for weighted loss functions.
        For binary classification, we can estimate based on class distribution.
        """
        if hasattr(self, 'train_dataset') and self.train_dataset:
            hateful_count = sum(sample['label'] for sample in self.train_dataset.samples)
            total_count = len(self.train_dataset.samples)
            non_hateful_count = total_count - hateful_count

            # Calculate weights inversely proportional to class frequencies
            if hateful_count > 0 and non_hateful_count > 0:
                max_count = max(hateful_count, non_hateful_count)
                weights = torch.tensor([
                    max_count / non_hateful_count,  # Weight for non-hateful class
                    max_count / hateful_count  # Weight for hateful class
                ], dtype=torch.float32)
                return weights

        # Default weights if dataset not loaded or counts are invalid
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

    def get_num_classes(self):
        """Return number of classes (2 for binary classification)."""
        return 2