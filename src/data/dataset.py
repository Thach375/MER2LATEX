"""
Dataset Module for MER2LATEX
=============================
PyTorch Dataset and DataLoader for training models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, Dict, List

from src.utils.constants import (
    IM2LATEX_IMAGE_PATH,
    IM2LATEX_TRAIN_CSV,
    IM2LATEX_VAL_CSV,
    IM2LATEX_TEST_CSV,
    IM2LATEX_OUTPUT_PATH,
    CROHME_IMAGE_PATH,
    CROHME_CSV_PATH,
    CROHME_OUTPUT_PATH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    MAX_SEQ_LENGTH,
    VOCAB_PATH,
    BATCH_SIZE
)
from src.tokenizer.tokenize import LaTeXTokenizer
from src.preprocessing.preprocess_pipelines import preprocess_im2latex, preprocess_crohme


class MERDataset(Dataset):
    """
    Dataset for Math Expression Recognition.
    Supports both IM2LATEX and CROHME datasets.
    """
    
    def __init__(
        self,
        dataset_type: str = 'im2latex',
        split: str = 'train',
        tokenizer: Optional[LaTeXTokenizer] = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        use_preprocessed: bool = True,
        augment: bool = False
    ):
        """
        Args:
            dataset_type: 'im2latex' or 'crohme'
            split: 'train', 'val', or 'test'
            tokenizer: LaTeXTokenizer instance
            max_seq_length: Maximum sequence length for labels
            use_preprocessed: Use preprocessed images if available
            augment: Apply data augmentation (for training)
        """
        self.dataset_type = dataset_type
        self.split = split
        self.max_seq_length = max_seq_length
        self.use_preprocessed = use_preprocessed
        self.augment = augment and (split == 'train')
        
        # Initialize tokenizer
        self.tokenizer = tokenizer if tokenizer else LaTeXTokenizer(VOCAB_PATH)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load dataset based on type and split."""
        if self.dataset_type == 'im2latex':
            self._load_im2latex()
        else:
            self._load_crohme()
    
    def _load_im2latex(self):
        """Load IM2LATEX dataset."""
        # Check for preprocessed data first
        preprocessed_csv = IM2LATEX_OUTPUT_PATH / f"{self.split}.csv"
        
        if self.use_preprocessed and preprocessed_csv.exists():
            self.df = pd.read_csv(preprocessed_csv)
            self.image_dir = None  # Use preprocessed_path column
            self.use_preprocessed_path = True
        else:
            # Use original data
            if self.split == 'train':
                self.df = pd.read_csv(IM2LATEX_TRAIN_CSV)
            elif self.split == 'val':
                self.df = pd.read_csv(IM2LATEX_VAL_CSV)
            else:
                self.df = pd.read_csv(IM2LATEX_TEST_CSV)
            
            self.image_dir = IM2LATEX_IMAGE_PATH
            self.use_preprocessed_path = False
        
        self.preprocess_fn = preprocess_im2latex
    
    def _load_crohme(self):
        """Load CROHME dataset."""
        # Check for preprocessed data first
        preprocessed_csv = CROHME_OUTPUT_PATH / f"{self.split}.csv"
        
        if self.use_preprocessed and preprocessed_csv.exists():
            self.df = pd.read_csv(preprocessed_csv)
            self.image_dir = None
            self.use_preprocessed_path = True
        else:
            # Load from CROHME processed data
            if not CROHME_CSV_PATH.exists():
                raise FileNotFoundError(
                    f"CROHME dataset not found at {CROHME_CSV_PATH}. "
                    "Please run: python -m src.utils.download_data"
                )
            
            full_df = pd.read_csv(CROHME_CSV_PATH)
            self.image_dir = CROHME_IMAGE_PATH
            
            from sklearn.model_selection import train_test_split
            train_df, temp_df = train_test_split(full_df, test_size=0.2, random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
            
            if self.split == 'train':
                self.df = train_df.reset_index(drop=True)
            elif self.split == 'val':
                self.df = val_df.reset_index(drop=True)
            else:
                self.df = test_df.reset_index(drop=True)
            
            self.use_preprocessed_path = False
        
        self.preprocess_fn = preprocess_crohme
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load image
        if self.use_preprocessed_path and 'preprocessed_path' in row:
            img_path = row['preprocessed_path']
            image = np.array(Image.open(img_path).convert('L'))
        else:
            img_path = self.image_dir / row['image']
            image = self.preprocess_fn(img_path, augment=self.augment)
            
            if image is None:
                # Return a blank image if preprocessing fails
                image = np.ones((TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8) * 255
        
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        image = np.expand_dims(image, axis=0)
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image)
        
        # Encode formula
        formula = row['formula'] if pd.notna(row['formula']) else ""
        token_ids = self.tokenizer.encode(formula, add_bos=True, add_eos=True)
        
        # Truncate or pad
        if len(token_ids) > self.max_seq_length:
            token_ids = token_ids[:self.max_seq_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        seq_length = len(token_ids)
        
        # Pad to max_seq_length
        padding_length = self.max_seq_length - seq_length
        token_ids = token_ids + [self.tokenizer.pad_id] * padding_length
        
        label_tensor = torch.tensor(token_ids, dtype=torch.long)
        length_tensor = torch.tensor(seq_length, dtype=torch.long)
        
        return {
            'image': image_tensor,
            'label': label_tensor,
            'length': length_tensor,
            'formula': formula
        }


class CombinedDataset(Dataset):
    """
    Combined dataset that merges IM2LATEX and CROHME.
    """
    
    def __init__(
        self,
        split: str = 'train',
        tokenizer: Optional[LaTeXTokenizer] = None,
        max_seq_length: int = MAX_SEQ_LENGTH,
        use_preprocessed: bool = True,
        augment: bool = False
    ):
        self.im2latex = MERDataset(
            dataset_type='im2latex',
            split=split,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            use_preprocessed=use_preprocessed,
            augment=augment
        )
        
        self.crohme = MERDataset(
            dataset_type='crohme',
            split=split,
            tokenizer=self.im2latex.tokenizer,
            max_seq_length=max_seq_length,
            use_preprocessed=use_preprocessed,
            augment=augment
        )
        
        self.tokenizer = self.im2latex.tokenizer
        self.im2latex_len = len(self.im2latex)
    
    def __len__(self) -> int:
        return len(self.im2latex) + len(self.crohme)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.im2latex_len:
            return self.im2latex[idx]
        else:
            return self.crohme[idx - self.im2latex_len]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader.
    """
    images = torch.stack([item['image'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    lengths = torch.stack([item['length'] for item in batch])
    formulas = [item['formula'] for item in batch]
    
    return {
        'images': images,
        'labels': labels,
        'lengths': lengths,
        'formulas': formulas
    }


def get_dataloader(
    dataset_type: str = 'im2latex',
    split: str = 'train',
    batch_size: int = BATCH_SIZE,
    tokenizer: Optional[LaTeXTokenizer] = None,
    num_workers: int = 4,
    use_preprocessed: bool = True,
    combined: bool = False
) -> Tuple[DataLoader, LaTeXTokenizer]:
    """
    Get DataLoader for training/validation/testing.
    
    Args:
        dataset_type: 'im2latex' or 'crohme'
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        tokenizer: Optional tokenizer (will create one if not provided)
        num_workers: Number of workers for data loading
        use_preprocessed: Use preprocessed images if available
        combined: Use combined dataset (IM2LATEX + CROHME)
    
    Returns:
        DataLoader and tokenizer
    """
    augment = (split == 'train')
    
    if combined:
        dataset = CombinedDataset(
            split=split,
            tokenizer=tokenizer,
            use_preprocessed=use_preprocessed,
            augment=augment
        )
        tokenizer = dataset.tokenizer
    else:
        dataset = MERDataset(
            dataset_type=dataset_type,
            split=split,
            tokenizer=tokenizer,
            use_preprocessed=use_preprocessed,
            augment=augment
        )
        tokenizer = dataset.tokenizer
    
    shuffle = (split == 'train')
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=(split == 'train')
    )
    
    return dataloader, tokenizer


if __name__ == "__main__":
    # Test dataset loading
    print("Testing MERDataset...")
    
    # Test IM2LATEX
    print("\n[TEST] IM2LATEX Dataset")
    try:
        train_loader, tokenizer = get_dataloader(
            dataset_type='im2latex',
            split='train',
            batch_size=4,
            use_preprocessed=False
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        
        # Get a batch
        batch = next(iter(train_loader))
        print(f"  Batch images shape: {batch['images'].shape}")
        print(f"  Batch labels shape: {batch['labels'].shape}")
        print(f"  Sample formula: {batch['formulas'][0][:50]}...")
        print("[OK] IM2LATEX Dataset")
    except Exception as e:
        print(f"[ERROR] IM2LATEX: {e}")
    
    # Test CROHME
    print("\n[TEST] CROHME Dataset")
    try:
        train_loader, tokenizer = get_dataloader(
            dataset_type='crohme',
            split='train',
            batch_size=4,
            use_preprocessed=False
        )
        print(f"  Train samples: {len(train_loader.dataset)}")
        
        batch = next(iter(train_loader))
        print(f"  Batch images shape: {batch['images'].shape}")
        print(f"  Batch labels shape: {batch['labels'].shape}")
        print("[OK] CROHME Dataset")
    except Exception as e:
        print(f"[ERROR] CROHME: {e}")
    
    print("\n[DONE] Dataset testing complete!")
