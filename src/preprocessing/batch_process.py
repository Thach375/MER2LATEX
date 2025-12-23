"""
Batch Preprocessing and Saving
================================
Process entire dataset and save as .npy files.

Run standalone:
    python -m src.preprocessing.batch_process
"""

import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from .preprocess_pipelines import preprocess_im2latex, preprocess_crohme
from src.utils.constants import *

def batch_preprocess_and_save(dataset_type='im2latex'):
    """
    Batch preprocess all images and save to disk (PNG only).

    Args:
        dataset_type: 'im2latex' or 'crohme'

    Returns:
        Dict of updated DataFrames with preprocessed image paths
    """
    if dataset_type == 'im2latex':
        # Process all splits
        train_df = pd.read_csv(IM2LATEX_LABEL_PATH / "im2latex_train.csv")
        val_df   = pd.read_csv(IM2LATEX_LABEL_PATH / "im2latex_validate.csv")
        test_df  = pd.read_csv(IM2LATEX_LABEL_PATH / "im2latex_test.csv")

        all_dfs = {'train': train_df, 'val': val_df, 'test': test_df}
        base_path = IM2LATEX_IMAGE_PATH
        output_path = IM2LATEX_OUTPUT_PATH
        preprocess_fn = preprocess_im2latex
    else:
        # CROHME - need to split manually
        df = pd.read_csv(CROHME_CSV_PATH)

        from sklearn.model_selection import train_test_split
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

        all_dfs = {'train': train_df, 'val': val_df, 'test': test_df}
        base_path = CROHME_IMAGE_PATH
        output_path = CROHME_OUTPUT_PATH
        preprocess_fn = preprocess_crohme

    def _to_uint8_grayscale(arr: np.ndarray) -> np.ndarray:
        """
        Convert preprocessed output to uint8 grayscale (H, W) for PNG saving.
        Supports (H,W), (1,H,W), (H,W,1), and (H,W,3).
        Assumes float outputs are in [0,1]; clips after scaling.
        """
        arr = np.asarray(arr)

        # Handle channels
        if arr.ndim == 3:
            if arr.shape[0] == 1:          # (1,H,W)
                arr = arr[0]
            elif arr.shape[-1] == 1:       # (H,W,1)
                arr = arr[..., 0]
            else:                          # (H,W,3) or others -> grayscale by mean
                arr = arr.mean(axis=-1)

        if arr.ndim != 2:
            raise ValueError(f"Unsupported preprocessed shape: {arr.shape}")

        # Float -> uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr_u8 = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
        else:
            arr_u8 = np.clip(arr, 0, 255).astype(np.uint8)

        return arr_u8

    # Process each split
    for split_name, df in all_dfs.items():
        print(f"\n{'='*60}")
        print(f"Processing {dataset_type.upper()} - {split_name.upper()} split")
        print(f"{'='*60}")

        # Create output directory
        split_output_path = output_path / split_name
        split_output_path.mkdir(exist_ok=True)

        preprocessed_paths = []
        valid_indices = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split_name}"):
            img_path = base_path / row['image']

            preprocessed = preprocess_fn(img_path, augment=False)
            if preprocessed is None:
                continue

            output_filename = Path(row['image']).stem + '.png'
            output_file = split_output_path / output_filename

            try:
                img_u8 = _to_uint8_grayscale(preprocessed)
                Image.fromarray(img_u8, mode="L").save(output_file, optimize=True)

                preprocessed_paths.append(str(output_file))
                valid_indices.append(idx)
            except Exception:
                # Skip problematic sample
                continue

        # Update DataFrame
        valid_df = df.loc[valid_indices].copy()
        valid_df['preprocessed_path'] = preprocessed_paths

        # Save updated CSV
        output_csv = output_path / f"{split_name}.csv"
        valid_df.to_csv(output_csv, index=False)

        print(f"✓ Processed {len(valid_df)} / {len(df)} images")
        print(f"✓ Saved to: {output_csv}")

        all_dfs[split_name] = valid_df

    return all_dfs


print("✓ Batch processing function ready")
print("\nNote: Saving all preprocessed images will require significant disk space.")
print("Consider processing on-the-fly during training instead.")