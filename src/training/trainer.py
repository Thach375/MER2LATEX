"""
Trainer for MER2LATEX
======================
Main training loop with wandb logging, checkpointing, and early stopping.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available. Install with: pip install wandb")

from src.utils.constants import (
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    WARMUP_STEPS,
    GRADIENT_CLIP,
    LABEL_SMOOTHING,
    CHECKPOINT_ROOT,
    LOGS_ROOT,
    VOCAB_PATH,
    MAX_SEQ_LENGTH
)
from src.data.dataset import get_dataloader, MERDataset
from src.models import create_model
from src.tokenizer.tokenize import LaTeXTokenizer
from src.evaluation import compute_metrics, decode_predictions
from src.training.callbacks import EarlyStopping, ModelCheckpoint, TrainingLogger


class Trainer:
    """
    Unified trainer for all MER2LATEX models.
    
    Features:
    - Multiple model support (A, B, C, D)
    - Wandb integration for experiment tracking
    - Checkpoint saving by best metric (val_loss or val_bleu)
    - Early stopping
    - Mixed precision training (AMP)
    - Gradient clipping
    - Learning rate scheduling
    """
    
    def __init__(
        self,
        model_name: str,
        dataset_type: str = 'im2latex',
        combined_dataset: bool = False,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        num_epochs: int = NUM_EPOCHS,
        gradient_clip: float = GRADIENT_CLIP,
        label_smoothing: float = LABEL_SMOOTHING,
        # Wandb settings
        use_wandb: bool = True,
        wandb_project: str = "MER2LATEX",
        wandb_run_name: Optional[str] = None,
        # Checkpoint settings
        checkpoint_dir: Optional[Path] = None,
        checkpoint_metric: str = 'val_bleu',  # BLEU tot nhat cho MER
        checkpoint_mode: str = 'max',  # max vi BLEU cao = tot
        save_best_only: bool = False,
        max_checkpoints: int = 3,
        # Early stopping - mac dinh TAT de train du epochs
        early_stopping: bool = False,
        patience: int = 5,
        # Other settings
        device: Optional[str] = None,
        use_amp: bool = True,
        num_workers: int = 4,
        resume_from: Optional[str] = None,
        seed: int = 42
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: 'model_a', 'model_b', 'model_c', or 'model_d'
            dataset_type: 'im2latex' or 'crohme'
            combined_dataset: Use both IM2LATEX and CROHME
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            num_epochs: Maximum number of epochs
            gradient_clip: Gradient clipping value
            label_smoothing: Label smoothing factor
            use_wandb: Enable wandb logging
            wandb_project: Wandb project name
            wandb_run_name: Wandb run name (auto-generated if None)
            checkpoint_dir: Directory for checkpoints
            checkpoint_metric: Metric to monitor for best checkpoint
            checkpoint_mode: 'min' for loss, 'max' for accuracy/bleu
            save_best_only: Only save when metric improves
            max_checkpoints: Maximum checkpoints to keep
            early_stopping: Enable early stopping
            patience: Early stopping patience
            device: Device to use ('cuda' or 'cpu')
            use_amp: Use automatic mixed precision
            num_workers: Number of data loading workers
            resume_from: Path to checkpoint to resume from
            seed: Random seed
        """
        # Set seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        self.model_name = model_name
        self.dataset_type = dataset_type
        self.combined_dataset = combined_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_clip = gradient_clip
        self.label_smoothing = label_smoothing
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.wandb_project = wandb_project
        self.use_amp = use_amp
        self.num_workers = num_workers
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[INFO] Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[INFO] Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Checkpoint directory - structure: checkpoints/model_name/timestamp/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if checkpoint_dir is None:
            self.checkpoint_dir = CHECKPOINT_ROOT / model_name / timestamp
        else:
            self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Log directory
        self.log_dir = LOGS_ROOT / model_name / timestamp
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._setup_data()
        self._setup_model()
        self._setup_optimizer()
        self._setup_criterion()
        
        # AMP scaler
        self.scaler = GradScaler() if self.use_amp and self.device.type == 'cuda' else None
        
        # Callbacks
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=self.checkpoint_dir,
            monitor=checkpoint_metric,
            mode=checkpoint_mode,
            save_best_only=save_best_only,
            max_keep=max_checkpoints,
            verbose=True
        )
        
        self.early_stopping = EarlyStopping(
            patience=patience,
            mode=checkpoint_mode,
            verbose=True
        ) if early_stopping else None
        
        self.logger = TrainingLogger(
            log_dir=self.log_dir,
            experiment_name=f"{model_name}_{dataset_type}"
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.best_val_bleu = 0.0
        self.best_metrics = {}
        
        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # Initialize wandb
        if self.use_wandb:
            self._init_wandb(wandb_run_name)
        
        # Log setup
        self._log_setup()
    
    def _setup_data(self):
        """Setup data loaders."""
        print("[INFO] Setting up data loaders...")
        
        self.train_loader, self.tokenizer = get_dataloader(
            dataset_type=self.dataset_type,
            split='train',
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            use_preprocessed=True,
            combined=self.combined_dataset
        )
        
        self.val_loader, _ = get_dataloader(
            dataset_type=self.dataset_type,
            split='val',
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            num_workers=self.num_workers,
            use_preprocessed=True,
            combined=self.combined_dataset
        )
        
        print(f"[INFO] Train samples: {len(self.train_loader.dataset)}")
        print(f"[INFO] Val samples: {len(self.val_loader.dataset)}")
        print(f"[INFO] Vocab size: {self.tokenizer.vocab_size}")
    
    def _setup_model(self):
        """Setup model."""
        print(f"[INFO] Setting up model: {self.model_name}")
        
        model_kwargs = {
            'hidden_dim': 256,
            'dropout': 0.1,
        }
        
        if self.model_name in ['model_b', 'model_c', 'model_d']:
            model_kwargs['pretrained'] = True
        
        self.model = create_model(
            self.model_name,
            vocab_size=self.tokenizer.vocab_size,
            **model_kwargs
        )
        
        self.model = self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[INFO] Total parameters: {num_params:,}")
        print(f"[INFO] Trainable parameters: {trainable_params:,}")
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler."""
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        total_steps = len(self.train_loader) * self.num_epochs
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
    
    def _setup_criterion(self):
        """Setup loss function."""
        if self.model_name == 'model_a':
            self.criterion = nn.CTCLoss(blank=self.tokenizer.pad_id, zero_infinity=True)
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.tokenizer.pad_id,
                label_smoothing=self.label_smoothing
            )
    
    def _init_wandb(self, run_name: Optional[str] = None):
        """Initialize wandb logging."""
        config = {
            'model_name': self.model_name,
            'dataset_type': self.dataset_type,
            'combined_dataset': self.combined_dataset,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'gradient_clip': self.gradient_clip,
            'label_smoothing': self.label_smoothing,
            'vocab_size': self.tokenizer.vocab_size,
            'max_seq_length': MAX_SEQ_LENGTH,
            'device': str(self.device),
            'use_amp': self.use_amp,
            'num_train_samples': len(self.train_loader.dataset),
            'num_val_samples': len(self.val_loader.dataset),
        }
        
        if run_name is None:
            run_name = f"{self.model_name}_{self.dataset_type}_{datetime.now().strftime('%m%d_%H%M')}"
        
        wandb.init(
            project=self.wandb_project,
            name=run_name,
            config=config,
            resume='allow',
            dir=str(self.log_dir)
        )
        
        wandb.watch(self.model, log='gradients', log_freq=100)
    
    def _log_setup(self):
        """Log training setup."""
        self.logger.log_hyperparams({
            'model_name': self.model_name,
            'dataset_type': self.dataset_type,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'num_epochs': self.num_epochs,
            'gradient_clip': self.gradient_clip,
            'label_smoothing': self.label_smoothing,
            'vocab_size': self.tokenizer.vocab_size,
        })
        self.logger.log_model_summary(self.model)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()
        
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.num_epochs} [Train]"
        )
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                if self.model_name == 'model_a':
                    loss = self._compute_ctc_loss(images, labels, lengths)
                else:
                    loss = self._compute_ce_loss(images, labels)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"[WARNING] NaN loss at batch {batch_idx}, skipping...")
                continue
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            current_lr = self.scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}",
                'grad': f"{grad_norm:.2f}"
            })
            
            # Log to wandb
            if self.use_wandb and self.global_step % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/grad_norm': grad_norm,
                    'global_step': self.global_step
                })
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / max(num_batches, 1)
        
        metrics = {
            'train_loss': avg_loss,
            'train_time': epoch_time,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        self.logger.log_metrics(self.current_epoch + 1, metrics, phase='train')
        
        return metrics
    
    def _compute_ctc_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute CTC loss for Model A."""
        log_probs = self.model(images)  # (T, B, vocab_size)
        
        T, B, _ = log_probs.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=self.device)
        
        # Remove BOS and EOS for CTC
        target_lengths = lengths - 2
        target_lengths = target_lengths.clamp(min=1)
        
        targets = labels[:, 1:]  # Remove BOS
        
        # Flatten targets
        targets_flat = []
        for i in range(B):
            tgt_len = target_lengths[i].item()
            targets_flat.append(targets[i, :tgt_len])
        targets_flat = torch.cat(targets_flat)
        
        loss = self.criterion(log_probs, targets_flat, input_lengths, target_lengths)
        
        return loss
    
    def _compute_ce_loss(
        self,
        images: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-entropy loss for seq2seq models."""
        decoder_input = labels[:, :-1]
        targets = labels[:, 1:]
        
        if self.model_name == 'model_b':
            tf_ratio = max(0.5, 1.0 - self.current_epoch / self.num_epochs)
            outputs, _ = self.model(images, decoder_input, teacher_forcing_ratio=tf_ratio)
        else:
            outputs = self.model(images, decoder_input)
        
        outputs = outputs.contiguous().view(-1, self.tokenizer.vocab_size)
        targets = targets.contiguous().view(-1)
        
        loss = self.criterion(outputs, targets)
        
        return loss
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        val_start = time.time()
        
        for batch in tqdm(self.val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]"):
            images = batch['images'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['lengths'].to(self.device)
            formulas = batch['formulas']
            
            # Compute loss
            with autocast(enabled=self.use_amp and self.device.type == 'cuda'):
                if self.model_name == 'model_a':
                    loss = self._compute_ctc_loss(images, labels, lengths)
                else:
                    loss = self._compute_ce_loss(images, labels)
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1
            
            # Decode predictions
            predictions = decode_predictions(
                self.model,
                images,
                self.tokenizer,
                model_type=self.model_name
            )
            
            all_predictions.extend(predictions)
            all_targets.extend(formulas)
        
        val_time = time.time() - val_start
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_targets)
        metrics['val_loss'] = avg_loss
        metrics['val_time'] = val_time
        
        # Log
        self.logger.log_metrics(self.current_epoch + 1, metrics, phase='val')
        
        # Log sample predictions
        if len(all_predictions) > 0:
            print(f"\n[Sample Predictions]")
            for i in range(min(3, len(all_predictions))):
                print(f"  Target: {all_targets[i][:50]}...")
                print(f"  Pred:   {all_predictions[i][:50]}...")
                print()
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """Full training loop."""
        print(f"\n{'='*60}")
        print(f"Starting training: {self.model_name}")
        print(f"Dataset: {self.dataset_type}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Batch size: {self.batch_size}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"{'='*60}\n")
        
        training_history = []
        
        try:
            for epoch in range(self.current_epoch, self.num_epochs):
                self.current_epoch = epoch
                
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self.validate()
                
                # Combine metrics
                epoch_metrics = {
                    **train_metrics,
                    **val_metrics,
                    'epoch': epoch + 1
                }
                training_history.append(epoch_metrics)
                
                # Print results
                print(f"\n{'='*40}")
                print(f"Epoch {epoch + 1}/{self.num_epochs} Summary")
                print(f"{'='*40}")
                print(f"  Train Loss:    {train_metrics['train_loss']:.4f}")
                print(f"  Val Loss:      {val_metrics['val_loss']:.4f}")
                print(f"  Exact Match:   {val_metrics['exact_match']:.4f}")
                print(f"  BLEU:          {val_metrics['bleu']:.4f}")
                print(f"  Edit Distance: {val_metrics['edit_distance']:.4f}")
                print(f"  Train Time:    {train_metrics['train_time']:.1f}s")
                print(f"  Val Time:      {val_metrics['val_time']:.1f}s")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log({
                        'epoch': epoch + 1,
                        'val/loss': val_metrics['val_loss'],
                        'val/exact_match': val_metrics['exact_match'],
                        'val/bleu': val_metrics['bleu'],
                        'val/edit_distance': val_metrics['edit_distance'],
                        'val/sympy_equivalence': val_metrics.get('sympy_equivalence', 0),
                    })
                
                # Update best metrics
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_metrics = val_metrics.copy()
                
                if val_metrics['bleu'] > self.best_val_bleu:
                    self.best_val_bleu = val_metrics['bleu']
                
                # Save checkpoint
                ckpt_result = self.checkpoint_callback(
                    epoch=epoch + 1,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=val_metrics,
                    scaler=self.scaler
                )
                
                # Early stopping
                if self.early_stopping:
                    if self.early_stopping(val_metrics['val_loss']):
                        print(f"\n[INFO] Early stopping triggered at epoch {epoch + 1}")
                        break
        
        except KeyboardInterrupt:
            print("\n[INFO] Training interrupted by user")
        
        # Final summary
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}")
        print(f"Best Val Loss: {self.best_val_loss:.4f}")
        print(f"Best BLEU: {self.best_val_bleu:.4f}")
        print(f"Best Metrics: {self.best_metrics}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        
        # Close wandb
        if self.use_wandb:
            # Log final summary
            wandb.run.summary['best_val_loss'] = self.best_val_loss
            wandb.run.summary['best_bleu'] = self.best_val_bleu
            wandb.run.summary['total_epochs'] = self.current_epoch + 1
            wandb.finish()
        
        return {
            'history': training_history,
            'best_metrics': self.best_metrics,
            'best_val_loss': self.best_val_loss,
            'best_bleu': self.best_val_bleu,
            'checkpoint_dir': str(self.checkpoint_dir),
            'log_dir': str(self.log_dir)
        }
    
    def save_checkpoint(self, path: Optional[Path] = None, is_best: bool = False):
        """Manual checkpoint saving."""
        if path is None:
            path = self.checkpoint_dir / f'manual_epoch_{self.current_epoch + 1}.pt'
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics,
            'model_name': self.model_name,
            'vocab_size': self.tokenizer.vocab_size,
            'config': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
            }
        }
        
        if self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        print(f"[INFO] Saved checkpoint: {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        print(f"[INFO] Loading checkpoint: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_metrics = checkpoint.get('best_metrics', {})
        
        if self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"[INFO] Resumed from epoch {self.current_epoch}")


def train_model(
    model_name: str,
    dataset_type: str = 'im2latex',
    combined: bool = False,
    batch_size: int = BATCH_SIZE,
    num_epochs: int = NUM_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    use_wandb: bool = True,
    checkpoint_metric: str = 'val_bleu',  # BLEU tot nhat cho MER
    checkpoint_mode: str = 'max',  # max vi BLEU cao = tot
    early_stopping: bool = False,  # Mac dinh TAT de train du epochs
    patience: int = 5,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to train a model.
    
    Args:
        model_name: Model to train ('model_a', 'model_b', 'model_c', 'model_d')
        dataset_type: Dataset to use ('im2latex' or 'crohme')
        combined: Use combined dataset
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        use_wandb: Enable wandb logging
        checkpoint_metric: Metric for best checkpoint (default: 'val_bleu')
        checkpoint_mode: 'min' for loss, 'max' for bleu/accuracy
        early_stopping: Enable early stopping (default: False)
        patience: Early stopping patience
        **kwargs: Additional trainer arguments
    
    Returns:
        Training results dictionary
    """
    # Auto-detect mode if not explicitly set
    if checkpoint_mode is None:
        checkpoint_mode = 'min' if 'loss' in checkpoint_metric else 'max'
    
    trainer = Trainer(
        model_name=model_name,
        dataset_type=dataset_type,
        combined_dataset=combined,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        use_wandb=use_wandb,
        checkpoint_metric=checkpoint_metric,
        checkpoint_mode=checkpoint_mode,
        early_stopping=early_stopping,
        patience=patience,
        **kwargs
    )
    
    return trainer.train()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MER2LATEX models')
    parser.add_argument('--model', type=str, default='model_b',
                       choices=['model_a', 'model_b', 'model_c', 'model_d'],
                       help='Model to train')
    parser.add_argument('--dataset', type=str, default='im2latex',
                       choices=['im2latex', 'crohme'],
                       help='Dataset to use')
    parser.add_argument('--combined', action='store_true',
                       help='Use combined dataset')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable wandb logging')
    parser.add_argument('--checkpoint-metric', type=str, default='val_bleu',
                       choices=['val_loss', 'val_bleu', 'val_exact_match'],
                       help='Metric for best checkpoint (default: val_bleu)')
    parser.add_argument('--early-stop', action='store_true',
                       help='Enable early stopping (default: disabled)')
    parser.add_argument('--patience', type=int, default=5,
                       help='Early stopping patience (only if --early-stop)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Determine checkpoint mode based on metric
    checkpoint_mode = 'min' if args.checkpoint_metric == 'val_loss' else 'max'
    
    results = train_model(
        model_name=args.model,
        dataset_type=args.dataset,
        combined=args.combined,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        use_wandb=not args.no_wandb,
        checkpoint_metric=args.checkpoint_metric,
        checkpoint_mode=checkpoint_mode,
        early_stopping=args.early_stop,  # Default OFF
        patience=args.patience,
        resume_from=args.resume
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Best val loss: {results['best_val_loss']:.4f}")
    print(f"Best BLEU: {results['best_bleu']:.4f}")
    print(f"Checkpoints: {results['checkpoint_dir']}")
    print(f"Logs: {results['log_dir']}")
