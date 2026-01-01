"""
Training Callbacks for MER2LATEX
=================================
Callbacks for checkpointing, early stopping, and logging.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import torch


class EarlyStopping:
    """
    Early stopping callback to stop training when metric stops improving.
    """
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for metrics like accuracy
            verbose: Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.should_stop = False
        
        if mode == 'min':
            self.compare = lambda x, y: x < y - min_delta
        else:
            self.compare = lambda x, y: x > y + min_delta
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.compare(score, self.best_score):
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] New best: {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("[EarlyStopping] Stopping training")
        
        return self.should_stop
    
    def reset(self):
        """Reset the callback state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False


class ModelCheckpoint:
    """
    Callback to save model checkpoints based on metric.
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_best_only: bool = False,
        save_last: bool = True,
        max_keep: int = 3,
        verbose: bool = True
    ):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            monitor: Metric to monitor
            mode: 'min' for loss, 'max' for accuracy/bleu
            save_best_only: Only save when metric improves
            save_last: Always save latest checkpoint
            max_keep: Maximum number of checkpoints to keep
            verbose: Print messages
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        self.max_keep = max_keep
        self.verbose = verbose
        
        self.best_score = None
        self.saved_checkpoints: List[Path] = []
        
        if mode == 'min':
            self.compare = lambda x, y: x < y
            self.best_score = float('inf')
        else:
            self.compare = lambda x, y: x > y
            self.best_score = float('-inf')
    
    def __call__(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        metrics: Dict[str, float],
        scaler: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Save checkpoint if appropriate.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Scheduler state
            metrics: Dictionary of metrics
            scaler: AMP scaler (optional)
        
        Returns:
            Dictionary with save info
        """
        score = metrics.get(self.monitor)
        if score is None:
            if self.verbose:
                print(f"[Checkpoint] Warning: {self.monitor} not in metrics")
            return {}
        
        is_best = self.compare(score, self.best_score)
        
        # Prepare checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'metrics': metrics,
            'best_score': self.best_score if is_best else score,
            'monitor': self.monitor
        }
        
        if scaler:
            checkpoint['scaler_state_dict'] = scaler.state_dict()
        
        saved_paths = []
        
        # Save best
        if is_best:
            self.best_score = score
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            saved_paths.append(str(best_path))
            
            if self.verbose:
                print(f"[Checkpoint] New best {self.monitor}: {score:.4f} -> {best_path}")
        
        # Save epoch checkpoint
        if not self.save_best_only or is_best:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch:03d}.pt'
            torch.save(checkpoint, epoch_path)
            saved_paths.append(str(epoch_path))
            self.saved_checkpoints.append(epoch_path)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
        
        # Save latest
        if self.save_last:
            latest_path = self.checkpoint_dir / 'latest.pt'
            torch.save(checkpoint, latest_path)
            if str(latest_path) not in saved_paths:
                saved_paths.append(str(latest_path))
        
        return {
            'saved_paths': saved_paths,
            'is_best': is_best,
            'best_score': self.best_score
        }
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only max_keep most recent."""
        while len(self.saved_checkpoints) > self.max_keep:
            old_path = self.saved_checkpoints.pop(0)
            if old_path.exists() and old_path.name not in ['best.pt', 'latest.pt']:
                old_path.unlink()
    
    def load_best(self, model: torch.nn.Module, device: str = 'cpu') -> Dict[str, Any]:
        """Load the best checkpoint."""
        best_path = self.checkpoint_dir / 'best.pt'
        if not best_path.exists():
            raise FileNotFoundError(f"No best checkpoint found at {best_path}")
        
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


class TrainingLogger:
    """
    Logger for training metrics and progress.
    """
    
    def __init__(
        self,
        log_dir: Path,
        experiment_name: str = 'experiment',
        log_to_file: bool = True,
        log_to_console: bool = True
    ):
        """
        Args:
            log_dir: Directory for log files
            experiment_name: Name of the experiment
            log_to_file: Write logs to file
            log_to_console: Print logs to console
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{timestamp}.log"
        self.metrics_file = self.log_dir / f"{experiment_name}_{timestamp}_metrics.json"
        
        self.history: List[Dict[str, Any]] = []
    
    def log(self, message: str, level: str = 'INFO'):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}"
        
        if self.log_to_console:
            print(formatted)
        
        if self.log_to_file:
            with open(self.log_file, 'a') as f:
                f.write(formatted + '\n')
    
    def log_metrics(self, epoch: int, metrics: Dict[str, float], phase: str = 'train'):
        """Log metrics for an epoch."""
        entry = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.history.append(entry)
        
        # Format metrics string
        metrics_str = ' | '.join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log(f"Epoch {epoch} [{phase}] {metrics_str}")
        
        # Save to JSON
        with open(self.metrics_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def log_hyperparams(self, hyperparams: Dict[str, Any]):
        """Log hyperparameters."""
        self.log("Hyperparameters:")
        for k, v in hyperparams.items():
            self.log(f"  {k}: {v}")
        
        # Save to separate file
        hp_file = self.log_dir / f"{self.experiment_name}_hyperparams.json"
        with open(hp_file, 'w') as f:
            json.dump(hyperparams, f, indent=2)
    
    def log_model_summary(self, model: torch.nn.Module):
        """Log model summary."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log(f"Model: {model.__class__.__name__}")
        self.log(f"  Total parameters: {total_params:,}")
        self.log(f"  Trainable parameters: {trainable_params:,}")
    
    def get_best_metrics(self, metric: str = 'val_loss', mode: str = 'min') -> Dict[str, Any]:
        """Get the best metrics from history."""
        if not self.history:
            return {}
        
        val_entries = [e for e in self.history if e.get('phase') == 'val']
        if not val_entries:
            return {}
        
        if mode == 'min':
            best = min(val_entries, key=lambda x: x.get(metric, float('inf')))
        else:
            best = max(val_entries, key=lambda x: x.get(metric, float('-inf')))
        
        return best
