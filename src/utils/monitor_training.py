#!/usr/bin/env python3
"""
Training Monitor
=================
Script de theo doi training progress tu terminal.

Usage:
    python -m src.utils.monitor_training                    # Xem run moi nhat
    python -m src.utils.monitor_training --run model_b/20240103_120000  # Xem run cu the
    python -m src.utils.monitor_training --live             # Theo doi real-time
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

# Colors for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def find_latest_log(logs_root="logs"):
    """Tim log moi nhat."""
    logs_path = Path(logs_root)
    if not logs_path.exists():
        return None
    
    latest = None
    latest_time = 0
    
    for model_dir in logs_path.iterdir():
        if model_dir.is_dir():
            for run_dir in model_dir.iterdir():
                if run_dir.is_dir():
                    mtime = run_dir.stat().st_mtime
                    if mtime > latest_time:
                        latest_time = mtime
                        latest = run_dir
    
    return latest


def load_training_log(log_dir):
    """Load training log tu file."""
    log_file = Path(log_dir) / "training_log.json"
    if not log_file.exists():
        return None
    
    with open(log_file) as f:
        return json.load(f)


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^60}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")


def print_metrics(epoch_data, prev_data=None):
    """In metrics cho 1 epoch."""
    epoch = epoch_data.get('epoch', '?')
    
    train_loss = epoch_data.get('train_loss', 0)
    val_loss = epoch_data.get('val_loss', 0)
    bleu = epoch_data.get('bleu', 0)
    exact_match = epoch_data.get('exact_match', 0)
    edit_dist = epoch_data.get('edit_distance', 0)
    
    # Calculate changes
    def get_change(key, higher_better=False):
        if prev_data is None:
            return ""
        curr = epoch_data.get(key, 0)
        prev = prev_data.get(key, 0)
        diff = curr - prev
        if abs(diff) < 0.0001:
            return ""
        if higher_better:
            color = Colors.GREEN if diff > 0 else Colors.RED
        else:
            color = Colors.GREEN if diff < 0 else Colors.RED
        sign = "+" if diff > 0 else ""
        return f" {color}({sign}{diff:.4f}){Colors.ENDC}"
    
    print(f"{Colors.CYAN}Epoch {epoch}{Colors.ENDC}")
    print(f"  Train Loss:    {train_loss:.4f}{get_change('train_loss', False)}")
    print(f"  Val Loss:      {val_loss:.4f}{get_change('val_loss', False)}")
    print(f"  BLEU:          {bleu:.4f}{get_change('bleu', True)}")
    print(f"  Exact Match:   {exact_match:.4f}{get_change('exact_match', True)}")
    print(f"  Edit Distance: {edit_dist:.4f}{get_change('edit_distance', False)}")
    
    # Overfitting warning
    if train_loss > 0 and val_loss > 0:
        gap = val_loss - train_loss
        if gap > 0.5:
            print(f"  {Colors.YELLOW}[WARNING] Possible overfitting! Gap: {gap:.4f}{Colors.ENDC}")
    
    print()


def check_convergence(history):
    """Kiem tra da hoi tu chua."""
    if len(history) < 5:
        return False, "Chua du data (can it nhat 5 epochs)"
    
    # Lay 5 val_loss gan nhat
    recent_losses = [h.get('val_loss', 0) for h in history[-5:]]
    
    # Tinh variance
    mean_loss = sum(recent_losses) / len(recent_losses)
    variance = sum((x - mean_loss)**2 for x in recent_losses) / len(recent_losses)
    
    if variance < 0.001:
        return True, f"Val loss da on dinh (variance={variance:.6f})"
    
    # Kiem tra xu huong
    improving = all(recent_losses[i] >= recent_losses[i+1] for i in range(len(recent_losses)-1))
    if improving:
        return False, "Van dang cai thien"
    
    return False, f"Chua hoi tu (variance={variance:.4f})"


def check_overfitting(history):
    """Kiem tra overfitting."""
    if len(history) < 3:
        return False, "Chua du data"
    
    recent = history[-3:]
    
    # Train loss giam nhung val loss tang
    train_decreasing = all(
        recent[i].get('train_loss', 0) >= recent[i+1].get('train_loss', 0) 
        for i in range(len(recent)-1)
    )
    val_increasing = all(
        recent[i].get('val_loss', 0) <= recent[i+1].get('val_loss', 0) 
        for i in range(len(recent)-1)
    )
    
    if train_decreasing and val_increasing:
        return True, "Train loss giam, Val loss tang => OVERFITTING!"
    
    # Gap qua lon
    latest = history[-1]
    gap = latest.get('val_loss', 0) - latest.get('train_loss', 0)
    if gap > 1.0:
        return True, f"Gap giua train/val qua lon: {gap:.4f}"
    
    return False, "Khong co dau hieu overfitting"


def display_summary(log_data, log_dir):
    """Hien thi tong ket training."""
    history = log_data.get('history', [])
    config = log_data.get('config', {})
    
    print_header("TRAINING SUMMARY")
    
    # Config
    print(f"{Colors.BOLD}Configuration:{Colors.ENDC}")
    print(f"  Model:     {config.get('model_name', 'Unknown')}")
    print(f"  Dataset:   {config.get('dataset_type', 'Unknown')}")
    print(f"  Epochs:    {config.get('num_epochs', 'Unknown')}")
    print(f"  Batch:     {config.get('batch_size', 'Unknown')}")
    print(f"  LR:        {config.get('learning_rate', 'Unknown')}")
    print(f"  Log Dir:   {log_dir}")
    print()
    
    if not history:
        print(f"{Colors.YELLOW}Chua co training data{Colors.ENDC}")
        return
    
    # Progress
    completed = len(history)
    total = config.get('num_epochs', completed)
    progress = completed / total * 100
    
    print(f"{Colors.BOLD}Progress: {completed}/{total} epochs ({progress:.1f}%){Colors.ENDC}")
    
    # Progress bar
    bar_width = 40
    filled = int(bar_width * progress / 100)
    bar = '█' * filled + '░' * (bar_width - filled)
    print(f"  [{bar}]")
    print()
    
    # Best metrics
    best_bleu = max(h.get('bleu', 0) for h in history)
    best_loss = min(h.get('val_loss', float('inf')) for h in history)
    best_exact = max(h.get('exact_match', 0) for h in history)
    
    print(f"{Colors.BOLD}Best Metrics:{Colors.ENDC}")
    print(f"  {Colors.GREEN}Best BLEU:        {best_bleu:.4f}{Colors.ENDC}")
    print(f"  {Colors.GREEN}Best Val Loss:    {best_loss:.4f}{Colors.ENDC}")
    print(f"  {Colors.GREEN}Best Exact Match: {best_exact:.4f}{Colors.ENDC}")
    print()
    
    # Status checks
    converged, conv_msg = check_convergence(history)
    overfitting, over_msg = check_overfitting(history)
    
    print(f"{Colors.BOLD}Status:{Colors.ENDC}")
    if converged:
        print(f"  {Colors.GREEN}[CONVERGED] {conv_msg}{Colors.ENDC}")
    else:
        print(f"  {Colors.CYAN}[TRAINING] {conv_msg}{Colors.ENDC}")
    
    if overfitting:
        print(f"  {Colors.RED}[OVERFITTING] {over_msg}{Colors.ENDC}")
    else:
        print(f"  {Colors.GREEN}[OK] {over_msg}{Colors.ENDC}")
    print()
    
    # Recent epochs
    print(f"{Colors.BOLD}Recent Epochs:{Colors.ENDC}")
    print("-" * 60)
    
    recent = history[-5:] if len(history) >= 5 else history
    for i, epoch_data in enumerate(recent):
        prev = recent[i-1] if i > 0 else None
        print_metrics(epoch_data, prev)


def live_monitor(log_dir, interval=10):
    """Theo doi real-time."""
    print_header("LIVE TRAINING MONITOR")
    print(f"Monitoring: {log_dir}")
    print(f"Refresh interval: {interval}s")
    print("Press Ctrl+C to stop\n")
    
    last_epoch = 0
    
    try:
        while True:
            log_data = load_training_log(log_dir)
            if log_data:
                history = log_data.get('history', [])
                if len(history) > last_epoch:
                    # Clear screen
                    os.system('clear' if os.name == 'posix' else 'cls')
                    display_summary(log_data, log_dir)
                    last_epoch = len(history)
            
            time.sleep(interval)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Monitoring stopped.{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    parser.add_argument('--run', type=str, default=None,
                       help='Specific run to monitor (e.g., model_b/20240103_120000)')
    parser.add_argument('--live', action='store_true',
                       help='Live monitoring mode')
    parser.add_argument('--interval', type=int, default=10,
                       help='Refresh interval for live mode (seconds)')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='Logs root directory')
    
    args = parser.parse_args()
    
    # Find log directory
    if args.run:
        log_dir = Path(args.logs_dir) / args.run
    else:
        log_dir = find_latest_log(args.logs_dir)
    
    if log_dir is None or not log_dir.exists():
        print(f"{Colors.RED}[ERROR] Khong tim thay log directory{Colors.ENDC}")
        print(f"  Kiem tra: {args.logs_dir}/")
        return 1
    
    if args.live:
        live_monitor(log_dir, args.interval)
    else:
        log_data = load_training_log(log_dir)
        if log_data:
            display_summary(log_data, log_dir)
        else:
            print(f"{Colors.YELLOW}Chua co training data tai {log_dir}{Colors.ENDC}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
