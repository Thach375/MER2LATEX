# Huong dan theo doi Training

## 1. Wandb (Khuyen dung)

### Setup Wandb (chi can lam 1 lan)

```bash
# Cai dat
pip install wandb

# Dang nhap (can tao account tai wandb.ai)
wandb login
# Nhap API key tu: https://wandb.ai/authorize
```

### Xem training tren Wandb

1. Truy cap: https://wandb.ai/YOUR_USERNAME/MER2LATEX
2. Chon run dang chay
3. Xem cac charts:
   - `train/loss` - Training loss theo step
   - `val/loss` - Validation loss theo epoch
   - `val/bleu` - BLEU score theo epoch
   - `val/exact_match` - Exact match rate

### Cac bieu do quan trong

| Chart | Y nghia | Mong doi |
|-------|---------|----------|
| `train/loss` | Loss khi train | Giam dan, on dinh |
| `val/loss` | Loss tren val set | Giam dan, khong tang lai |
| `val/bleu` | BLEU score | Tang dan, cang cao cang tot |
| `train/loss` vs `val/loss` | So sanh | Gap nho = tot |

---

## 2. Terminal Monitor (Offline)

```bash
# Xem training moi nhat
python -m src.utils.monitor_training

# Xem training cu the
python -m src.utils.monitor_training --run model_b/20240103_120000

# Theo doi real-time (cap nhat moi 10s)
python -m src.utils.monitor_training --live

# Thay doi tan suat cap nhat
python -m src.utils.monitor_training --live --interval 30
```

---

## 3. Truc tiep tu Terminal

Khi train, terminal se in:

```
==========================================
Epoch 5/30 Summary
==========================================
  Train Loss:    0.8234
  Val Loss:      0.9123
  Exact Match:   0.3245
  BLEU:          0.5678
  Edit Distance: 12.34
  Train Time:    245.3s
  Val Time:      45.2s
```

---

## 4. Log Files

Logs duoc luu tai: `logs/{model_name}/{timestamp}/`

```
logs/
└── model_b/
    └── 20240103_120000/
        ├── training_log.json    # Full training history
        ├── config.json          # Training config
        └── metrics.csv          # Metrics theo epoch
```

### Doc log file

```python
import json

with open('logs/model_b/20240103_120000/training_log.json') as f:
    data = json.load(f)

# Xem history
for epoch in data['history']:
    print(f"Epoch {epoch['epoch']}: BLEU={epoch['bleu']:.4f}")
```

---

## 5. Dau hieu Hoi tu (Convergence)

**Model da hoi tu khi:**
- Val loss khong giam them sau 3-5 epochs
- BLEU score on dinh
- Train loss va Val loss gap nho (<0.2)

**Chua hoi tu:**
- Val loss van dang giam
- BLEU van tang
- Gap train/val lon

---

## 6. Dau hieu Overfitting

**Overfitting khi:**
- Train loss GIAM nhung Val loss TANG
- Gap giua train/val > 0.5
- BLEU bat dau giam

**Cach xu ly:**
1. Giam epochs
2. Tang dropout
3. Data augmentation
4. Dung Early Stopping

---

## 7. Kiem tra nhanh

```bash
# Xem checkpoint moi nhat
ls -la checkpoints/model_b/*/

# Xem best model
cat checkpoints/model_b/*/checkpoint_info.json

# Xem GPU usage
nvidia-smi

# Xem training process
ps aux | grep python
```

---

## 8. Wandb Alerts (Optional)

Setup alert khi training xong hoac gap van de:

```python
# Trong code training
import wandb
wandb.alert(
    title="Training Complete",
    text=f"Best BLEU: {best_bleu:.4f}"
)
```

Hoac setup tren wandb.ai:
1. Vao Project Settings
2. Alerts > Create Alert
3. Chon dieu kien (e.g., run finished, metric threshold)
