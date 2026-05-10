"""
04_train_cnn_lstm.py
====================
Modelo principal alinhado com a arquitetura do projeto: CNN + LSTM híbrida
sobre Mel-spectrograms. Inspirado em Zhao et al. (2019) e similares ao que
é usado nas componentes de Teclado/Rato e Contexto deste projeto.

Arquitetura:
   Mel-spectrogram (1, 64, 188)
        │
   ┌────▼─────────────────┐
   │  CNN block 1: Conv2D 32 → BN → ReLU → MaxPool
   │  CNN block 2: Conv2D 64 → BN → ReLU → MaxPool
   │  CNN block 3: Conv2D 128 → BN → ReLU → MaxPool
   └────┬─────────────────┘
        │ reshape → (T, F)
   ┌────▼────────────────┐
   │  LSTM bidireccional × 2 (hidden=128)
   │  Attention pooling
   └────┬────────────────┘
        │
   ┌────▼─────┐
   │ FC 128   │
   │ Dropout  │
   │ FC n_cls │
   └──────────┘

Treina em PyTorch. Hold-out por speaker (test = 20% atores random seed 42).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).parent
SEQ_NPZ = ROOT / "data" / "features" / "sequences.npz"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# Hiperparâmetros
EPOCHS = 60
BATCH_SIZE = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42


# ---------------- Dataset ---------------- #
class SERDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, train: bool = False):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.train = train

    def __len__(self):
        return len(self.X)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """SpecAugment leve (frequency + time masking)."""
        x = x.copy()
        n_mels, n_frames = x.shape
        # Frequency mask
        f = np.random.randint(0, 12)
        f0 = np.random.randint(0, max(1, n_mels - f))
        x[f0:f0 + f, :] = x.min()
        # Time mask
        t = np.random.randint(0, 20)
        t0 = np.random.randint(0, max(1, n_frames - t))
        x[:, t0:t0 + t] = x.min()
        return x

    def __getitem__(self, idx: int):
        x = self.X[idx]
        if self.train and np.random.rand() < 0.5:
            x = self._augment(x)
        return torch.from_numpy(x).unsqueeze(0), self.y[idx]


# ---------------- Modelo ---------------- #
class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Linear(dim, 1)

    def forward(self, x):  # x: (B, T, D)
        attn = torch.softmax(self.W(x).squeeze(-1), dim=1)  # (B, T)
        return (x * attn.unsqueeze(-1)).sum(dim=1)            # (B, D)


class CNNLSTM(nn.Module):
    def __init__(self, n_classes: int, n_mels: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Dropout2d(0.3),
        )
        # Após 3× pool 2×2, n_mels=64 → 8 ; channels=128
        self.cnn_out_dim = 128 * (n_mels // 8)
        self.lstm = nn.LSTM(
            input_size=self.cnn_out_dim,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attn = AttentionPool(256)
        self.head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):                     # x: (B, 1, n_mels, T)
        z = self.cnn(x)                       # (B, C, M', T')
        B, C, M, T = z.shape
        z = z.permute(0, 3, 1, 2).reshape(B, T, C * M)  # (B, T, C*M)
        z, _ = self.lstm(z)                   # (B, T, 256)
        z = self.attn(z)                      # (B, 256)
        return self.head(z)


# ---------------- Treino / Avaliação ---------------- #
def split_by_speaker(actors: np.ndarray, test_ratio: float = 0.2):
    rng = np.random.default_rng(SEED)
    unique = np.unique(actors)
    rng.shuffle(unique)
    n_test = max(1, int(test_ratio * len(unique)))
    test_actors = set(unique[:n_test].tolist())
    test_mask = np.isin(actors, list(test_actors))
    return ~test_mask, test_mask, sorted(test_actors)


def make_loaders(X, y, actors) -> Tuple[DataLoader, DataLoader, list]:
    train_mask, test_mask, test_actors = split_by_speaker(actors)
    print(f"[INFO] Test actors: {test_actors}")
    print(f"[INFO] Train: {train_mask.sum()}   Test: {test_mask.sum()}")

    # Normalização global computada apenas no treino
    mean = X[train_mask].mean()
    std = X[train_mask].std() + 1e-8
    X_norm = (X - mean) / std

    # Class-balanced sampler
    class_counts = np.bincount(y[train_mask])
    weights = 1.0 / class_counts[y[train_mask]]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = SERDataset(X_norm[train_mask], y[train_mask], train=True)
    test_ds  = SERDataset(X_norm[test_mask],  y[test_mask],  train=False)

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0),
        test_actors,
        (mean, std),
    )


def train_one(task: str, X, y_str, actors) -> dict:
    print(f"\n========== A treinar tarefa: {task} ==========")
    le = LabelEncoder()
    y = le.fit_transform(y_str)
    n_classes = len(le.classes_)
    print(f"[INFO] Classes: {list(le.classes_)}")

    train_dl, test_dl, test_actors, (mean, std) = make_loaders(X, y, actors)

    model = CNNLSTM(n_classes=n_classes).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_f1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += loss.item() * xb.size(0); n += xb.size(0)
        scheduler.step()

        # Eval
        model.eval()
        all_p, all_y = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                xb = xb.to(DEVICE)
                logits = model(xb)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_p.extend(preds); all_y.extend(yb.numpy())
        acc = accuracy_score(all_y, all_p)
        f1 = f1_score(all_y, all_p, average="macro", zero_division=0)
        history.append({"epoch": epoch, "train_loss": loss_sum / n, "test_acc": acc, "test_f1": f1})
        print(f"  epoch {epoch:3d}/{EPOCHS}  loss={loss_sum/n:.4f}  test_acc={acc:.3f}  test_f1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restaurar melhor estado
    if best_state is not None:
        model.load_state_dict(best_state)

    # Métricas finais
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            xb = xb.to(DEVICE)
            preds = model(xb).argmax(dim=1).cpu().numpy()
            all_p.extend(preds); all_y.extend(yb.numpy())
    report = classification_report(all_y, all_p, target_names=list(le.classes_),
                                   output_dict=True, zero_division=0)
    cm = confusion_matrix(all_y, all_p).tolist()

    # Guardar
    ckpt = {
        "state_dict": model.state_dict(),
        "label_encoder_classes": le.classes_.tolist(),
        "norm_mean": float(mean),
        "norm_std": float(std),
        "task": task,
    }
    torch.save(ckpt, MODELS_DIR / f"cnn_lstm_{task}.pt")

    return {
        "task": task,
        "best_f1": best_f1,
        "final_acc": accuracy_score(all_y, all_p),
        "final_f1":  f1_score(all_y, all_p, average="macro", zero_division=0),
        "report": report,
        "confusion_matrix": cm,
        "classes": list(le.classes_),
        "test_actors": test_actors,
        "history": history,
    }


def main() -> int:
    if not SEQ_NPZ.exists():
        print(f"[ERRO] {SEQ_NPZ} não existe. Execute 02_feature_extraction.py.")
        return 1
    npz = np.load(SEQ_NPZ, allow_pickle=False)
    X = npz["X"]
    y_emotion = npz["y_emotion"]
    y_fatigue = npz["y_fatigue"]
    actors = npz["actors"]
    print(f"[INFO] X shape = {X.shape}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    res_emo = train_one("emotion", X, y_emotion, actors)
    res_fat = train_one("fatigue", X, y_fatigue.astype(str), actors)

    out = {"emotion": res_emo, "fatigue": res_fat}
    with open(MODELS_DIR / "cnn_lstm_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n[OK] Modelos e métricas guardados em {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
