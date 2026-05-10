"""
02_feature_extraction.py
========================
Extrai features acústicas do RAVDESS para alimentar:
  (a) classificadores clássicos  → vetor de features estatísticas (CSV)
  (b) modelos sequenciais (CNN-LSTM) → sequências temporais (NPZ)

Features extraídas (alinhadas com README do projeto: MFCCs, Pitch, Energia):
  - 40 MFCCs + Δ + ΔΔ
  - Pitch (F0) via PYIN
  - RMS Energy
  - Zero-Crossing Rate
  - Spectral centroid, rolloff, bandwidth, contrast
  - Estatísticas: média, std, min, max, skew

Privacy-by-Design: o ficheiro de saída contém APENAS features matemáticas,
nunca o sinal de áudio bruto.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.stats import skew
from tqdm import tqdm

# ---------------- Constantes ---------------- #
RAW_DIR = Path(__file__).parent / "data" / "raw"
FEAT_DIR = Path(__file__).parent / "data" / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16_000     # alinhado com Wav2Vec2/HuBERT
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 256
TARGET_SECONDS = 3.0     # janela uniforme para sequências
MAX_FRAMES = int(SAMPLE_RATE * TARGET_SECONDS / HOP_LENGTH) + 1   # ≈ 188

# RAVDESS emotion code → nome
EMOTION_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Mapeamento para classificação binária (alerta vs fadigado)
# Baseado no modelo circumplexo de Russell (low arousal → fadiga)
FATIGUE_MAP: Dict[str, int] = {
    "neutral":   1,   # 1 = fadigado (low arousal)
    "calm":      1,
    "sad":       1,
    "happy":     0,   # 0 = alerta (high arousal)
    "surprised": 0,
    "angry":     0,
    "fearful":   0,
    "disgust":   0,
}

FNAME_RE = re.compile(
    r"^(?P<mod>\d{2})-(?P<voc>\d{2})-(?P<emo>\d{2})-(?P<int>\d{2})-"
    r"(?P<stm>\d{2})-(?P<rep>\d{2})-(?P<actor>\d{2})\.wav$"
)


# ---------------- Helpers ---------------- #
@dataclass
class FileMeta:
    path: Path
    emotion: str
    intensity: str
    statement: str
    repetition: str
    actor: int
    gender: str          # "M" / "F"


def parse_filename(p: Path) -> FileMeta | None:
    m = FNAME_RE.match(p.name)
    if m is None:
        return None
    actor = int(m["actor"])
    return FileMeta(
        path=p,
        emotion=EMOTION_MAP[m["emo"]],
        intensity=m["int"],
        statement=m["stm"],
        repetition=m["rep"],
        actor=actor,
        gender="F" if actor % 2 == 0 else "M",
    )


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    # Pre-emphasis filter (alpha = 0.97) — realça frequências altas
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    # Normalização de amplitude
    peak = np.max(np.abs(y)) + 1e-9
    return y / peak


def pad_or_trim(y: np.ndarray, target_len: int) -> np.ndarray:
    if len(y) >= target_len:
        return y[:target_len]
    return np.pad(y, (0, target_len - len(y)), mode="constant")


# ---------------- Feature extractors ---------------- #
def stats_vector(x: np.ndarray, prefix: str) -> Dict[str, float]:
    """Estatísticas de um vetor de features (1D ou 2D ao longo do tempo)."""
    if x.ndim == 1:
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std":  float(np.std(x)),
            f"{prefix}_min":  float(np.min(x)),
            f"{prefix}_max":  float(np.max(x)),
            f"{prefix}_skew": float(skew(x)) if np.std(x) > 1e-9 else 0.0,
        }
    out: Dict[str, float] = {}
    for i, row in enumerate(x):
        out.update(stats_vector(row, f"{prefix}{i}"))
    return out


def extract_handcrafted(y: np.ndarray) -> Dict[str, float]:
    """Vetor de features para classificadores clássicos."""
    feats: Dict[str, float] = {}

    # MFCC + Δ + ΔΔ
    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    feats.update(stats_vector(mfcc, "mfcc"))
    feats.update(stats_vector(librosa.feature.delta(mfcc), "dmfcc"))
    feats.update(stats_vector(librosa.feature.delta(mfcc, order=2), "ddmfcc"))

    # Pitch (F0) via PYIN — robusto para fala
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"),
            sr=SAMPLE_RATE, frame_length=N_FFT, hop_length=HOP_LENGTH,
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        feats.update(stats_vector(f0, "f0"))
        feats["f0_voiced_ratio"] = float(np.mean(voiced_flag.astype(float)))
    except Exception:
        # Caso PYIN falhe em ficheiros muito curtos
        for k in ("mean", "std", "min", "max", "skew"):
            feats[f"f0_{k}"] = 0.0
        feats["f0_voiced_ratio"] = 0.0

    # Energia (RMS)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    feats.update(stats_vector(rms, "rms"))

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    feats.update(stats_vector(zcr, "zcr"))

    # Spectral
    cent = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    feats.update(stats_vector(cent, "centroid"))

    roll = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    feats.update(stats_vector(roll, "rolloff"))

    bw = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    feats.update(stats_vector(bw, "bandwidth"))

    contrast = librosa.feature.spectral_contrast(y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
    feats.update(stats_vector(contrast, "contrast"))

    return feats


def extract_sequence(y: np.ndarray) -> np.ndarray:
    """Mel-spectrogram + MFCC para CNN-LSTM. Shape: (n_features, n_frames)."""
    target_len = int(SAMPLE_RATE * TARGET_SECONDS)
    y = pad_or_trim(y, target_len)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=64
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    return log_mel  # (64, MAX_FRAMES)


# ---------------- Main pipeline ---------------- #
def collect_files() -> List[FileMeta]:
    if not RAW_DIR.exists():
        print(f"[ERRO] {RAW_DIR} não existe. Execute 01_download_ravdess.py primeiro.")
        sys.exit(1)
    files: List[FileMeta] = []
    for p in sorted(RAW_DIR.rglob("*.wav")):
        meta = parse_filename(p)
        if meta is not None:
            files.append(meta)
    return files


def main() -> int:
    files = collect_files()
    if not files:
        print("[ERRO] Nenhum ficheiro RAVDESS encontrado.")
        return 1
    print(f"[INFO] {len(files)} ficheiros encontrados.")

    rows: List[Dict] = []
    seqs: List[np.ndarray] = []
    seq_labels: List[str] = []
    seq_actors: List[int] = []
    seq_fatigue: List[int] = []

    for meta in tqdm(files, desc="Extracting features"):
        try:
            y = load_audio(meta.path)
        except Exception as e:
            print(f"[WARN] Falhou {meta.path.name}: {e}")
            continue

        # (a) Handcrafted vector
        feats = extract_handcrafted(y)
        feats.update({
            "file": meta.path.name,
            "actor": meta.actor,
            "gender": meta.gender,
            "emotion": meta.emotion,
            "fatigue": FATIGUE_MAP[meta.emotion],
            "intensity": meta.intensity,
        })
        rows.append(feats)

        # (b) Sequence (Mel-spectrogram)
        seqs.append(extract_sequence(y))
        seq_labels.append(meta.emotion)
        seq_actors.append(meta.actor)
        seq_fatigue.append(FATIGUE_MAP[meta.emotion])

    # Guardar handcrafted CSV
    df = pd.DataFrame(rows)
    out_csv = FEAT_DIR / "features.csv"
    df.to_csv(out_csv, index=False)
    print(f"[OK] Handcrafted features → {out_csv} (shape {df.shape})")

    # Guardar sequências NPZ
    seq_arr = np.stack([
        # truncar/pad ao mesmo número de frames
        s[:, :MAX_FRAMES] if s.shape[1] >= MAX_FRAMES
        else np.pad(s, ((0, 0), (0, MAX_FRAMES - s.shape[1])), mode="constant", constant_values=s.min())
        for s in seqs
    ]).astype(np.float32)

    out_npz = FEAT_DIR / "sequences.npz"
    np.savez_compressed(
        out_npz,
        X=seq_arr,
        y_emotion=np.array(seq_labels),
        y_fatigue=np.array(seq_fatigue, dtype=np.int8),
        actors=np.array(seq_actors, dtype=np.int16),
    )
    print(f"[OK] Sequences → {out_npz} (X shape {seq_arr.shape})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
