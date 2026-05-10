"""
05_evaluate.py
==============
Produz visualizações e relatórios para o relatório:
  - Matriz de confusão (baseline + CNN-LSTM, multiclasse + binário)
  - Comparação de modelos (barplot)
  - Distribuição de features por emoção (boxplots)
  - t-SNE/UMAP do espaço de features
  - Curvas de aprendizagem (CNN-LSTM)

Saídas em: reports/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
FEAT_CSV = ROOT / "data" / "features" / "features.csv"
SEQ_NPZ = ROOT / "data" / "features" / "sequences.npz"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def plot_confusion(cm: np.ndarray, labels: list, title: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax, cbar_kws={"label": "row-normalized"})
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  [OK] {out.name}")


def confusion_matrices() -> None:
    # Baseline
    bp = MODELS_DIR / "baseline_metrics.json"
    if bp.exists():
        with open(bp) as f:
            m = json.load(f)
        if "multiclass_final" in m:
            d = m["multiclass_final"]
            plot_confusion(np.array(d["confusion_matrix"]), d["labels"],
                           "Baseline (best) — Multiclass", REPORTS_DIR / "cm_baseline_multiclass.png")
        if "binary_final" in m:
            d = m["binary_final"]
            plot_confusion(np.array(d["confusion_matrix"]),
                           ["alerta", "fadigado"],
                           "Baseline RF — Binário (alerta/fadigado)",
                           REPORTS_DIR / "cm_baseline_binary.png")

    # CNN-LSTM
    cp = MODELS_DIR / "cnn_lstm_metrics.json"
    if cp.exists():
        with open(cp) as f:
            m = json.load(f)
        if "emotion" in m:
            d = m["emotion"]
            plot_confusion(np.array(d["confusion_matrix"]), d["classes"],
                           "CNN-LSTM — Multiclass (8 emoções)", REPORTS_DIR / "cm_cnnlstm_multiclass.png")
        if "fatigue" in m:
            d = m["fatigue"]
            plot_confusion(np.array(d["confusion_matrix"]),
                           ["alerta", "fadigado"],
                           "CNN-LSTM — Binário (alerta/fadigado)",
                           REPORTS_DIR / "cm_cnnlstm_binary.png")


def model_comparison() -> None:
    rows = []
    bp = MODELS_DIR / "baseline_metrics.json"
    if bp.exists():
        with open(bp) as f:
            m = json.load(f)
        for name, d in m.get("multiclass", {}).items():
            rows.append({"task": "multiclass", "model": name,
                         "acc": d["acc_mean"], "f1": d["f1_mean"]})
        for name, d in m.get("binary", {}).items():
            rows.append({"task": "binary", "model": name,
                         "acc": d["acc_mean"], "f1": d["f1_mean"]})

    cp = MODELS_DIR / "cnn_lstm_metrics.json"
    if cp.exists():
        with open(cp) as f:
            m = json.load(f)
        if "emotion" in m:
            rows.append({"task": "multiclass", "model": "CNN-LSTM",
                         "acc": m["emotion"]["final_acc"], "f1": m["emotion"]["final_f1"]})
        if "fatigue" in m:
            rows.append({"task": "binary", "model": "CNN-LSTM",
                         "acc": m["fatigue"]["final_acc"], "f1": m["fatigue"]["final_f1"]})

    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["multiclass", "binary"]):
        sub = df[df["task"] == task].melt(id_vars=["model"], value_vars=["acc", "f1"],
                                          var_name="metric", value_name="score")
        sns.barplot(data=sub, x="model", y="score", hue="metric", ax=ax)
        ax.set_title(f"Comparação de modelos — {task}")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)
    print("  [OK] model_comparison.png")


def feature_boxplots() -> None:
    if not FEAT_CSV.exists():
        return
    df = pd.read_csv(FEAT_CSV)
    keys = ["f0_mean", "rms_mean", "centroid_mean", "f0_voiced_ratio"]
    keys = [k for k in keys if k in df.columns]
    if not keys:
        return
    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 5))
    if len(keys) == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        sns.boxplot(data=df, x="emotion", y=k, ax=ax,
                    order=["neutral", "calm", "sad", "happy", "surprised", "angry", "fearful", "disgust"])
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(k)
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_boxplots.png", dpi=150)
    plt.close(fig)
    print("  [OK] feature_boxplots.png")


def tsne_features() -> None:
    if not FEAT_CSV.exists():
        return
    df = pd.read_csv(FEAT_CSV)
    meta_cols = ["file", "actor", "gender", "emotion", "fatigue", "intensity"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    X = StandardScaler().fit_transform(df[feat_cols].values)
    print("  [INFO] A computar t-SNE (pode demorar 1–2 min)...")
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X)
    plot_df = pd.DataFrame({"x": Z[:, 0], "y": Z[:, 1], "emotion": df["emotion"]})
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="emotion", ax=ax,
                    palette="tab10", s=30, alpha=0.85)
    ax.set_title("t-SNE — espaço de features handcrafted (RAVDESS)")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "tsne_features.png", dpi=150)
    plt.close(fig)
    print("  [OK] tsne_features.png")


def learning_curves() -> None:
    cp = MODELS_DIR / "cnn_lstm_metrics.json"
    if not cp.exists():
        return
    with open(cp) as f:
        m = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, task in zip(axes, ["emotion", "fatigue"]):
        if task not in m:
            continue
        h = pd.DataFrame(m[task]["history"])
        ax.plot(h["epoch"], h["train_loss"], label="train loss")
        ax2 = ax.twinx()
        ax2.plot(h["epoch"], h["test_acc"], color="tab:green", label="test acc")
        ax2.plot(h["epoch"], h["test_f1"], color="tab:orange", label="test f1")
        ax.set_title(f"CNN-LSTM — {task}")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("score")
        ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(REPORTS_DIR / "learning_curves.png", dpi=150)
    plt.close(fig)
    print("  [OK] learning_curves.png")


def main() -> int:
    print("[INFO] A gerar visualizações em", REPORTS_DIR)
    confusion_matrices()
    model_comparison()
    feature_boxplots()
    tsne_features()
    learning_curves()
    print("\n[OK] Concluído. Veja a pasta reports/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
