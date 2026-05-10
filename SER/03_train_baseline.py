"""
03_train_baseline.py
====================
Treina classificadores clássicos (Random Forest, SVM, Gradient Boosting)
sobre o vetor de features handcrafted produzido por 02_feature_extraction.py.

Usa Leave-One-Speaker-Out CV para evitar speaker leakage (28 atores).

Saídas:
  - models/baseline_rf.joblib
  - models/baseline_svm.joblib
  - models/scaler.joblib
  - models/baseline_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).parent
FEAT_CSV = ROOT / "data" / "features" / "features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)


def load_features():
    if not FEAT_CSV.exists():
        print(f"[ERRO] {FEAT_CSV} não existe. Execute 02_feature_extraction.py.")
        sys.exit(1)
    df = pd.read_csv(FEAT_CSV)
    meta_cols = ["file", "actor", "gender", "emotion", "fatigue", "intensity"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    X = df[feature_cols].values.astype(np.float32)
    y_emo = df["emotion"].values
    y_fat = df["fatigue"].values.astype(np.int8)
    actors = df["actor"].values.astype(np.int16)
    return X, y_emo, y_fat, actors, feature_cols


def loso_cv(model_factory, X, y, groups) -> Dict:
    """Leave-One-Speaker-Out cross-validation."""
    logo = LeaveOneGroupOut()
    accs, f1s = [], []
    for train_idx, test_idx in logo.split(X, y, groups=groups):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test  = scaler.transform(X[test_idx])
        model = model_factory()
        model.fit(X_train, y[train_idx])
        preds = model.predict(X_test)
        accs.append(accuracy_score(y[test_idx], preds))
        f1s.append(f1_score(y[test_idx], preds, average="macro", zero_division=0))
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std":  float(np.std(accs)),
        "f1_mean":  float(np.mean(f1s)),
        "f1_std":   float(np.std(f1s)),
    }


def hold_out_eval(model_factory, X, y, groups) -> Dict:
    """80/20 split por speaker para report final."""
    unique_actors = np.unique(groups)
    rng = np.random.default_rng(42)
    rng.shuffle(unique_actors)
    n_test = max(1, int(0.2 * len(unique_actors)))
    test_actors = set(unique_actors[:n_test])

    test_mask = np.isin(groups, list(test_actors))
    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = model_factory()
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)

    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=sorted(np.unique(y))).tolist()
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "macro_f1": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "report": report,
        "confusion_matrix": cm,
        "labels": sorted(np.unique(y).tolist()),
        "test_actors": sorted(test_actors),
    }, model, scaler


def main() -> int:
    print("[INFO] A carregar features...")
    X, y_emo, y_fat, actors, feature_cols = load_features()
    print(f"[INFO] X shape = {X.shape}, n_actors = {len(np.unique(actors))}")

    results: Dict = {"feature_count": len(feature_cols)}

    # ---------------- Multiclasse (8 emoções) ---------------- #
    print("\n=== Tarefa: Multiclasse (8 emoções) ===")
    factories = {
        "RandomForest": lambda: RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1, random_state=42
        ),
        "SVM_RBF": lambda: SVC(C=10, kernel="rbf", gamma="scale", probability=True),
        "GradientBoost": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=42
        ),
    }
    results["multiclass"] = {}
    best_model_name, best_f1 = None, -1.0
    for name, factory in factories.items():
        print(f"\n[CV-LOSO] {name} ...")
        cv = loso_cv(factory, X, y_emo, actors)
        print(f"   acc = {cv['acc_mean']:.3f} ± {cv['acc_std']:.3f}    "
              f"macro-f1 = {cv['f1_mean']:.3f} ± {cv['f1_std']:.3f}")
        results["multiclass"][name] = cv
        if cv["f1_mean"] > best_f1:
            best_f1 = cv["f1_mean"]
            best_model_name = name

    print(f"\n[INFO] Melhor modelo: {best_model_name}. A treinar com hold-out final...")
    final_eval, model, scaler = hold_out_eval(factories[best_model_name], X, y_emo, actors)
    results["multiclass_final"] = final_eval
    print(f"   Hold-out: acc = {final_eval['accuracy']:.3f}   macro-f1 = {final_eval['macro_f1']:.3f}")

    joblib.dump(model, MODELS_DIR / f"baseline_{best_model_name.lower()}_emotion.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler_emotion.joblib")

    # ---------------- Binário (alerta vs fadigado) ---------------- #
    print("\n=== Tarefa: Binária (alerta vs fadigado) ===")
    results["binary"] = {}
    for name, factory in factories.items():
        cv = loso_cv(factory, X, y_fat, actors)
        print(f"   {name:14s}  acc = {cv['acc_mean']:.3f}   f1 = {cv['f1_mean']:.3f}")
        results["binary"][name] = cv

    # Treinar RF binário final (geralmente o mais robusto)
    final_bin, model_bin, scaler_bin = hold_out_eval(factories["RandomForest"], X, y_fat, actors)
    results["binary_final"] = final_bin
    joblib.dump(model_bin, MODELS_DIR / "baseline_rf_fatigue.joblib")
    joblib.dump(scaler_bin, MODELS_DIR / "scaler_fatigue.joblib")
    joblib.dump(feature_cols, MODELS_DIR / "feature_cols.joblib")

    # Guardar métricas
    out_json = MODELS_DIR / "baseline_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Métricas guardadas em {out_json}")
    print(f"[OK] Modelos guardados em {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
