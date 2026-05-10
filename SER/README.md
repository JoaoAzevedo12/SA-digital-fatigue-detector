# Speech Emotion Recognition (SER) — Componente Acústica

Componente de Análise Acústica do projeto **Biometria Comportamental e IA na Prevenção de Exaustão Digital** (Sensorização e Ambiente, UMinho 2025/2026).

## Objetivo

Detetar estados emocionais correlacionados com fadiga digital a partir do sinal de voz, alinhado com a arquitetura **CNN + LSTM** do resto do sistema e respeitando os princípios de **Privacy-by-Design** (apenas features matemáticas — MFCCs, pitch, energia — nunca áudio bruto persistido).

## Estrutura

```
SER/
├── README.md                   # este ficheiro
├── requirements.txt            # dependências Python
├── 01_download_ravdess.py      # download do dataset RAVDESS
├── 02_feature_extraction.py    # extração de features (MFCC, prosódia, espectro)
├── 03_train_baseline.py        # baseline: Random Forest + SVM
├── 04_train_cnn_lstm.py        # modelo principal: CNN + LSTM (alinhado com projeto)
├── 05_evaluate.py              # matriz de confusão, métricas, plots
├── 06_realtime_inference.py    # captura microfone + predição em tempo real
└── data/                       # criada automaticamente
    ├── raw/                    # RAVDESS .wav
    └── features/               # features extraídas (.npy / .csv)
```

## Pipeline

```
[RAVDESS .wav] ──► [VAD + Pre-emphasis] ──► [MFCC + Δ + ΔΔ + Prosódia]
                                                       │
                                                       ▼
                            ┌──────────────┬──────────────────┐
                            │   Baseline   │     CNN-LSTM      │
                            │ RandomForest │ (modelo principal)│
                            └──────────────┴──────────────────┘
                                                       │
                                                       ▼
                                              [P(fadiga | voz)]
                                                       │
                                          [Fusão tardia com Contexto + Teclado/Rato]
```

## Como executar

### 1. Instalar dependências

```bash
cd SER
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

### 2. Descarregar dataset (uma vez)

```bash
python 01_download_ravdess.py
```

Faz download do RAVDESS Audio Speech (~200 MB) para `data/raw/`.
Fonte: <https://zenodo.org/record/1188976>

### 3. Extrair features

```bash
python 02_feature_extraction.py
```

Gera `data/features/features.csv` (vetor por ficheiro) e `data/features/sequences.npz` (sequências para CNN-LSTM).

### 4. Treinar modelos

```bash
python 03_train_baseline.py     # Random Forest + SVM (rápido, minutos)
python 04_train_cnn_lstm.py     # CNN-LSTM (mais lento, ~30 min CPU / 5 min GPU)
```

Os modelos são guardados em `models/`.

### 5. Avaliação

```bash
python 05_evaluate.py
```

Produz:
- Matriz de confusão (PNG)
- Métricas por classe (CSV)
- Comparação de modelos (gráfico)
- Embeddings t-SNE (PNG)

### 6. Inferência em tempo real (opcional)

```bash
python 06_realtime_inference.py
```

Captura áudio do microfone, segmenta com VAD, e prevê emoção em janelas de 3 s.

## Mapeamento de emoções → fadiga

RAVDESS contém 8 emoções. Para o classificador binário de fadiga:

| Emoção RAVDESS | Estado projetado |
|---|---|
| neutral, calm, sad | low arousal → **fadigado / desligado** |
| happy, surprised, angry, fearful | high arousal → **alerta** |
| disgust | **alerta** (alta intensidade) |

Os modelos suportam ambos os modos: classificação multiclasse (8 emoções) e binária (alerta/fadigado).

## Privacy-by-Design

Em produção (integração com o sistema principal):
- Áudio **nunca é guardado** — apenas as features extraídas em RAM.
- VAD com `webrtcvad` filtra silêncio (não processa quando o utilizador não fala).
- Inferência *on-device* (Edge AI) — modelo TFLite/ONNX otimizado.
- Possibilidade de **Federated Learning** para *fine-tuning* personalizado por utilizador sem partilha do áudio.

## Métricas esperadas (RAVDESS)

| Modelo | Accuracy | Macro-F1 | Tempo treino |
|---|---|---|---|
| SVM (MFCC) | ~55% | 0.53 | <1 min |
| Random Forest (eGeMAPS-like) | ~62% | 0.60 | ~2 min |
| CNN-LSTM (Mel-spec) | **~75–80%** | **0.74** | ~30 min CPU |

Cross-validation: **Leave-One-Speaker-Out** (24 folds) para evitar *speaker leakage*.

## Bibliografia chave

- Livingstone, S. R., Russo, F. A. (2018). *RAVDESS*. PLoS ONE.
- Eyben, F. et al. (2016). *eGeMAPS*. IEEE Trans. Affective Computing.
- Zhao, J. et al. (2019). *Speech emotion recognition using deep 1D & 2D CNN LSTM networks*.
- Schuller, B. et al. (2014). *INTERSPEECH ComParE Cognitive & Physical Load Challenge*.

## Autor

**Vicente Castro** (PG60395) — Componente de Análise Acústica.
