# Análise Acústica (Speech Emotion Recognition) — Componente do Detetor de Fadiga Digital

> Documento de apoio para a redação do relatório (Sensorização e Ambiente, 2025/2026)
> Conteúdo organizado segundo a estrutura LNCS @ Springer pedida no enunciado.

---

## 1. Avaliação de Viabilidade (TL;DR)

A componente de **Speech Emotion Recognition (SER)** é **perfeitamente viável** para este trabalho, e **encaixa-se bem** na arquitetura multimodal já existente (Contexto via DeBERTa + Teclado/Rato). Com prazo até 21 de maio de 2026 (≈11 dias), recomenda-se:

| Cenário | Esforço | Resultado esperado |
|---|---|---|
| **A — Apenas no relatório** (descritivo, sem implementação) | Baixo (1–2 dias) | Capítulo completo justificado teoricamente |
| **B — Baseline com dataset público** (RAVDESS + MFCC + classificador clássico) | Médio (2–3 dias) | Modelo treinado + métricas para o relatório |
| **C — Modelo pré-treinado (Wav2Vec2)** | Médio-alto (3–5 dias) | Resultados estado-da-arte com pouca implementação |
| **D — Aquisição própria + integração no pipeline** | Alto (>5 dias) | Sistema integrado completo (ideal mas arriscado) |

**Recomendação realista:** **Cenário B + secção do relatório a discutir B/C/D**. Permite mostrar resultados reais sem comprometer a entrega das outras componentes.

---

## 2. Contextualização no Projeto

O *Digital Fatigue Detector* já trata duas modalidades:

1. **Contexto semântico** — texto/conteúdo que o utilizador consome (modelo DeBERTa em `modelo_deberta.ipynb`).
2. **Comportamento motor** — padrões de teclado e rato (`dataset_emocoes_*`, `mouse2vec.py`).

A **voz** é uma terceira modalidade altamente informativa: a fadiga manifesta-se acusticamente em alterações de prosódia (descida do *pitch*, redução de variabilidade, articulação mais lenta, pausas mais longas, redução de energia). A literatura confirma correlações fortes entre características vocais e estados de fadiga, *stress* e carga cognitiva (Krajewski et al., 2009; Schuller et al., 2014).

Adicionar SER ao sistema permite:

- **Fusão multimodal** (texto + comportamento + voz) — robustez e generalização.
- **Cobertura de momentos sem teclado/rato** — videoconferências, ditados, comandos de voz.
- **Validação cruzada** das predições das outras modalidades.

---

## 3. Domínio e Objetivos (para a secção *Introduction* do relatório)

**Problema:** A fadiga digital é multifatorial e dificilmente detetável por uma única modalidade. Sinais comportamentais (teclado/rato) podem falhar em tarefas que envolvem fala (reuniões, dictado, *brainstorming* por voz).

**Objetivo desta componente:** Conceber e avaliar um módulo de classificação automática do estado emocional/fadiga do utilizador a partir de sinal acústico capturado pelo microfone, integrável no detetor de fadiga digital.

**Hipótese:** Características prosódicas e espectrais da fala contêm informação discriminativa sobre o estado de fadiga e podem complementar (ou substituir, em períodos de silêncio comportamental) os outros sensores.

---

## 4. Sensor e Coletor de Dados

### 4.1 Sensor

- **Microfone do dispositivo** (built-in laptop / *headset*).
- Frequência de amostragem recomendada: **16 kHz** (compatível com a maioria dos modelos pré-treinados — Wav2Vec2/HuBERT).
- Mono channel, 16-bit PCM.

### 4.2 Pipeline de Aquisição

```
[Mic] → [VAD] → [Segmentação 2–5s] → [Pré-processamento] → [Features] → [Modelo] → [Predição]
                                            ↓
                                     [Fusão tardia com Contexto + Teclado/Rato]
```

Componentes essenciais:

1. **Voice Activity Detection (VAD)** — apenas processar quando há fala (`webrtcvad` ou `silero-vad`). Evita gravar silêncio e privacidade desnecessária.
2. **Segmentação** — janelas de 2–5 s com sobreposição de 50%.
3. **Pré-processamento** — normalização de amplitude, *pre-emphasis filter* (α=0.97), remoção de DC offset, redução de ruído (opcional, `noisereduce`).
4. **Anonimização** — apenas armazenar *features* extraídas (MFCCs, embeddings), nunca o áudio bruto. **Crucial para questões éticas/RGPD.**

### 4.3 Implementação Sugerida

Bibliotecas Python:

- `sounddevice` ou `pyaudio` — captura em tempo real.
- `librosa` — *feature extraction*.
- `torch` + `transformers` — modelos pré-treinados.
- `webrtcvad` — VAD.

---

## 5. Datasets Públicos (para treino/avaliação)

| Dataset | Descrição | Emoções | N amostras |
|---|---|---|---|
| **RAVDESS** | 24 atores, atuação | 8 (incl. neutral, calm, sad, fearful) | 1440 |
| **CREMA-D** | 91 atores, multi-étnico | 6 | 7442 |
| **TESS** | 2 atrizes, palavras isoladas | 7 | 2800 |
| **SAVEE** | 4 atores britânicos | 7 | 480 |
| **EMO-DB** | Alemão, 10 atores | 7 | 535 |
| **IEMOCAP** | Diálogos espontâneos | 10 | ~12h |

**Recomendação:** Começar por **RAVDESS** (bem rotulado, balanceado, tamanho gerível) e/ou **CREMA-D** (mais diversidade). Para treino realista, considerar **fusão de datasets** com mapeamento de etiquetas.

**Mapeamento para fadiga:** As emoções *sad*, *neutral*, *calm*, *bored*, *tired* costumam associar-se a estados de baixa ativação ↔ fadiga. *Happy*, *angry*, *excited* a alta ativação. Pode-se fazer um modelo binário (alerta vs. fadigado) ou multiclasse (Russell's circumplex: *valence × arousal*).

---

## 6. Características Acústicas (Feature Engineering)

### 6.1 *Handcrafted features* (baseline clássico)

- **MFCCs** (13–40 coeficientes) + Δ + ΔΔ — espectro percetual.
- **F0 (pitch)** média, desvio-padrão, contorno — entoação.
- **Energy / RMS** — intensidade vocal.
- **Spectral centroid, rolloff, flux, bandwidth** — timbre.
- **Zero-Crossing Rate**.
- **Jitter & Shimmer** — qualidade vocal (correlacionados com fadiga).
- **Speaking rate / pause statistics** — ritmo (correlato direto de fadiga cognitiva).

→ Conjunto **eGeMAPS** (88 features) é o standard e está em `opensmile`.

### 6.2 *Deep features*

- **Mel-spectrogram** (input para CNNs).
- **Embeddings Wav2Vec2 / HuBERT / WavLM** (768 ou 1024 dim).

---

## 7. Modelos a Considerar

### 7.1 Baseline clássico (recomendado começar aqui)

- **Random Forest / XGBoost / SVM** sobre features eGeMAPS ou MFCC.
- Vantagem: rápido, interpretável, paralelizável com o pipeline atual.
- Espera-se ~60–70% accuracy em RAVDESS multiclasse.

### 7.2 Deep Learning *from scratch*

- **CNN-2D sobre Mel-spectrogram** (Resnet-style).
- **CNN-LSTM** ou **Transformer encoder** para capturar dinâmica temporal.

### 7.3 Transfer Learning (recomendado para resultados)

- **Wav2Vec2** pré-treinado (e.g. `facebook/wav2vec2-base`) + classificador.
- **HuBERT-base** + linear head.
- Modelos *já fine-tuned* em SER no HuggingFace Hub:
  - `superb/wav2vec2-base-superb-er`
  - `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
  - `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`

→ **Caminho mais eficiente** para obter ≥75% accuracy em RAVDESS.

### 7.4 Comparação Sugerida no Relatório

Treinar 2–3 modelos e comparar:

| Modelo | Features | Accuracy esperada (RAVDESS) | Tempo treino |
|---|---|---|---|
| Random Forest | eGeMAPS | ~60% | minutos |
| CNN | Mel-spec | ~70% | dezenas min |
| Wav2Vec2 fine-tuned | Raw audio | ~80–85% | 1–2h GPU |

---

## 8. Integração no Sistema Multimodal

A arquitetura final pode usar **fusão tardia** (decisão), que é a mais simples:

```
Contexto (DeBERTa) ─→ P(fadiga | contexto)
Teclado/Rato      ─→ P(fadiga | comportamento)  ─→  [Combinador]  ─→  Predição final
SER (Wav2Vec2)    ─→ P(fadiga | voz)                  ↑
                                                pesos aprendidos
                                              (média ponderada,
                                               regressão logística,
                                               ou stacking)
```

Alternativas:

- **Fusão precoce** (concatenar embeddings dos 3 modelos antes do classificador final).
- **Fusão híbrida / *attention-based***.
- **Voto maioritário** (mais simples e mais explicável).

---

## 9. Visualização de Dados (secção do relatório)

Sugestões para gráficos a incluir:

- *Waveform* + *Mel-spectrogram* de exemplos por emoção.
- Distribuição de F0 e energia por classe (boxplots).
- t-SNE / UMAP dos embeddings Wav2Vec2 colorido por emoção.
- Matriz de confusão dos modelos.
- Comparação de accuracy por modelo (barplot).
- Curva ROC multiclasse (one-vs-rest).
- Importância de features (Random Forest).

---

## 10. Métricas e Avaliação

- **Accuracy, Macro-F1, Weighted-F1** (datasets desbalanceados).
- **Matriz de confusão por emoção**.
- **Cross-validation por *speaker*** (Leave-One-Speaker-Out) — crítico para evitar *leakage* (o modelo aprender a reconhecer pessoas em vez de emoções).
- **Cross-corpus evaluation** — treinar em RAVDESS, testar em CREMA-D, para mostrar generalização.
- **Métricas de fadiga**: se for binário (alerta/fadigado), incluir precision/recall específicos.

---

## 11. Análise Crítica e Limitações (secção essencial do relatório)

1. **Privacidade**: A captura contínua de áudio é altamente sensível. O sistema deve operar *on-device*, sem armazenamento de áudio bruto, com VAD para minimizar gravação.
2. **Generalização entre falantes**: Modelos treinados em datasets atuados sofrem de *domain gap* face a fala espontânea.
3. **Variabilidade individual**: O *baseline* vocal de cada pessoa difere — necessária calibração inicial ou *speaker-adaptive* fine-tuning.
4. **Ruído ambiente**: Cenários reais (escritório, café) introduzem ruído que degrada modelos treinados em condições limpas.
5. **Diferença entre emoção e fadiga**: SER deteta emoção; fadiga é um construto mais complexo (cognitivo + emocional + físico). A literatura suporta a correlação mas não a equivalência.
6. **Latência**: Janelas de 2–5 s introduzem atraso na deteção em tempo real.
7. **Língua**: Maioria dos datasets é em inglês; usar em PT pode requerer fine-tuning ou modelos multilingues (XLS-R).

---

## 12. Sugestões e Trabalho Futuro

- **Fine-tuning *speaker-specific*** — calibração por utilizador no primeiro uso.
- **Multimodalidade com vídeo** — combinar voz com expressão facial (FER).
- **Aprendizagem contínua** com *feedback* explícito do utilizador.
- **Abordagem dimensional** (valência/arousal) em vez de classes discretas.
- **Modelos *self-supervised*** mais recentes (WavLM-Large, EmoTale).
- **Privacy-preserving inference** com *federated learning*.

---

## 13. Plano de Implementação Sugerido (se optares pelo Cenário B/C)

**Dia 1**
- Carregar RAVDESS, EDA básica.
- Pipeline `librosa` para extrair MFCCs + features eGeMAPS.

**Dia 2**
- Treinar baseline (Random Forest + XGBoost).
- Métricas + matriz de confusão.

**Dia 3**
- Adicionar Wav2Vec2 pré-treinado como *feature extractor* + classificador linear.
- Comparar com baseline.

**Dia 4**
- Captura simples via microfone (script `sounddevice`) + inferência em tempo real.
- Visualizações para o relatório.

**Dia 5**
- Integração com o módulo de fusão (predição combinada com Contexto + Teclado/Rato).

**Dia 6+**
- Escrita da secção do relatório, *peer-review* interno, ajustes.

---

## 14. Bibliografia Recomendada (para o relatório)

- Schuller, B., Steidl, S., Batliner, A. (2014). *The INTERSPEECH 2014 Computational Paralinguistics Challenge: Cognitive & Physical Load*.
- Krajewski, J., Batliner, A., Golz, M. (2009). *Acoustic sleepiness detection*.
- Eyben, F., Scherer, K. R., Schuller, B., et al. (2016). *The Geneva Minimalistic Acoustic Parameter Set (eGeMAPS)*.
- Baevski, A. et al. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*.
- Pepino, L. et al. (2021). *Emotion Recognition from Speech Using wav2vec 2.0 Embeddings*. INTERSPEECH.
- Livingstone, S. R., Russo, F. A. (2018). *The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)*.
- Russell, J. A. (1980). *A circumplex model of affect*.

---

## 15. Conclusão

A inclusão da Análise Acústica via SER é tecnicamente exequível e cientificamente bem fundamentada. A combinação com as restantes modalidades reforça a robustez do sistema e cobre cenários onde teclado/rato não são informativos (reuniões, ditados). A recomendação é seguir o **Cenário B + C** (baseline + transfer learning com Wav2Vec2) e dedicar a secção do relatório à descrição teórica completa, comparação de modelos, e análise crítica das limitações de privacidade e generalização.

Se o tempo apertar, **mesmo o Cenário A (apenas no relatório, sem implementação)** é defensável: pode ser apresentado como "trabalho futuro a integrar na arquitetura multimodal", desde que justificado pela complexidade da aquisição de áudio e questões de privacidade.
