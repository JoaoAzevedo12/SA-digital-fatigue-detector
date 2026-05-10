"""
06_realtime_inference.py
========================
Captura áudio do microfone, segmenta com VAD (webrtcvad), extrai features
e prevê a emoção/fadiga com o modelo CNN-LSTM treinado.

Privacy-by-Design (alinhado com o README do projeto):
  - O áudio NUNCA é guardado em disco.
  - Apenas as predições agregadas (probabilidades) são logged.
  - Pode parar imediatamente com Ctrl+C.

Uso:
  python 06_realtime_inference.py --task emotion
  python 06_realtime_inference.py --task fatigue
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Imports protegidos para falhar com mensagem amigável
try:
    import sounddevice as sd
except Exception as e:
    print("[ERRO] 'sounddevice' não instalado ou sem backend de áudio.", e)
    sys.exit(1)

try:
    import webrtcvad
except Exception:
    webrtcvad = None
    print("[AVISO] webrtcvad não disponível; a usar threshold de energia simples.")

import librosa

# Importar arquitetura do training script
sys.path.insert(0, str(Path(__file__).parent))
from importlib import import_module
cnn_lstm = import_module("04_train_cnn_lstm")  # type: ignore

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

SAMPLE_RATE = 16_000
FRAME_MS = 30                          # webrtcvad suporta 10/20/30 ms
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
WINDOW_S = 3.0                         # janela de inferência
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_S)
SLIDE_S = 1.0                          # passo entre inferências
SLIDE_SAMPLES = int(SAMPLE_RATE * SLIDE_S)


def load_model(task: str):
    ckpt_path = MODELS_DIR / f"cnn_lstm_{task}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Modelo {ckpt_path} não encontrado. Treine primeiro com 04_train_cnn_lstm.py.")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    classes = ckpt["label_encoder_classes"]
    model = cnn_lstm.CNNLSTM(n_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, classes, ckpt["norm_mean"], ckpt["norm_std"]


def has_voice(frame_int16: np.ndarray, vad) -> bool:
    if vad is None:
        return float(np.sqrt(np.mean(frame_int16.astype(np.float32) ** 2))) > 300.0
    return vad.is_speech(frame_int16.tobytes(), SAMPLE_RATE)


def extract_log_mel(y: np.ndarray, mean: float, std: float) -> torch.Tensor:
    mel = librosa.feature.melspectrogram(
        y=y.astype(np.float32), sr=SAMPLE_RATE,
        n_fft=1024, hop_length=256, n_mels=64,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    target_frames = int(SAMPLE_RATE * WINDOW_S / 256) + 1
    if log_mel.shape[1] < target_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, target_frames - log_mel.shape[1])),
                         mode="constant", constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :target_frames]
    log_mel = (log_mel - mean) / (std + 1e-8)
    return torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["emotion", "fatigue"], default="emotion")
    parser.add_argument("--device", type=int, default=None, help="ID do microfone (sd.query_devices())")
    args = parser.parse_args()

    print(f"[INFO] A carregar modelo CNN-LSTM (task={args.task})...")
    model, classes, mean, std = load_model(args.task)

    vad = webrtcvad.Vad(2) if webrtcvad is not None else None
    if args.device is not None:
        sd.default.device = (args.device, None)
    print(f"[INFO] Microfone: {sd.query_devices(sd.default.device[0])['name'] if sd.default.device else 'default'}")

    buf = collections.deque(maxlen=WINDOW_SAMPLES)
    last_inference_t = 0.0
    voice_frames_total, voiced = 0, 0

    print("[INFO] Pronto. Fala perto do microfone. Ctrl+C para parar.\n")

    def callback(indata, frames, time_info, status):
        nonlocal voice_frames_total, voiced, last_inference_t
        if status:
            print(f"[WARN] {status}")
        # mono int16
        chunk = indata[:, 0]
        # Para VAD precisamos de int16 frames de FRAME_SAMPLES
        for i in range(0, len(chunk) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            frm_f32 = chunk[i:i + FRAME_SAMPLES]
            frm_i16 = (np.clip(frm_f32, -1.0, 1.0) * 32767).astype(np.int16)
            if has_voice(frm_i16, vad):
                voiced += 1
            voice_frames_total += 1
            buf.extend(frm_f32.tolist())

        now = time.time()
        if len(buf) >= WINDOW_SAMPLES and now - last_inference_t >= SLIDE_S:
            last_inference_t = now
            window = np.fromiter(buf, dtype=np.float32, count=WINDOW_SAMPLES)
            voice_ratio = voiced / max(1, voice_frames_total)
            voiced, voice_frames_total = 0, 0

            if voice_ratio < 0.2:
                print(f"[silence] voice={voice_ratio:.2f} — ignorado.")
                return

            with torch.no_grad():
                x = extract_log_mel(window, mean, std)
                logits = model(x)
                probs = torch.softmax(logits, dim=1).squeeze(0).numpy()
            top = int(np.argmax(probs))
            print(f"[{time.strftime('%H:%M:%S')}]  voice={voice_ratio:.2f}   "
                  f"pred={classes[top]:<10s}   p={probs[top]:.2f}   "
                  + "  ".join(f"{c[:3]}={p:.2f}" for c, p in zip(classes, probs)))

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=FRAME_SAMPLES, callback=callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[INFO] Terminado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
