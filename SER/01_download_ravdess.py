"""
01_download_ravdess.py
======================
Faz download do dataset RAVDESS (Audio Speech) do Zenodo
e descomprime-o em ./data/raw/

RAVDESS - The Ryerson Audio-Visual Database of Emotional Speech and Song
Livingstone & Russo (2018), CC BY-NC-SA 4.0
Fonte: https://zenodo.org/record/1188976

Estrutura dos ficheiros: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.wav
Exemplo: 03-01-06-01-02-01-12.wav
  03 = Audio only
  01 = speech
  06 = fearful   (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
  01 = normal intensity
  02 = "Dogs are sitting by the door"
  01 = primeira repetição
  12 = ator 12 (ímpar=masculino, par=feminino)
"""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
DEST_DIR = Path(__file__).parent / "data" / "raw"
ZIP_PATH = DEST_DIR / "Audio_Speech_Actors_01-24.zip"


def download_with_progress(url: str, dest: Path, chunk_size: int = 1 << 14) -> None:
    """Download HTTP com barra de progresso."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] A descarregar {url}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name
        ) as bar:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    print(f"[OK] Guardado em {dest}")


def unzip(zip_path: Path, dest: Path) -> None:
    """Descomprimir o zip."""
    print(f"[INFO] A descomprimir para {dest}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    print("[OK] Descompressão completa.")


def count_files(directory: Path) -> int:
    return len(list(directory.rglob("*.wav")))


def main() -> int:
    DEST_DIR.mkdir(parents=True, exist_ok=True)

    if count_files(DEST_DIR) >= 1400:
        print(f"[INFO] Dataset já presente ({count_files(DEST_DIR)} .wav). A saltar download.")
        return 0

    if not ZIP_PATH.exists():
        try:
            download_with_progress(URL, ZIP_PATH)
        except Exception as e:
            print(f"[ERRO] Download falhou: {e}", file=sys.stderr)
            print(f"[DICA] Faça download manual de {URL}")
            print(f"[DICA] e coloque o ficheiro em {ZIP_PATH}")
            return 1
    else:
        print(f"[INFO] Zip já existe em {ZIP_PATH}")

    unzip(ZIP_PATH, DEST_DIR)

    n = count_files(DEST_DIR)
    print(f"[OK] {n} ficheiros .wav extraídos.")
    if n < 1400:
        print("[AVISO] Esperados ~1440 ficheiros. Verifique a integridade do zip.")
        return 2

    # Limpeza opcional do zip (libertar espaço)
    # ZIP_PATH.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
