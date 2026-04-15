import time
import csv
from pynput import mouse, keyboard
import os 

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(DIRETORIO_ATUAL, "log_contexto.csv")

# Lista para guardar os eventos antes de passar para CSV
eventos = []

def registar_evento(tipo, detalhe):
    timestamp = time.time()
    eventos.append([timestamp, tipo, detalhe])
    print(f"{timestamp} | {tipo} | {detalhe}")

# --- CALLBACKS DO RATO ---
def on_move(x, y):
    # Para não encher o CSV, podes registar o movimento apenas a cada X milissegundos
    # ou usar esta função para calcular a distância percorrida depois.
    pass 

def on_click(x, y, button, pressed):
    acao = "Click_Press" if pressed else "Click_Release"
    registar_evento("Rato", f"{acao}_{button.name}")

# --- CALLBACKS DO TECLADO ---
def on_press(key):
    # Gravamos apenas que UMA tecla foi primida (Privacy-by-Design: não interessa qual)
    registar_evento("Teclado", "Key_Press")

def on_release(key):
    registar_evento("Teclado", "Key_Release")
    if key == keyboard.Key.esc:
        # Pressionar ESC para parar a gravação e guardar no CSV
        guardar_dados()
        return False

# --- GUARDAR DADOS ---
def guardar_dados():
    with open(CAMINHO_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Sensor", "Acao"])
        writer.writerows(eventos)
    print("Dados guardados com sucesso em 'log_contexto.csv'!")

# --- INICIAR OS LISTENERS ---
print("A gravar interações... Pressiona ESC para parar.")
listener_rato = mouse.Listener(on_click=on_click, on_move=on_move)
listener_teclado = keyboard.Listener(on_press=on_press, on_release=on_release)

listener_rato.start()
listener_teclado.start()
listener_rato.join()
listener_teclado.join()