import time
import csv
from pynput import mouse, keyboard
import os 

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(DIRETORIO_ATUAL, "log_contexto.csv")

eventos = []

# ---------------------------------------------------------
# PASSO 1: A LÓGICA DE PRIVACIDADE E CATEGORIZAÇÃO
# ---------------------------------------------------------
def categorizar_tecla(key):
    """
    Transforma a tecla premida numa categoria ética. 
    Garante o Privacy-by-Design: nunca sabemos o que foi escrito, apenas a família da tecla.
    """
    try:
        # Tentar ver se é um caractere normal (letras, números, símbolos)
        char = key.char.lower()
        if char in ['w', 'a', 's', 'd']:
            return "Key_WASD"
        else:
            return "Key_AlphaNumeric"
    except AttributeError:
        # Se der erro, é porque é uma tecla especial (Shift, Espaço, Setas...)
        if key == keyboard.Key.space:
            return "Key_Space"
        elif key in [keyboard.Key.backspace, keyboard.Key.delete]:
            return "Key_Delete"
        elif key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]:
            return "Key_Arrow"
        elif key == keyboard.Key.enter:
            return "Key_Enter"
        else:
            return "Key_Modifier" 

def registar_evento(tipo, detalhe):
    timestamp = time.time()
    eventos.append([timestamp, tipo, detalhe])
    print(f"{timestamp} | {tipo} | {detalhe}")

# --- CALLBACKS DO RATO ---
def on_move(x, y):
    pass 

def on_click(x, y, button, pressed):
    acao = "Click_Press" if pressed else "Click_Release"
    registar_evento("Rato", f"{acao}_{button.name}")

# --- CALLBACKS DO TECLADO ---
def on_press(key):
    categoria = categorizar_tecla(key)
    registar_evento("Teclado", f"{categoria}_Press")

def on_release(key):
    categoria = categorizar_tecla(key)
    registar_evento("Teclado", f"{categoria}_Release")
    
    if key == keyboard.Key.esc:
        # Pressionar ESC para parar a gravação e guardar no CSV
        print("\nA parar a monitorização...")
        guardar_dados()
        return False

# --- GUARDAR DADOS ---
def guardar_dados():
    with open(CAMINHO_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Sensor", "Acao"])
        writer.writerows(eventos)
    print(f"Dados guardados com sucesso em:\n{CAMINHO_CSV}")

# --- INICIAR OS LISTENERS ---
print("🛡️ MindGuard AI - Sensorização Ética Iniciada.")
print("A gravar interações... Pressiona ESC para parar.")
listener_rato = mouse.Listener(on_click=on_click, on_move=on_move)
listener_teclado = keyboard.Listener(on_press=on_press, on_release=on_release)

listener_rato.start()
listener_teclado.start()
listener_rato.join()
listener_teclado.join()
