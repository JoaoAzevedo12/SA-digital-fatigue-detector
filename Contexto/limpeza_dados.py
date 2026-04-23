import time
import csv
import math
import os 
from pynput import mouse, keyboard
from datetime import datetime

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(DIRETORIO_ATUAL, "log_contexto.csv")

# Variáveis globais
eventos = []
ultima_pos_rato = None  # Guarda a coordenada (x, y) da última interação

# ---------------------------------------------------------
# PASSO 1: LÓGICA DE PRIVACIDADE E CATEGORIZAÇÃO
# ---------------------------------------------------------
def categorizar_tecla(key):
    """
    Garante o Privacy-by-Design: nunca sabemos o que foi escrito, apenas a família da tecla.
    """
    try:
        char = key.char.lower()
        if char in ['w', 'a', 's', 'd']:
            return "Key_WASD"
        else:
            return "Key_AlphaNumeric"
    except AttributeError:
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

def calcular_distancia(pos_atual):
    """
    Calcula os pixels percorridos desde a última ação (Teorema de Pitágoras).
    """
    global ultima_pos_rato
    if ultima_pos_rato is None:
        ultima_pos_rato = pos_atual
        return 0
    
    dist = math.sqrt((pos_atual[0] - ultima_pos_rato[0])**2 + 
                     (pos_atual[1] - ultima_pos_rato[1])**2)
    
    ultima_pos_rato = pos_atual
    return dist

def registar_evento(tipo, detalhe, distancia=0):
    """
    Regista o evento com Tempo Legível, Timestamp Unix, Sensor, Ação e Distância.
    """
    timestamp_unix = time.time()
    tempo_legivel = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    dist_arredondada = round(distancia)
    eventos.append([tempo_legivel, timestamp_unix, tipo, detalhe, dist_arredondada])
    print(f"{tempo_legivel} | {tipo} | {detalhe} | Dist: {dist_arredondada}px")

# --- CALLBACKS DO RATO ---
def on_move(x, y):
    pass 

def on_click(x, y, button, pressed):
    if pressed:
        dist = calcular_distancia((x, y))
        acao = f"Click_Press_{button.name}"
        registar_evento("Rato", acao, dist)

# --- CALLBACKS DO TECLADO ---
def on_press(key):
    # Captar onde o rato está parado enquanto se escreve
    pos_atual_rato = mouse.Controller().position
    dist = calcular_distancia(pos_atual_rato)
    
    categoria = categorizar_tecla(key)
    registar_evento("Teclado", f"{categoria}_Press", dist)

def on_release(key):
    # Não calculamos distância no release para não duplicar dados desnecessários
    categoria = categorizar_tecla(key)
    registar_evento("Teclado", f"{categoria}_Release", 0)
    
    if key == keyboard.Key.esc:
        print("\n[!] A parar a monitorização...")
        guardar_dados()
        return False

# --- GUARDAR DADOS ---
def guardar_dados():
    with open(CAMINHO_CSV, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Datetime", "Timestamp", "Sensor", "Acao", "Distancia_Pixels"])
        writer.writerows(eventos)
    print(f"✅ Dados guardados com sucesso em:\n{CAMINHO_CSV}")

# --- INICIAR OS LISTENERS ---
print("🛡️ MindGuard AI - Sensorização Ética e Espacial Iniciada.")
print("A gravar interações... Pressiona ESC para parar e guardar o ficheiro.")

listener_rato = mouse.Listener(on_click=on_click)
listener_teclado = keyboard.Listener(on_press=on_press, on_release=on_release)

listener_rato.start()
listener_teclado.start()
listener_rato.join()
listener_teclado.join()