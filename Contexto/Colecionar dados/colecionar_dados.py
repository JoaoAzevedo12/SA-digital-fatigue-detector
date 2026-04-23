import time
import csv
import os
import threading
import torch
import pickle
from pynput import mouse, keyboard
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ---------------------------------------------------------
# 1. CONFIGURAR CAMINHOS E FICHEIROS
# ---------------------------------------------------------
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
LOG_CSV = os.path.join(DIRETORIO_ATUAL, "log_contexto.csv")
CONTEXTO_CSV = os.path.join(DIRETORIO_ATUAL, "contexto.csv")

# Caminhos para o modelo que treinaste no Colab
MODELO_PATH = os.path.join(DIRETORIO_ATUAL, "modelo_contexto.pt")
MAPPING_PATH = os.path.join(DIRETORIO_ATUAL, "label_mapping_contexto.pkl")

# ---------------------------------------------------------
# 2. CARREGAR Modelo
# ---------------------------------------------------------
print("Correr modelo RoBERTa...")
# Forçamos a usar CPU porque a maioria dos PCs não tem GPU NVIDIA configurada para o PyTorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

tokenizer = AutoTokenizer.from_pretrained('roberta-base', use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

# Carregar os pesos (conhecimento) que geraste no Colab
model.load_state_dict(torch.load(MODELO_PATH, map_location=device))
model.to(device)
model.eval() # Modo de Inferência (não aprende mais, só prevê)

# Carregar o dicionário de tradução
with open(MAPPING_PATH, 'rb') as f:
    label_mapping = pickle.load(f)
reverse_mapping = {v: k for k, v in label_mapping.items()}

# ---------------------------------------------------------
# 3. VARIÁVEIS GLOBAIS E FICHEIROS CSV
# ---------------------------------------------------------
eventos_minuto_atual = []
executando = True # Flag para sabermos quando o utilizador carregou no ESC

# Criar cabeçalhos nos CSVs se os ficheiros ainda não existirem
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["Datetime", "Acoes_Agrupadas"])

if not os.path.exists(CONTEXTO_CSV):
    with open(CONTEXTO_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["Datetime", "Contexto_Previsto", "Confianca"])

# ---------------------------------------------------------
# 4. A LÓGICA DE SENSORES (Privacy-by-Design)
# ---------------------------------------------------------
def categorizar_tecla(key):
    try:
        char = key.char.lower()
        if char in ['w', 'a', 's', 'd']: return "Key_WASD"
        else: return "Key_AlphaNumeric"
    except AttributeError:
        if key == keyboard.Key.space: return "Key_Space"
        elif key in [keyboard.Key.backspace, keyboard.Key.delete]: return "Key_Delete"
        elif key in [keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right]: return "Key_Arrow"
        elif key == keyboard.Key.enter: return "Key_Enter"
        else: return "Key_Modifier" 

def registar_evento(detalhe):
    # Guardamos apenas a string da ação (ex: "Key_WASD_Press")
    eventos_minuto_atual.append(detalhe)

def on_click(x, y, button, pressed):
    acao = "Click_Press" if pressed else "Click_Release"
    registar_evento(f"{acao}_{button.name}")

def on_move(x, y):
    pass # Omitido para não encher a memória, o rato gera dezenas de eventos por segundo

def on_press(key):
    registar_evento(f"{categorizar_tecla(key)}_Press")

def on_release(key):
    registar_evento(f"{categorizar_tecla(key)}_Release")
    if key == keyboard.Key.esc:
        global executando
        executando = False
        print("\n A desligar os sensores...")
        return False

# ---------------------------------------------------------
# 5. O MOTOR DE INFERÊNCIA (Corre de 60 em 60 segundos)
# ---------------------------------------------------------
def motor_de_analise():
    while executando:
        # Espera 60 segundos, mas acorda a cada 1 segundo para verificar se carregaste no ESC
        for _ in range(60):
            if not executando: break
            time.sleep(1)
            
        if not executando: break

        # 1. Copiar os dados e limpar o balde para o próximo minuto não perder cliques
        dados_janela = eventos_minuto_atual.copy()
        eventos_minuto_atual.clear()
        
        agora = time.strftime("%Y-%m-%d %H:%M:%S")

        if len(dados_janela) < 5:
            print(f"[{agora}] 💤 Janela ignorada (Inatividade / Poucas ações).")
            continue

        # 2. Agrupar numa "frase"
        frase_acoes = " ".join(dados_janela)

        # 3. Guardar no log_contexto.csv
        with open(LOG_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([agora, frase_acoes])

        # 4. Inferência (Perguntar à IA)
        inputs = tokenizer(frase_acoes, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # Converter matemática em previsões (Softmax)
            probabilidades = torch.nn.functional.softmax(logits, dim=1)[0]
            pred_id = torch.argmax(probabilidades).item()
            confianca = probabilidades[pred_id].item()

        contexto_previsto = reverse_mapping[pred_id]
        
        # 5. Imprimir no ecrã e guardar no contexto.csv
        print(f"\n[{agora}] Previsão da IA: {contexto_previsto.upper()} (Confiança: {confianca*100:.1f}%)")
        with open(CONTEXTO_CSV, 'a', newline='') as f:
            csv.writer(f).writerow([agora, contexto_previsto, round(confianca, 4)])

# ---------------------------------------------------------
# 6. INICIAR O SISTEMA
# ---------------------------------------------------------
# Arranca o cronómetro de 60 segundos numa "Thread" paralela (em segundo plano)
thread_ia = threading.Thread(target=motor_de_analise)
thread_ia.start()

print("\nSensorização Ativa.")
print("A gravar e a analisar o teu comportamento de 1 em 1 minuto...")
print("Pressiona ESC para parar o programa.\n")

listener_rato = mouse.Listener(on_click=on_click, on_move=on_move)
listener_teclado = keyboard.Listener(on_press=on_press, on_release=on_release)

listener_rato.start()
listener_teclado.start()

# O programa principal fica aqui a "escutar" o rato e teclado
listener_teclado.join() 
listener_rato.join()

# Quando o ESC é pressionado, espera que a Thread da IA termine graciosamente
thread_ia.join()
print("✅ Sistema encerrado com sucesso.")