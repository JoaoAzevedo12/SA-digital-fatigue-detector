import time
import csv
import os
import threading
import torch
import pickle
from pynput import mouse, keyboard
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import firebase_admin
from firebase_admin import credentials, firestore


# ---------------------------------------------------------
# 1. CONFIGURAR CAMINHOS E FICHEIROS
# ---------------------------------------------------------
DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
LOG_CSV = os.path.join(DIRETORIO_ATUAL, "log_contexto.csv")
CONTEXTO_CSV = os.path.join(DIRETORIO_ATUAL, "contexto.csv")

# Caminhos para o modelo que treinaste no Colab
MODELO_PATH = os.path.join(DIRETORIO_ATUAL, "modelo_contexto.pt")
MAPPING_PATH = os.path.join(DIRETORIO_ATUAL, "label_mapping_contexto.pkl")


# Iniciar ligação ao Firebase
try:
    cred = credentials.Certificate(os.path.join(DIRETORIO_ATUAL, "digital-fatigue-detector-firebase-adminsdk-fbsvc-e639133943.json"))
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    FIREBASE_ATIVO = True
    print("Ligação ao Firebase Firestore estabelecida com sucesso!")
except Exception as e:
    print(f"Aviso: Não foi possível ligar ao Firebase. Erro: {e}")
    FIREBASE_ATIVO = False

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
buffer_firestore = []
executando = True # Flag para sabermos quando o utilizador carregou no ESC

# Criar cabeçalhos nos CSVs se os ficheiros ainda não existirem
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["Datetime", "Acoes_Agrupadas"])

if not os.path.exists(CONTEXTO_CSV):
    with open(CONTEXTO_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(["Datetime", "Contexto_Previsto", "Confianca"])



# ---------------------------------------------------------
# 4. FUNÇÃO DE AGREGAÇÃO E UPLOAD (Com Limpeza Automática)
# ---------------------------------------------------------
def agrupar_e_enviar_firestore():
    global buffer_firestore
    if not buffer_firestore or not FIREBASE_ATIVO:
        return

    blocos_agrupados = []
    bloco_atual = None

    # Lógica de agrupamento (Sessionização)
    for prev in buffer_firestore:
        if bloco_atual is None:
            bloco_atual = {
                "Data_Inicio": prev["tempo"],
                "Data_Fim": prev["tempo"],
                "Contexto": prev["contexto"],
                "Confiancas": [prev["confianca"]]
            }
        elif prev["contexto"] == bloco_atual["Contexto"]:
            # Se for o mesmo contexto contínuo, atualizamos a Data de Fim
            bloco_atual["Data_Fim"] = prev["tempo"]
            bloco_atual["Confiancas"].append(prev["confianca"])
        else:
            # Contexto mudou! Guardamos o bloco anterior
            blocos_agrupados.append(bloco_atual)
            bloco_atual = {
                "Data_Inicio": prev["tempo"],
                "Data_Fim": prev["tempo"],
                "Contexto": prev["contexto"],
                "Confiancas": [prev["confianca"]]
            }
    
    # Adicionar o último bloco pendente
    if bloco_atual:
        blocos_agrupados.append(bloco_atual)

    print(f"\n📦 A iniciar upload para Firestore... ({len(blocos_agrupados)} sessões)")
    
    # --- REDE DE SEGURANÇA: Só apaga se o envio for bem sucedido ---
    try:
        for bloco in blocos_agrupados:
            media_acc = sum(bloco["Confiancas"]) / len(bloco["Confiancas"])
            
            doc_data = {
                "Data_Inicio": bloco["Data_Inicio"],
                "Data_Fim": bloco["Data_Fim"],
                "Contexto": bloco["Contexto"],
                "Accuracy": round(media_acc, 4)
            }
            
            # Enviar para a coleção "sessoes_contexto"
            db.collection("sessoes_contexto").add(doc_data)
        
        print("✅ Upload concluído com sucesso!")
        
        # --- LIMPEZA DOS DADOS LOCAIS ---
        # 1. Limpar a memória do Python
        buffer_firestore.clear() 
        
        # 2. Substituir os CSVs apenas pelos cabeçalhos limpos
        with open(LOG_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(["Datetime", "Acoes_Agrupadas"])

        with open(CONTEXTO_CSV, 'w', newline='') as f:
            csv.writer(f).writerow(["Datetime", "Contexto_Previsto", "Confianca"])
            
        print("🗑️ Ficheiros CSV locais limpos (Armazenamento Efémero ativado).")

    except Exception as e:
        # Se a internet falhar, ele avisa e NÃO apaga nada. Tenta outra vez passados 10 min.
        print(f"❌ Erro ao enviar para o Firestore: {e}")
        print("⚠️ Os dados NÃO foram apagados localmente. Nova tentativa no próximo ciclo.")
# ---------------------------------------------------------
# 5. A LÓGICA DE SENSORES (Privacy-by-Design)
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
# 6. O MOTOR DE INFERÊNCIA (Corre de 60 em 60 segundos)
# ---------------------------------------------------------
def motor_de_analise():
    global executando # Avisar o Python que vamos ler esta variável global
    minutos_decorridos = 0 # Inicialização local clara
    
    while executando:
        # Espera 60 segundos (1 min), verificando o ESC a cada segundo
        for _ in range(60):
            if not executando: 
                break
            time.sleep(1)
            
        if not executando: 
            break

        # 1. Copiar dados e limpar balde
        dados_janela = eventos_minuto_atual.copy()
        eventos_minuto_atual.clear()
        
        agora = time.strftime("%Y-%m-%d %H:%M:%S")

        # 2. Verificar se houve atividade
        if len(dados_janela) < 5:
            print(f"[{agora}] 💤 Janela ignorada (Inatividade).")
            minutos_decorridos += 1 # Incremento mesmo em inatividade para o upload de 10min
        else:
            frase_acoes = " ".join(dados_janela)

            # Guardar no log local
            with open(LOG_CSV, 'a', newline='') as f:
                csv.writer(f).writerow([agora, frase_acoes])

            # 3. Inferência (IA)
            inputs = tokenizer(frase_acoes, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                probabilidades = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
                pred_id = torch.argmax(probabilidades).item()
                confianca = probabilidades[pred_id].item()

            contexto_previsto = reverse_mapping[pred_id]
            
            print(f"\n[{agora}] 🎯 Previsão: {contexto_previsto.upper()} (Confiança: {confianca*100:.1f}%)")
            
            with open(CONTEXTO_CSV, 'a', newline='') as f:
                csv.writer(f).writerow([agora, contexto_previsto, round(confianca, 4)])

            # Guardar para o Firestore
            buffer_firestore.append({
                "tempo": agora,
                "contexto": contexto_previsto,
                "confianca": confianca
            })

            minutos_decorridos += 1
        
        # 4. Verificar se é altura do Upload (10 minutos)
        if minutos_decorridos >= 10:
            print(f"\n⏰ Passaram {minutos_decorridos} minutos. A preparar upload...")
            agrupar_e_enviar_firestore()
            minutos_decorridos = 0 # Reset do contador

    # Ao sair do loop (ESC), faz o último envio se houver dados
    if buffer_firestore:
        print("\n🛑 Programa interrompido. A enviar dados finais para a cloud...")
        agrupar_e_enviar_firestore()

# ---------------------------------------------------------
# 7. INICIAR O SISTEMA
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