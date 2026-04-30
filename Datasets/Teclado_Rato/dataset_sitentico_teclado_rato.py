import csv
import random
import time
from datetime import datetime, timedelta
import os

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(DIRETORIO_ATUAL, "dataset_emocoes_raw.csv")

# 2000 janelas por combinação para garantir que o RoBERTa tem dados suficientes
JANELAS_POR_COMBINACAO = 500 

CONTEXTOS = ["Trabalhar", "Ler", "Jogar", "Lazer"]
EMOCOES = ["Normal", "Fadiga", "Stress"]

# --- OPÇÕES DE AÇÃO ATUALIZADAS ---
OPCOES_ACAO = [
    ("Teclado", "Key_AlphaNumeric_Press"), 
    ("Teclado", "Key_WASD_Press"),
    ("Teclado", "Key_Arrow_Press"), 
    ("Teclado", "Key_Space_Press"), 
    ("Teclado", "Key_Delete_Press"), 
    ("Rato", "Click_Press_left"),
    ("Rato", "Click_Press_right") # <-- Nova Ação Adicionada
]

# Perfis de Contexto Base (Ajustados para 7 probabilidades)
# [AlphaNum, WASD, Setas, Espaço, Delete, Click_Left, Click_Right]
PERFIS_BASE = {
    "Trabalhar": {
        "ev": (90, 270), 
        "probs": [0.60, 0.05, 0.05, 0.15, 0.10, 0.03, 0.02], 
        "dist": (0, 50)
    },
    "Ler": {
        "ev": (6, 48),   
        "probs": [0.05, 0.00, 0.60, 0.10, 0.00, 0.15, 0.10], 
        "dist": (20, 200)
    },
    "Jogar": {
        "ev": (240, 480), 
        "probs": [0.05, 0.45, 0.00, 0.15, 0.00, 0.20, 0.15], 
        "dist": (300, 1500)
    },
    "Lazer": {
        "ev": (12, 60),  
        "probs": [0.15, 0.05, 0.10, 0.15, 0.00, 0.35, 0.20], 
        "dist": (100, 600)
    }
}

dados_gerados = []
tempo_atual = datetime.now()
timestamp_unix = time.time()

print("🧠 A gerar Dataset Multimodal (Contexto + Emoção) com Click Direito...")

for contexto in CONTEXTOS:
    for emocao in EMOCOES:
        print(f"-> Simulando: {contexto} em estado {emocao}...")
        config = PERFIS_BASE[contexto]
        
        for i in range(JANELAS_POR_COMBINACAO):
            mult_eventos = 1.0
            mult_erro = 0.0
            mult_distancia = 1.0
            
            if emocao == "Fadiga":
                mult_eventos = 0.5   
                mult_erro = 0.15     
                mult_distancia = 0.6 
            elif emocao == "Stress":
                mult_eventos = 1.4   
                mult_erro = 0.20     
                mult_distancia = 1.8 

            num_eventos = int(random.randint(config["ev"][0], config["ev"][1]) * mult_eventos)
            
            # Ajustar probabilidade de Delete (erro)
            probs = config["probs"].copy()
            probs[4] = min(0.40, probs[4] + mult_erro) 
            
            passo_tempo = 60.0 / num_eventos if num_eventos > 0 else 60.0
            
            for _ in range(num_eventos):
                sensor, acao = random.choices(OPCOES_ACAO, weights=probs, k=1)[0]
                
                distancia = 0
                if sensor == "Rato":
                    dist_min, dist_max = config["dist"]
                    distancia = int(random.randint(dist_min, dist_max) * mult_distancia)
                
                tempo_legivel = tempo_atual.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                dados_gerados.append([
                    tempo_legivel, timestamp_unix, sensor, acao, 
                    distancia, contexto, emocao
                ])
                
                variacao_tempo = passo_tempo * random.uniform(0.5, 1.5) 
                tempo_atual += timedelta(seconds=variacao_tempo)
                timestamp_unix += variacao_tempo
            
            tempo_atual += timedelta(seconds=random.uniform(1, 3))

print("\n💾 A gravar ficheiro...")
with open(CAMINHO_CSV, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Datetime", "Timestamp", "Sensor", "Acao", "Distancia_Pixels", "Contexto_Label", "Emocao_Label"])
    writer.writerows(dados_gerados)

print(f"✅ Dataset pronto com cliques binários: {len(dados_gerados)} linhas.")