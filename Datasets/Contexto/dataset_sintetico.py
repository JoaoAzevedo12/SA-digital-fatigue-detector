import csv
import random
import time
from datetime import datetime, timedelta
import os

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(DIRETORIO_ATUAL, "dataset_treino_raw.csv")

# Vai gerar 300 janelas de 60 segundos (1 minuto) por categoria
# Isso equivale a 5 horas exatas de dados sintéticos para cada categoria (20 horas no total)
JANELAS_POR_CATEGORIA = 2000 

# As nossas categorias e o seu comportamento estatístico (Agora em APM - Actions Per Minute)
# formato: (Min Eventos por 60s, Max Eventos, Probabilidades [AlphaNum, WASD, Setas, Espaço/Enter, Delete, Rato_Clique], Range Distância Rato)
PERFIS = {
    "Trabalhar": {
        "eventos_60s": (90, 270), # APM de trabalho focado (Digitação contínua)
        "probs_acao": [0.60, 0.05, 0.05, 0.15, 0.10, 0.05], 
        "distancia_rato": (0, 50) 
    },
    "Ler": {
        "eventos_60s": (6, 48), # Muito lento (scroll esporádico ou virar página)
        "probs_acao": [0.05, 0.00, 0.60, 0.10, 0.00, 0.25], 
        "distancia_rato": (20, 200) 
    },
    "Jogar": {
        "eventos_60s": (240, 480), # APM de Gaming competitivo (Foco e stress)
        "probs_acao": [0.10, 0.50, 0.00, 0.15, 0.00, 0.25], 
        "distancia_rato": (300, 1500) 
    },
    "Lazer": {
        "eventos_60s": (12, 60), # Ritmo relaxado (YouTube, Redes Sociais)
        "probs_acao": [0.20, 0.05, 0.10, 0.15, 0.00, 0.50], 
        "distancia_rato": (100, 600) 
    }
}

OPCOES_ACAO = [
    ("Teclado", "Key_AlphaNumeric_Press"),
    ("Teclado", "Key_WASD_Press"),
    ("Teclado", "Key_Arrow_Press"),
    ("Teclado", "Key_Space_Press"), 
    ("Teclado", "Key_Delete_Press"),
    ("Rato", "Click_Press_left")
]

dados_gerados = []
tempo_atual = datetime.now()
timestamp_unix = time.time()

print("⏳ A gerar dataset sintético de longo termo (Janelas de 1 Minuto)...")

for categoria, config in PERFIS.items():
    print(f"A simular comportamento: {categoria}...")
    
    for _ in range(JANELAS_POR_CATEGORIA):
        # Quantos eventos o "humano" fez nestes 60 segundos?
        num_eventos = random.randint(config["eventos_60s"][0], config["eventos_60s"][1])
        
        # O tempo avança dentro da janela de 60s
        passo_tempo = 60.0 / num_eventos if num_eventos > 0 else 60.0
        
        for _ in range(num_eventos):
            # Escolher a ação baseada nas probabilidades
            sensor, acao = random.choices(OPCOES_ACAO, weights=config["probs_acao"], k=1)[0]
            
            distancia = 0
            if sensor == "Rato":
                distancia = random.randint(config["distancia_rato"][0], config["distancia_rato"][1])
            
            tempo_legivel = tempo_atual.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            
            dados_gerados.append([tempo_legivel, timestamp_unix, sensor, acao, distancia, categoria])
            
            # Adicionar imperfeição humana no tempo entre cliques
            variacao_tempo = passo_tempo * random.uniform(0.5, 1.5) 
            tempo_atual += timedelta(seconds=variacao_tempo)
            timestamp_unix += variacao_tempo
            
        # Garante o salto temporal para separar bem as janelas no ficheiro
        salto = random.uniform(1, 5)
        tempo_atual += timedelta(seconds=salto)
        timestamp_unix += salto

with open(CAMINHO_CSV, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Datetime", "Timestamp", "Sensor", "Acao", "Distancia_Pixels", "Contexto_Label"])
    writer.writerows(dados_gerados)

print(f"✅ Sucesso! Geradas {len(dados_gerados)} linhas de interações humanas em '{CAMINHO_CSV}'.")