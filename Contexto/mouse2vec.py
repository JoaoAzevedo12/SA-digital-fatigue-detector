import pandas as pd
from gensim.models import Word2Vec
import os 

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(DIRETORIO_ATUAL, "log_contexto.csv")

# 1. Carregar os dados originais (o log contínuo do Passo 1, não o agrupado)
print("A carregar os dados brutos do rato...")
df = pd.read_csv(CAMINHO_CSV)

# 2. Transformar os eventos em "Frases"
# Vamos agrupar as ações de cada 10 segundos numa lista de "palavras"
df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('Datetime', inplace=True)

# Agrupar as ações numa única string ("frase") a cada 10 segundos
frases_de_rato = df['Acao'].resample('10s').apply(list)

# Remover blocos vazios (onde não se mexeu no computador)
frases_de_rato = frases_de_rato[frases_de_rato.astype(bool)]

print("\nExemplo de uma 'Frase' lida pelo modelo:")
print(frases_de_rato.iloc[0]) 
# Saída esperada: ['Move', 'Click_Press_left', 'Click_Release_left', 'Key_Press']

# 3. Treinar o modelo MOUSE2VEC (A replicação do artigo!)
print("\nA treinar o modelo Word2Vec com os movimentos do rato...")

# Treinamos a IA para entender a relação espacial e temporal das "palavras" do rato
modelo_mouse2vec = Word2Vec(
    sentences=frases_de_rato, 
    vector_size=50,  # Transforma cada ação num vetor matemático de 50 dimensões
    window=5,        # Olha para as 5 ações antes e depois para entender o contexto
    min_count=1,     # Ignorar ações muito raras
    workers=4        # Usar o processador do pc ao máximo
)

print("\nTreino concluído com sucesso!")

# 4. A Magia: Perguntar à IA o que ela aprendeu!
# Vamos ver que ações o modelo acha que são contextualmente parecidas
acao_teste = 'Click_Press_left'
if acao_teste in modelo_mouse2vec.wv:
    similares = modelo_mouse2vec.wv.most_similar(acao_teste)
    print(f"\nAções que normalmente acontecem no mesmo contexto que '{acao_teste}':")
    for acao, probabilidade in similares:
        print(f"- {acao} (Probabilidade: {probabilidade*100:.1f}%)")