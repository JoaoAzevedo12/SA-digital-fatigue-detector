import pandas as pd
import os 

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
CAMINHO_CSV = os.path.join(DIRETORIO_ATUAL, "log_contexto.csv")
CAMINHO_CSV_LIMPO = os.path.join(DIRETORIO_ATUAL, "dataset_contexto_limpo.csv")

# 1. Carregar os dados brutos gerados no Passo 1
# Assume-se que o ficheiro se chama 'log_contexto.csv'
df = pd.read_csv(CAMINHO_CSV)

# 2. Converter o Timestamp (segundos) para um formato de data/hora real
df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')

# 3. Definir a data/hora como o índice (necessário para agrupar por tempo)
df.set_index('Datetime', inplace=True)

# 4. Criar colunas para identificar os eventos específicos que nos interessam
df['Teclas_Pressionadas'] = df['Acao'].apply(lambda x: 1 if 'Key_Press' in str(x) else 0)
df['Cliques_Esquerdos'] = df['Acao'].apply(lambda x: 1 if 'Click_Press_left' in str(x) else 0)
df['Cliques_Direitos'] = df['Acao'].apply(lambda x: 1 if 'Click_Press_right' in str(x) else 0)

# 5. O SEGREDO: Agrupar tudo em janelas de 10 segundos ('10s') e somar!
janelas = df.resample('10s').sum(numeric_only=True)

# Limpar colunas desnecessárias (como a soma dos timestamps que não faz sentido)
janelas = janelas.drop(columns=['Timestamp'])

# 6. Mostrar o resultado (as "Features" para o nosso modelo de IA)
print("--- DADOS PRONTOS PARA A INTELIGÊNCIA ARTIFICIAL ---")
print(janelas.head())

# Guardar o dataset final limpo
janelas.to_csv(CAMINHO_CSV_LIMPO)