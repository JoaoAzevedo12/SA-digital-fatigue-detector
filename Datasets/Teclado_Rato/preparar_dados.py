import os
import pandas as pd

def preparar_dataset_emocoes_completo(input_file, output_file, segundos_por_janela=60):
    print(f"📂 A carregar dados brutos de:\n   -> {input_file}")
    # Carregamos o CSV original
    df = pd.read_csv(input_file)
    
    # 1. Converter a coluna Datetime para um formato de tempo real
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 2. Agrupar por Janela de Tempo, Contexto e Emoção
    print(f"⏱️ A agrupar ações em blocos de {segundos_por_janela} segundos...")
    df_grouped = df.groupby([
        pd.Grouper(key='Datetime', freq=f'{segundos_por_janela}s'),
        'Contexto_Label',
        'Emocao_Label'
    ])['Acao'].apply(lambda x: " ".join(x)).reset_index()
    
    # Renomear as colunas para manter a clareza e compatibilidade com o notebook
    # 'text' é o padrão para o conteúdo, 'label' é o alvo do treino (emoção)
    df_grouped = df_grouped.rename(columns={
        'Acao': 'text',
        'Emocao_Label': 'label',
        'Contexto_Label': 'context'
    })
    
    # 3. Filtrar janelas sem atividade (menos de 5 ações)
    tamanho_original = len(df_grouped)
    df_grouped = df_grouped[df_grouped['text'].str.split().str.len() > 5]
    print(f"🧹 Ruído removido: {tamanho_original - len(df_grouped)} janelas ignoradas.")
    
    # 4. BALANCEAMENTO por Emoção (Garantir que Fadiga, Stress e Normal tenham o mesmo peso)
    print("\n📊 Contagem por Emoção ANTES do balanceamento:")
    contagem_inicial = df_grouped['label'].value_counts()
    print(contagem_inicial)
    
    print("\n⚖️ A aplicar Balanceamento...")
    menor_quantidade = contagem_inicial.min()
    df_balanceado = df_grouped.groupby('label').sample(n=menor_quantidade, random_state=42)
    
    # 5. Misturar as linhas para o modelo não decorar a ordem[cite: 1]
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 6. Guardar mantendo TODAS as colunas importantes
    # Agora incluímos 'text', 'label' (emoção) e 'context' (trabalho/lazer/etc)
    df_final = df_balanceado[['text', 'label', 'context']]
    df_final.to_csv(output_file, index=False)
    
    print(f"\n✅ Sucesso! O novo dataset mantém o contexto e a emoção.")
    print(f"📊 Contagem final: {df_final['label'].value_counts().to_dict()}")
    print(f"💾 Guardado em: {output_file}")

if __name__ == "__main__":
    DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
    
    FICHEIRO_ENTRADA = os.path.join(DIRETORIO_ATUAL, 'dataset_emocoes_raw.csv')
    FICHEIRO_SAIDA = os.path.join(DIRETORIO_ATUAL, 'dataset_emocoes_balanceado.csv')
    
    preparar_dataset_emocoes_completo(FICHEIRO_ENTRADA, FICHEIRO_SAIDA)