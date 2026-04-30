import os
import pandas as pd

def preparar_dataset_balanceado(input_file, output_file, segundos_por_janela=60):
    print(f"📂 A carregar dados brutos de:\n   -> {input_file}")
    df = pd.read_csv(input_file)
    
    # 1. Converter a coluna Datetime para um formato de tempo real
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    
    # 2. Agrupar por Janela de Tempo (60s) e Categoria
    print(f"⏱️ A agrupar ações em blocos de {segundos_por_janela} segundos...")
    df_grouped = df.groupby([
        pd.Grouper(key='Datetime', freq=f'{segundos_por_janela}s'),
        'Contexto_Label'
    ])['Acao'].apply(lambda x: " ".join(x)).reset_index()
    
    # Renomear as colunas
    df_grouped = df_grouped.rename(columns={'Contexto_Label': 'label', 'Acao': 'text'})
    
    # 3. Filtrar ruído (Remover minutos com menos de 5 ações)
    tamanho_original = len(df_grouped)
    df_grouped = df_grouped[df_grouped['text'].str.split().str.len() > 5]
    print(f"🧹 Ruído removido: {tamanho_original - len(df_grouped)} janelas ignoradas por inatividade.")
    
    # 4. BALANCEAMENTO (Undersampling)
    print("\n📊 Contagem ANTES do balanceamento:")
    contagem_inicial = df_grouped['label'].value_counts()
    print(contagem_inicial)
    
    print("\n⚖️ A aplicar Balanceamento...")
    menor_quantidade = contagem_inicial.min()
    df_balanceado = df_grouped.groupby('label').sample(n=menor_quantidade, random_state=42)
    
    print(f"\n🎯 Sucesso! Todas as categorias têm agora exatamente {menor_quantidade} exemplos.")
    
    # 5. Misturar as linhas (Shuffle)
    df_balanceado = df_balanceado.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 6. Guardar para o Colab
    df_final = df_balanceado[['text', 'label']]
    df_final.to_csv(output_file, index=False)
    
    print(f"\n💾 Ficheiro final gravado com sucesso em:\n   -> {output_file}")


if __name__ == "__main__":
    # Descobre automaticamente a pasta onde este script está guardado
    DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
    
    # Constrói os caminhos absolutos
    FICHEIRO_ENTRADA = os.path.join(DIRETORIO_ATUAL, 'dataset_treino_raw.csv')
    FICHEIRO_SAIDA = os.path.join(DIRETORIO_ATUAL, 'dataset_balanceado.csv')
    
    preparar_dataset_balanceado(FICHEIRO_ENTRADA, FICHEIRO_SAIDA)
