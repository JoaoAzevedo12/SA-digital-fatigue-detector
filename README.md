# 🧠 Biometria Comportamental e IA na Prevenção de Exaustão Digital

Este repositório contém a investigação e a arquitetura de um sistema inteligente focado na monitorização do bem-estar e na prevenção de *burnout* durante o uso quotidiano do computador. O projeto utiliza técnicas avançadas de **biometria comportamental** e **Inteligência Artificial** para detetar sinais de fadiga cognitiva de forma não invasiva e *privacy-preserving*.

---

## 🚀 Visão Geral

A solução proposta transita do paradigma tradicional de monitorização de produtividade para um ecossistema focado na saúde do utilizador. Através da análise de micro-padrões de interação com periféricos e sinais vocais, o sistema consegue inferir estados de exaustão digital sem comprometer a privacidade individual.

### 🛠️ Eixos Tecnológicos
O sistema baseia-se na convergência de quatro dimensões de análise:
* **Keystroke Dynamics:** Análise de ritmos de digitação (*Dwell time*, *Flight time*).
* **Mouse Dynamics:** Monitorização da cinemática do cursor (velocidade, aceleração, tempos de silêncio).
* **Speech Emotion Recognition (SER):** Extração de características paralinguísticas (MFCCs, *Pitch*, Energia vocal) para deteção de *stress* em áudio.
* **Reconhecimento de Contexto:** Utilização de NLP e *Transformers* (*Mouse2Vec*) para inferir a tarefa em execução através da "assinatura física" do movimento, sem captura de ecrã.

---

## 🏗️ Arquitetura do Sistema

O sistema é desenhado de forma modular, garantindo escalabilidade e proteção de dados:

1.  **Camada de Sensorização:** Recolha passiva de eventos de teclado, rato, áudio e contexto.
2.  **Processamento e Extração:** Transformação de dados brutos em métricas temporais, espaciais e acústicas.
3.  **Modelação Comportamental:** Utilização de redes híbridas **CNN + LSTM** para identificar padrões de fadiga e anomalias face a uma *baseline* individual.
4.  **Feedback e Bem-Estar:** Interface de visualização de indicadores agregados e sugestões de pausas inteligentes através de **ludificação**.

---

## 🛡️ Privacidade e Ética (Privacy-by-Design)

O projeto foi concebido sob o princípio da **Minimização de Dados** e em conformidade com o **RGPD**.
* **Edge AI & Federated Learning:** O processamento ocorre localmente no dispositivo, evitando a exposição de dados sensíveis em plataformas *cloud*.
* **Anonimização:** O conteúdo semântico da fala e o conteúdo textual da digitação nunca são gravados; apenas as características matemáticas (vetores de tempo e frequência) são processadas.
* **Consentimento Dinâmico:** O utilizador mantém o controlo total sobre o nível de monitorização e a granularidade dos dados partilhados.

---

## 💻 Tecnologias Utilizadas

* **Linguagens:** Python (Processamento de Dados e ML).
* **Modelos:** *Machine Learning* (Random Forest, SVM) e *Deep Learning* (CNN, LSTM, Transformers).
* **Bibliotecas:** TensorFlow/PyTorch, Librosa (Áudio), Pandas/NumPy.
* **Formatação:** Typst (Relatório e Apresentação).

---

## 👥 Autores

Trabalho realizado no âmbito da Unidade Curricular de **Sensorização e Ambiente** (26.03.2026):

* **Diogo Azevedo** (PG61217)
* **João Azevedo** (PG61693)
* **Vicente Castro** (PG60395)
* **Martim Ferreira** (PG60390)

**Departamento de Informática, Escola de Engenharia – Universidade do Minho**

---

## 📄 Licença

Este projeto foi desenvolvido para fins académicos. Todos os direitos de propriedade intelectual pertencem aos autores e à respetiva instituição de ensino.
