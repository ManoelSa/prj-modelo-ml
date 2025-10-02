# 🦟 Predição de Gravidade da Dengue

## 🎯 Objetivo
A dengue é uma doença que pode evoluir para **quadros graves**, exigindo atenção rápida no diagnóstico e tratamento.
Também conhecida como dengue com sinais de alarme, a dengue grave é aquela que ocorre quando, de três a sete dias após o início dos sintomas tradicionais, o paciente entra em uma fase crítica, apresentando piora no estado clínico geral. A doença progride, geralmente, para sintomas graves e pode inclusive levar a óbito.  
O objetivo deste projeto é desenvolver um modelo de classificação baseado em Machine Learning capaz de **prever a gravidade da doença** a partir de dados clínicos, demográficos e epidemiológicos, permitindo:  
- **Apoiar profissionais de saúde** na tomada de decisão.  
- **Agilizar o tratamento** de pacientes com maior risco.  
- **Auxiliar na gestão hospitalar**, prevenindo sobrecarga e melhorando o planejamento de recursos.

## 🌟 Visão Geral
Este projeto aplica técnicas de **aprendizado supervisionado** para treinar modelos que estimam a probabilidade de um caso de dengue evoluir para forma grave.  
O modelo final foi integrado em uma aplicação **Streamlit**, oferecendo uma interface simples para previsão em tempo real.

## ✨ Principais Funcionalidades

- 🔎 Análise exploratória e limpeza dos dados.  
- 🧠 Treinamento e comparação entre diferentes algoritmos de classificação.  
- 📈 Avaliação de métricas como *Recall*, *Precision*, *ROC AUC* e *Matriz de Confusão*.  
- 💾 Salvamento e carregamento de modelos com **joblib**.  
- 🌐 Aplicação **Streamlit** para previsão de gravidade com inserção de dados de pacientes. 

## 🛠️ Detalhes de Implementação

- **Linguagem:** Python 3.12.6  
- **Bibliotecas principais:**  
  - pandas, numpy, os  
  - scikit-learn (LogisticRegression, DecisionTreeClassifier, GridSearchCV, métricas)  
  - xgboost  
  - matplotlib  
  - joblib  
  - streamlit  

- **Modelos utilizados nos testes:**  
  - LogisticRegression  
  - DecisionTreeClassifier 
  - XGBClassifier 

## 🗂️ Estrutura do Projeto
```text
prj-modelo-ml/
├── dataset/ # Conjunto de dados em formato parquet
│ └── DENGBR25.parquet
├── models/ # Modelos treinados
│ └── xgb_model.pkl
├── notebook/ # Notebooks de experimentação e análises
│ └── EDA.ipynb
| └── modelo.ipynb
├── src/ # (aplicação em Streamlit)
│ └── app.py
├── .gitignore # Arquivos ignorados no versionamento
├── README.md # Documentação do projeto
└── requirements.txt # Dependências do projeto 

```

## 🔀 Fluxo de Uso da Aplicação

```text
[Usuário] --(input de dados clínicos)--> Streamlit 
            |
            v
     [Modelo carregado com joblib]
            |
            v
   [Previsão de gravidade do caso]
            |
            v
[Interface] Exibe resultado e métricas associadas

```

## 💻 Exemplos de uso Localmente
```bash
# 1. Clone o repositório
git https://github.com/ManoelSa/prj-modelo-ml.git
cd prj-modelo-ml

# 2. (Opcional) Crie e ative um ambiente virtual
python -m venv venv
venv\Scripts\activate #Linux: source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. Modelo pré-treinado já disponível
# O arquivo models/xgb_model.pkl já pode ser usado pela aplicação

# 5. (Opcional) Treinar novamente o modelo
# Abra e execute o notebook em: notebook/modelo.ipynb
# Obs.: A etapa de "Tuning hiperparam - Conferindo melhores modelos" leva em média uns 20 min para execução.

# 6. Execute a aplicação Streamlit
cd src
streamlit run app.py

# 7. Url de Acesso
http://localhost:8501

```

## 🏆 Escolha do Modelo Final

Foram testados três algoritmos principais, comparados pelas métricas mais relevantes para o problema (foco na **classe 1 – casos graves**).

| Modelo                  | Accuracy | Precision | Recall | F1-score | ROC AUC |
|--------------------------|----------|-----------|--------|----------|---------|
| **XGBClassifier**        | 0.706    | 0.056     | **0.639** | **0.103**  | **0.731** |
| DecisionTreeClassifier   | 0.696    | 0.053     | 0.624  | 0.098    | 0.712   |
| LogisticRegression       | 0.700    | 0.052     | 0.598  | 0.095    | 0.702   |

### Critérios da escolha
- **Recall**: XGBoost obteve o melhor valor, fundamental para **não deixar de identificar casos graves**.  
- **ROC AUC**: maior separação entre classes, garantindo maior robustez.  
- **F1-score**: também superior, mesmo com o desbalanceamento da base.  

### Conclusão 
Apesar de todos os modelos apresentarem métricas próximas e tendo como desafio o desbalanceamento da base, o **XGBoost foi escolhido como modelo final** por:

- Obter os melhores valores de **recall, F1-score e ROC AUC**, mesmo que com margens pequenas.   
- Ser mais robusto para capturar **relações não lineares** entre variáveis clínicas, demográficas e epidemiológicas.  
- Embora o processo de busca de hiperparâmetros tenha demandado mais tempo, uma vez definidos os melhores parâmetros, o XGBoost apresentou **treinamento rápido e pouco custoso**, viabilizando re-treinos periódicos em produção.

### Observação final

Embora o **XGBoost** tenha apresentado os melhores resultados entre os modelos avaliados, as métricas globais ainda não são boas devido ao **forte desbalanceamento da base**.  

- **Accuracy (~0.71)**: Devido o desbalanceamento da base ela parece alta mas acaba sendo enganosa, pois o modelo acerta muito a classe Não grave (0) que é a majoritária.
- **Recall (~0.64)**: indica a proporção de casos graves (classe 1) que o modelo conseguiu identificar corretamente.  
- **Precision (~0.06)**: mostra que, entre os casos classificados como graves, poucos realmente eram. Isso pode ser esperado, pois o modelo tende a gerar **mais falsos positivos** para não deixar passar casos graves.  
- **F1-score (~0.10)**: baixo pelo desbalanceamento, mas superior ao dos demais modelos.  
- **ROC AUC (~0.73)**: mede a capacidade do modelo de **ranquear** casos graves acima dos não graves (0.5 = aleatório; 1.0 = perfeito). É **independente do threshold** e indica **separação moderada** entre as classes.

📌 **Importante**: na aplicação **Streamlit**, o **threshold de decisão** (por padrão 0.5) pode ser ajustado.  
- **Threshold mais baixo** → aumenta a sensibilidade (recall), detectando mais casos graves, mas gera mais alarmes falsos.  
- **Threshold mais alto** → reduz alarmes falsos, mas aumenta o risco de não identificar alguns casos graves.  

Esse ajuste permite calibrar o modelo conforme o **contexto de uso** (apoio clínico, gestão hospitalar, análises epidemiológicas, etc.).


> ℹ️ _Aviso: Este projeto tem fins educacionais e de pesquisa. O modelo não substitui avaliação médica._