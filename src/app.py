import streamlit as st
import pandas as pd
import joblib
import os

# === Carregar modelo treinado ===
ROOT = os.path.abspath(os.path.join(os.getcwd(), ".."))
file_modelo = os.path.join(ROOT, "models", "xgb_model.pkl")
model = joblib.load(file_modelo)

# slider para sensibilidade (recall) via threshold
THRESHOLD = st.sidebar.slider("Threshold de classificação", 0.0, 1.0, 0.5, 0.01)
st.sidebar.markdown("""
**ℹ️ Sobre o Threshold de Classificação**  
- Diminuir o valor → o modelo classifica **mais casos como graves**  
  👉 aumenta o *recall* (menos falsos negativos), mas pode gerar mais falsos positivos.  
- Aumentar o valor → o modelo classifica **menos casos como graves**  
  👉 aumenta a *precisão* (menos falsos positivos), mas pode deixar passar casos graves.
""")


st.title("🦟 Previsão de Gravidade da Dengue")
st.write("Aplicativo para estimar risco de gravidade da dengue usando Machine Learning (XGBoost).")

def bin_input(label: str) -> int:
    # retorna 0/1 (Não/Sim) no mesmo formato do treino
    return st.radio(label, [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")

# ===============================
# 📋 BLOCO 1 - Informações Gerais
# ===============================
st.header("📋 Informações Gerais")

# Sexo (M=0, F=1)
cs_sexo = st.selectbox(
    "Sexo",
    options=[("Masculino", 0), ("Feminino", 1)],
    format_func=lambda x: x[0]
)[1]

# Dias de sintomas
dias_sintomas = st.number_input("Dias desde o início dos sintomas", min_value=0, max_value=185, value=0)

# Faixa Etária — baseline = Adolescente (todas dummies = 0)
st.subheader("Faixa Etária")
faixa_etaria = st.radio("Selecione", options=["Adolescente", "Crianca", "Adulto", "Idoso"])
faixa_etaria_Adulto  = 1 if faixa_etaria == "Adulto"  else 0
faixa_etaria_Crianca = 1 if faixa_etaria == "Crianca" else 0
faixa_etaria_Idoso   = 1 if faixa_etaria == "Idoso"   else 0
# Adolescente => todos 0

# Gestante — só aparece se sexo = Feminino
if cs_sexo == 1:
    st.subheader("Gestante")
    gestante_cat = st.radio(
        "Selecione",
        options=["gestante", "nao", "nao_aplica", "ignorado"]
    )
    gestante_cat_ignorado   = 1 if gestante_cat == "ignorado"   else 0
    gestante_cat_nao        = 1 if gestante_cat == "nao"        else 0
    gestante_cat_nao_aplica = 1 if gestante_cat == "nao_aplica" else 0
else:
    # Preenche automático para masculino: não gestante
    gestante_cat_ignorado   = 0
    gestante_cat_nao        = 1
    gestante_cat_nao_aplica = 0

# ===============================
# 🩺 BLOCO 2 - Sintomas Clínicos
# ===============================
st.header("🩺 Sintomas Clínicos")
febre       = bin_input("Febre")
mialgia     = bin_input("Mialgia")
cefaleia    = bin_input("Cefaleia")
exantema    = bin_input("Exantema")
vomito      = bin_input("Vômito")
nausea      = bin_input("Náusea")
dor_costas  = bin_input("Dor nas Costas")
conjuntvit  = bin_input("Conjuntivite")
artrite     = bin_input("Artrite")
artralgia   = bin_input("Artralgia")
petequia_n  = bin_input("Petéquias")
leucopenia  = bin_input("Leucopenia")
laco        = bin_input("Teste do Laço Positivo")
dor_retro   = bin_input("Dor Retroorbital")

# ==============================================
# ⚕️ BLOCO 3 - Doenças Pré-existentes / Comorb.
# ==============================================
st.header("⚕️ Doenças Pré-existentes / Comorbidades")
diabetes    = bin_input("Diabetes")
hematolog   = bin_input("Doença Hematológica")
hepatopat   = bin_input("Doença Hepática")
renal       = bin_input("Doença Renal")
hipertensa  = bin_input("Hipertensão")
acido_pept  = bin_input("Doença Péptica")
auto_imune  = bin_input("Doença Autoimune")

# ========= Monta exatamente X_train.columns =========
X_COLS = [
    'cs_sexo',
    'febre','mialgia','cefaleia','exantema','vomito','nausea',
    'dor_costas','conjuntvit','artrite','artralgia','petequia_n','leucopenia',
    'laco','dor_retro',
    'diabetes','hematolog','hepatopat','renal','hipertensa','acido_pept','auto_imune',
    'dias_sintomas',
    'faixa_etaria_Adulto','faixa_etaria_Crianca','faixa_etaria_Idoso',
    'gestante_cat_ignorado','gestante_cat_nao','gestante_cat_nao_aplica'
]

row = [
    cs_sexo,
    febre, mialgia, cefaleia, exantema, vomito, nausea,
    dor_costas, conjuntvit, artrite, artralgia, petequia_n, leucopenia,
    laco, dor_retro,
    diabetes, hematolog, hepatopat, renal, hipertensa, acido_pept, auto_imune,
    dias_sintomas,
    faixa_etaria_Adulto, faixa_etaria_Crianca, faixa_etaria_Idoso,
    gestante_cat_ignorado, gestante_cat_nao, gestante_cat_nao_aplica
]

input_data = pd.DataFrame([row], columns=X_COLS)

# ========= Predição =========
if st.button("🔮 Prever Gravidade"):
    prob = model.predict_proba(input_data)[0, 1]
    pred = int(prob >= THRESHOLD)

    st.subheader("📊 Resultado da Previsão")
    st.write(f"**Probabilidade de caso grave:** {prob:.2%}")
    st.write(f"Threshold aplicado: **{THRESHOLD:.2f}**")

    if pred == 1:
        st.error("⚠️ O modelo prevê **ALTO risco de gravidade**.")
    else:
        st.success("✅ O modelo prevê **BAIXO risco de gravidade**.")
