import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Configuração básica da página ----------
st.set_page_config(
    page_title="Classificação de Células Sanguíneas",
    layout="wide"
)

st.title("Classificação de Células Sanguíneas com Deep Learning")
st.markdown(
    """
    Este aplicativo demonstra uma solução de **classificação automática de células sanguíneas**
    usando modelos de *deep learning*, com foco no **ResNet-50** treinado sobre o dataset
    de imagens microscópicas de sangue (Kaggle – Blood Cell Images).

    Abaixo você pode ver o **resumo dos experimentos** realizados com três modelos
    (DenseNet-161, ResNet-50 e VGG-11), seguido da área para **enviar uma imagem de lâmina de sangue**
    e obter a **predição em português**.
    """
)

st.subheader("Resultados dos Experimentos")

# Valores de exemplo — depois você pode ajustar com os números exatos do notebook
data = {
    "Modelo": ["DenseNet-161", "ResNet-50", "VGG-11"],
    "Epochs": [12, 15, 15],

    # Val Loss / Val Acc / Val F1
    "Best Val Loss":  [0.0031, 0.0029, 0.0028],
    "Final Val Loss": [0.0045, 0.0149, 0.0028],

    "Best Val Acc":   [0.9995, 0.9995, 1.0000],
    "Final Val Acc":  [0.9985, 0.9965, 1.0000],

    "Best Val F1":    [0.9995, 0.9995, 1.0000],
    "Final Val F1":   [0.9985, 0.9965, 1.0000],

    # Métricas de teste
    "Test Acc":       [0.8762, 0.8854, 0.8762],
    "Test Precision": [0.9037, 0.9064, 0.8999],
    "Test Recall":    [0.8761, 0.8854, 0.8761],
    "Test F1":        [0.8795, 0.8880, 0.8793], 
}

df_results = pd.DataFrame(data)
st.dataframe(df_results, use_container_width=True)

# ---------- Seção 2: Gráficos ----------
st.subheader("Comparação Visual dos Modelos")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Acurácia e F1-Score no Conjunto de Teste**")
    fig, ax = plt.subplots()
    ax.bar(df_results["Modelo"], df_results["Test Acc"], label="Acurácia")
    ax.bar(df_results["Modelo"], df_results["Test F1"], alpha=0.7, label="F1-Score")
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Valor")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.markdown("**Precisão e Recall**")
    fig2, ax2 = plt.subplots()
    ax2.bar(df_results["Modelo"], df_results["Test Precision"], label="Precisão")
    ax2.bar(df_results["Modelo"], df_results["Test Recall"], alpha=0.7, label="Recall")
    ax2.set_ylim(0.8, 1.0)
    ax2.set_ylabel("Valor")
    ax2.legend()
    st.pyplot(fig2)

st.markdown("---")

st.subheader("Desempenho Detalhado do Modelo Final (ResNet-50)")

resCol1, resCol2 = st.columns(2)

with resCol1:
    st.markdown("### Gráfico de Loss por Época")
    st.image("src/assets/loss-resnet50.png", use_column_width=True)

with resCol2:
    st.markdown("### Matriz de Confusão (ResNet-50)")
    st.image("src/assets/confusion_matrix_resnet50.png", use_column_width=True)

# ---------- Seção 3: Upload de imagem e predição ----------
st.subheader("Classificação de uma Imagem de Lâmina de Sangue")

st.markdown(
    """
    Envie uma imagem de uma célula sanguínea para que o modelo classifique
    entre as seguintes categorias (em português):

    - **Neutrófilo** (NEUTROPHIL)  
    - **Linfócito** (LYMPHOCYTE)  
    - **Monócito** (MONOCYTE)  
    - **Eosinófilo** (EOSINOPHIL)  
    """
)

with st.expander("O que são esses tipos de células sanguíneas?"):
    st.markdown(
        """
        **Neutrófilos**  
        São glóbulos brancos responsáveis pela **primeira linha de defesa** contra
        infecções, principalmente bactérias. São as células de defesa mais abundantes
        no sangue e chegam rápido ao local da inflamação.

        **Linfócitos**  
        Também são glóbulos brancos, mas atuam principalmente na **resposta imune
        específica**. Incluem linfócitos **T** e **B**, envolvidos na produção de
        anticorpos e na memória imunológica.

        **Monócitos**  
        São células que circulam no sangue e podem se transformar em **macrófagos**
        quando entram nos tecidos. Têm papel importante na **fagocitose** (englobar
        e “limpar” micro-organismos e restos celulares) e na regulação da resposta
        inflamatória.

        **Eosinófilos**  
        Participam principalmente das respostas contra **parasitas** e estão muito
        relacionados a **reações alérgicas** (como rinite alérgica, asma, etc.).
        Aumentos de eosinófilos podem indicar alergias ou alguns tipos de infecções
        parasitárias.
        """
    )


uploaded_file = st.file_uploader(
    "Selecione uma imagem (JPG ou PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Imagem enviada", use_column_width=True)
    st.info("A etapa de predição será implementada no próximo passo.")
