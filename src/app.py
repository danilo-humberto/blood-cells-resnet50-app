import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

from db import log_interaction, get_last_interactions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_EN = ["EOSINOPHIL", "LYMPHOCYTE", "MONOCYTE", "NEUTROPHIL"]

CLASS_PT = {
    "EOSINOPHIL": "Eosinófilo",
    "LYMPHOCYTE": "Linfócito",
    "MONOCYTE": "Monócito",
    "NEUTROPHIL": "Neutrófilo",
}

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def load_model_resnet50():
    model = models.resnet50(weights=None)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(CLASS_EN))

    state_dict = torch.load("models/resnet50_best.pth", map_location=device)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model

def predict_image(model, file):
    image = Image.open(file).convert("RGB")

    tensor = image_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

    best_idx = int(np.argmax(probs))
    label_en = CLASS_EN[best_idx]
    label_pt = CLASS_PT[label_en]
    prob_best = float(probs[best_idx])

    probs_pt = {
        CLASS_PT[CLASS_EN[i]]: float(probs[i])
        for i in range(len(CLASS_EN))
    }

    return image, label_pt, prob_best, probs_pt

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

data = {
    "Modelo": ["DenseNet-161", "ResNet-50", "VGG-11", "DDRNet"],
    "Epochs": [12, 15, 15, 30],

    "Best Val Loss":  [0.0031, 0.0029, 0.0028, 1.3880],
    "Final Val Loss": [0.0045, 0.0149, 0.0028, 1.3880],

    "Best Val Acc":   [0.9995, 0.9995, 1.0000, None],
    "Final Val Acc":  [0.9985, 0.9965, 1.0000, None],

    "Best Val F1":    [0.9995, 0.9995, 1.0000, None],
    "Final Val F1":   [0.9985, 0.9965, 1.0000, None],

    "Test Acc":       [0.8762, 0.8854, 0.8762, 0.2460],
    "Test Precision": [0.9037, 0.9064, 0.8999, None],
    "Test Recall":    [0.8761, 0.8854, 0.8761, None],
    "Test F1":        [0.8795, 0.8880, 0.8793, 0.0987], 
}

df_results = pd.DataFrame(data)
st.dataframe(df_results, width='stretch')

df_top = df_results[df_results["Modelo"] != "DDRNet"]

st.subheader("Comparação Visual dos Modelos")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Acurácia e F1-Score no Conjunto de Teste**")
    fig, ax = plt.subplots()
    ax.bar(df_top["Modelo"],df_top["Test Acc"], label="Acurácia")
    ax.bar(df_top["Modelo"], df_top["Test F1"], alpha=0.7, label="F1-Score")
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

with col3:
    st.markdown("**DDRNet**")
    st.image("src/assets/ddrnet.png", width='stretch')

st.markdown("---")

st.subheader("Desempenho Detalhado do Modelo Final (ResNet-50)")

resCol1, resCol2 = st.columns(2)

with resCol1:
    st.markdown("### Gráfico de Loss por Época")
    st.image("src/assets/loss-resnet50.png", width='stretch')

with resCol2:
    st.markdown("### Matriz de Confusão (ResNet-50)")
    st.image("src/assets/confusion_matrix_resnet50.png", width='stretch')

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
    model = load_model_resnet50()
    
    image, label, prob_best, probs = predict_image(model, uploaded_file)
    st.image(uploaded_file, caption="Imagem enviada", width='stretch')
    st.success(
        f"Classe prevista: **{label}**\n\n"
        f"Confiança: **{prob_best * 100:.2f}%**"
    )

    prob_eosinophil = probs.get("Eosinófilo", 0.0)
    prob_lymphocyte = probs.get("Linfócito", 0.0)
    prob_monocyte   = probs.get("Monócito", 0.0)
    prob_neutrophil = probs.get("Neutrófilo", 0.0)
    
    filename = getattr(uploaded_file, "name", None)
    
    log_interaction(
        model_name="ResNet-50",
        predicted_class=label,
        confidence=prob_best,
        prob_eosinophil=prob_eosinophil,
        prob_lymphocyte=prob_lymphocyte,
        prob_monocyte=prob_monocyte,
        prob_neutrophil=prob_neutrophil,
        filename=filename,
    )

st.markdown("---")
st.subheader("Histórico recente de interações")

df_logs = get_last_interactions(limit=20)

if df_logs.empty:
    st.info("Nenhuma interação registrada ainda. Envie uma imagem para começar o histórico.")
else:
    st.dataframe(df_logs, width="stretch")
