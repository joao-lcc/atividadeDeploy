import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt  # Import necessário para exibição de imagens
import os


def preprocess_image(image_path, target_size=(100, 100)):
    """
    Carrega uma imagem, redimensiona e normaliza para o modelo.
    """
    img = load_img(image_path, target_size=target_size)  # Carrega a imagem
    img_array = img_to_array(img)  # Converte para array NumPy
    img_array = img_array / 255.0  # Normaliza para o intervalo [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão batch
    return img, img_array


# Caminho do modelo salvo
model_path = './modelo/modelo_fruits360_resnet50.h5'

# Carregar o modelo
modelResNet = tf.keras.models.load_model(model_path)
modelos = {"ResNet 50": modelResNet}


# Caminho para o diretório de treino do dataset
train_dir = "C:\\Users\\joaoc\\.cache\\kagglehub\\datasets\\moltean\\fruits\\versions\\11\\fruits-360_dataset_100x100\\fruits-360\\Training"

# Obter os nomes das classes a partir das subpastas
class_labels = {i: class_name for i, class_name in enumerate(sorted(os.listdir(train_dir)))}

# Título e menu de navegação
st.title("Classificação de frutas e legumes")
st.sidebar.title("Navegação")
selecao = st.sidebar.radio("Escolha a seção", ["Visão Geral", "Importar imagem", "Estatísticas Descritivas"])

# Seção de Visão Geral
if selecao == "Visão Geral":
    st.header("Visão Geral")
    st.write("Este dashboard permite que você importe uma imagem de uma fruta, fruto ou legume e obtenha a classificação da espécie.")
    
    # Importação da imagem
    st.subheader("Importe sua imagem")
    uploaded_file = st.file_uploader("Faça upload de um arquivo JPG", type="jpg")
    if uploaded_file is not None:
        st.write("Imagem importada com sucesso!")
        
        # Fazer a previsão da imagem
        # Pré-processar a imagem
        img, image = preprocess_image(uploaded_file)
        predictions = modelResNet.predict(image).flatten()
       # Obter as 3 maiores probabilidades e seus índices
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Ordena e pega os 3 maiores índices
        top_3_probs = predictions[top_3_indices]  # Probabilidades correspondentes
        top_3_classes = [class_labels[idx] for idx in top_3_indices]  # Classes correspondentes

        st.write("Resultados das previsões:")

        # Mostrar o resultado
        st.write(f"A classe prevista para a imagem é: {top_3_classes[0]} com probabilidade {top_3_probs[0]:.2f}")
        st.write("Top 3 probabilidades:")
        for i in range(3):
            st.write(f"Classe: {top_3_classes[i]} | Probabilidade: {top_3_probs[i]:.2f}")

# Seção de Classificação de Espécies
elif selecao == "Importar imagem":
    st.header("Importando imagem")
    st.write("oi")
    
    
elif selecao == "Estatísticas Descritivas":
    st.header("Estatísticas Descritivas")
    st.write("oi 2")

