import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt  # Import necessário para exibição de imagens
import json
import zipfile
import os
import io
# Função para criar o ZIP com imagens JPG
def criar_zip_com_imagens(pasta):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for raiz, _, arquivos in os.walk(pasta):
            for arquivo in arquivos:
                if arquivo.endswith(".jpg"):  # Filtrar apenas arquivos JPG
                    caminho_completo = os.path.join(raiz, arquivo)
                    zip_file.write(caminho_completo, os.path.relpath(caminho_completo, pasta))
    buffer.seek(0)
    return buffer

pastaImagens = "./imagens/"
arquivo_zip = criar_zip_com_imagens(pastaImagens)

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

# Caminho para o arquivo JSON com as labels
labels_path = './class_labels.json'

# Carregar as labels do JSON
with open(labels_path, 'r') as json_file:
    class_labels = json.load(json_file)

# Título e menu de navegação
st.title("Classificação de frutas e legumes utilizando ResNet-50")
st.sidebar.title("Navegação")
selecao = st.sidebar.radio("Escolha a seção", ["Visão Geral", "Importar imagem", "Informações sobre o dataset", "Informações sobre o modelo"])

# Seção de Visão Geral
if selecao == "Visão Geral":
    st.header("Visão Geral")
    st.write("Olá! Este é um aplicativo para classificação de frutas e legumes. Você pode importar uma imagem de um fruta ou legume e o modelo irá prever a classe da imagem. Além disso, você pode visualizar informações sobre o dataset e o modelo utilizado. Selecione uma aba no menu lateral e descubra mais!.")
    st.write("Este projeto serve como atividade avaliativa para a disciplina de inteligência artificial do curso de Ciências da Computação, UNESP Bauru.")

# Seção de Importação das imagens
elif selecao == "Importar imagem":
    st.header("Importando imagem")
    st.write("Nesta página, você pode importar uma imagem de uma fruta ou legume e o modelo irá prever a classe da imagem. Faça o upload de uma imagem no formato JPG para ver o resultado.")
    st.write("Se precisar, utilize o arquivo .ZIP abaixo com imagens de exemplo para testar.")
    st.download_button(
    label="Baixar ZIP com imagens",
    data=arquivo_zip,
    file_name="imagens.zip",
    mime="application/zip"
    )
    uploaded_file = st.file_uploader("Arquivos JPEG ou PNG", type=["jpg", "png"])
    if uploaded_file is not None:
        st.write("Imagem importada com sucesso!")
        # Exibir a imagem importada
        st.image(uploaded_file, caption='Imagem importada', use_column_width=True)

        # Pré-processar a imagem
        img, image = preprocess_image(uploaded_file)
        predictions = modelResNet.predict(image).flatten()
       # Obter as 3 maiores probabilidades e seus índices
        top_3_indices = np.argsort(predictions)[-3:][::-1]  # Ordena e pega os 3 maiores índices
        top_3_probs = predictions[top_3_indices]  # Probabilidades correspondentes
        top_3_classes = [class_labels[str(idx)] for idx in top_3_indices]  # Classes correspondentes

        st.write("Resultados das previsões:")

        # Mostrar o resultado
        st.write(f"A classe prevista para a imagem é: {top_3_classes[0]} com probabilidade {top_3_probs[0]:.2f}")
        st.write("As 3 maiores probabilidades entre as classes do dataset foram:")
        for i in range(3):
            st.write(f"Classe: {top_3_classes[i]} | Probabilidade: {top_3_probs[i]:.2f}")
    
    
elif selecao == "Informações sobre o dataset":
    st.header("Informações do dataset")
    st.write("Número de imagens: 94110")
    st.write("Número de classes: 141")
    st.write("Classes do dataset:")
    st.write(list(class_labels.values()))
    st.write("Exemplo de imagens do dataset:")
    st.image("./imagens/pessego.jpg", caption='Exemplo de imagens da classe "Peach 1', use_column_width=True)
    st.image("./imagens/maçã.jpg", caption='Exemplo de imagens da classe "Apple 1"', use_column_width=True)

elif selecao == "Informações sobre o modelo":
    st.header("Informações sobre o modelo")
    st.write("O modelo ResNet-50 é uma rede neural convolucional profunda com 50 camadas, desenvolvida pela Microsoft. A arquitetura ResNet, abreviação de Residual Network, foi introduzida em 2015 e revolucionou o campo de visão computacional ao vencer a competição ImageNet. A principal inovação da ResNet é o uso de conexões residuais, que permitem que os gradientes fluam diretamente através da rede, facilitando o treinamento de redes muito profundas. Essas conexões residuais ajudam a mitigar o problema do desaparecimento do gradiente, comum em redes profundas. O ResNet-50 é amplamente utilizado em tarefas de classificação de imagens devido à sua capacidade de extrair características ricas e discriminativas das imagens. Ele é treinado em um grande conjunto de dados e pode ser adaptado para diferentes tarefas de visão computacional, como detecção de objetos e segmentação de imagens. A arquitetura ResNet-50 é composta por blocos residuais, cada um contendo várias camadas convolucionais, seguidas por normalização em lote e funções de ativação ReLU. No final, há uma camada totalmente conectada que produz as previsões finais. A ResNet-50 é conhecida por seu equilíbrio entre profundidade e eficiência computacional, tornando-a uma escolha popular para muitos aplicativos de visão computacional.")

