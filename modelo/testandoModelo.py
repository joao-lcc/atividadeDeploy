import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt  # Import necessário para exibição de imagens
import json

# Caminho do modelo salvo
model_path = './modelo/modelo_fruits360_resnet50.h5'

# Carregar o modelo
model = tf.keras.models.load_model(model_path)

# Caminho para o arquivo JSON com as labels
labels_path = './class_labels.json'

# Carregar as labels do JSON
with open(labels_path, 'r') as json_file:
    class_labels = json.load(json_file)



# Função para carregar e pré-processar a imagem
def preprocess_image(image_path, target_size=(100, 100)):
    """
    Carrega uma imagem, redimensiona e normaliza para o modelo.
    """
    img = load_img(image_path, target_size=target_size)  # Carrega a imagem
    img_array = img_to_array(img)  # Converte para array NumPy
    img_array = img_array / 255.0  # Normaliza para o intervalo [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona uma dimensão batch
    return img, img_array

# Caminho da imagem para teste
image_path = './imagens/banana.png'  # Substitua pelo caminho da imagem

# Pré-processar a imagem
img, image = preprocess_image(image_path)

# Plotar a imagem original
plt.imshow(img)
plt.axis('off')  # Remove os eixos para melhor visualização
plt.title("Imagem de Entrada")
plt.show()

# Realizar a predição
predictions = model.predict(image).flatten()
# Obter as 3 maiores probabilidades e seus índices
top_3_indices = np.argsort(predictions)[-3:][::-1]  # Ordena e pega os 3 maiores índices
top_3_probs = predictions[top_3_indices]  # Probabilidades correspondentes
top_3_classes = [class_labels[str(idx)] for idx in top_3_indices]  # Classes correspondentes

# Mostrar o resultado
print(f"A classe prevista para a imagem é: {top_3_classes[0]} com probabilidade {top_3_probs[0]:.2f}")
print("Top 3 probabilidades:")
for i in range(3):
    print(f"Classe: {top_3_classes[i]} | Probabilidade: {top_3_probs[i]:.2f}")
