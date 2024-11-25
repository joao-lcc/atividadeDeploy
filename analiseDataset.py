import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Caminhos do dataset
train_dir = "C:\\Users\\joaoc\\.cache\\kagglehub\\datasets\\moltean\\fruits\\versions\\11\\fruits-360_dataset_100x100\\fruits-360\\Training"
test_dir = "C:\\Users\\joaoc\\.cache\\kagglehub\\datasets\\moltean\\fruits\\versions\\11\\fruits-360_dataset_100x100\\fruits-360\Test"

# Pré-processamento dos dados
train_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalização
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Geradores de dados
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),  # Tamanho das imagens
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),
    batch_size=32,
    class_mode='categorical'
)

print("Número de classes:", train_generator.num_classes)
print("Distribuição das classes:")
print(train_generator.class_indices)


x_batch, y_batch = next(train_generator)
for i in range(5):
    plt.imshow(x_batch[i])
    plt.title(f"Classe: {np.argmax(y_batch[i])}")
    plt.show()
