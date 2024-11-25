import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

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

# Carregar modelo pré-treinado (ResNet50)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
base_model.trainable = True  # Congela as camadas do modelo base

# Adicionar camadas específicas para o problema
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compilação do modelo
model.compile(optimizer=Adam(learning_rate=1e-5),  # Teste valores como 1e-4 ou 1e-5
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Resumo do modelo
model.summary()

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
]

# Treinamento do modelo
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=20,  # Ajuste o número de épocas conforme necessário,
    callbacks=callbacks,
    verbose=1
)

# Avaliação do modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}")

# Salvar o modelo treinado
model.save('modelo_fruits360_resnet50.h5')
print("Modelo salvo como 'modelo_fruits360_resnet50.h5'.")
