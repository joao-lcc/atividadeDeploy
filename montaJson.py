import os
import json

# Caminho para o diretório de treino
train_dir = r"C:\\Users\\joaoc\\.cache\\kagglehub\\datasets\\moltean\\fruits\\versions\\11\\fruits-360_dataset_100x100\\fruits-360\\Training"

# Obter as classes
class_labels = {i: class_name for i, class_name in enumerate(sorted(os.listdir(train_dir)))}

# Caminho para salvar o arquivo JSON
output_path = './class_labels.json'

# Salvar as labels no arquivo JSON
with open(output_path, 'w') as json_file:
    json.dump(class_labels, json_file)

print(f"Labels salvas no arquivo: {output_path}")
