import json
from pathlib import Path

# Debe coincidir con el archivo anterior
DEFAULT_JSON_PATH = r"C:\PersonalProjects\vnnUNet\data\nnUNet\nnUNet_preprocessed\Dataset099_LungTest\nnUNetPlans.json"
DEFAULT_MODE = "3d_fullres"
ARCH_PATH = r"extraction/network/nnunet_extracted_architecture.py"
TEST_PATH = r"C:\PersonalProjects\vnnUNet\extraction\network\test_architecture.py"

def generate_test_arquitecture(json_path, mode, nombre_clase):

    with open(json_path, "r") as f:
        plans = json.load(f)

    config = plans["configurations"][mode]
    arch_kwargs = config["architecture"]["arch_kwargs"]

    # Serializar kwargs correctamente
    kwargs_str = json.dumps(arch_kwargs, indent=4)
    kwargs_str = kwargs_str.replace("true", "True").replace("false", "False").replace("null", "None")

    code = f'''"""
Test para instanciar la arquitectura '{nombre_clase}'
"""

from nnunet_extracted_architecture import {nombre_clase}
import torch

# kwargs tomados desde el JSON
arch_kwargs = {kwargs_str}

# Instanciar modelo
model = {nombre_clase}(**arch_kwargs)

# Mostrar la red
print(model)

# Simular entrada
input_shape = [1, arch_kwargs['features_per_stage'][0]]
if 'Conv2d' in arch_kwargs['conv_op']:
    input_shape += [64, 64]
else:
    input_shape += [32, 64, 64]

x = torch.randn(*input_shape)
output = model(x)

print(f"Salida del modelo: shape = {{output.shape}}")
'''

    with open(TEST_PATH, "w") as f:
        f.write(code)

    print(f"[✓] Script de test guardado en: {TEST_PATH}")