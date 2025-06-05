import json
import os
from pathlib import Path
import requests
import re
from utils.generate_test_arquitecture import generate_test_arquitecture

# Ruta base para guardar todo lo generado
NETWORK_DIR = Path(__file__).parent / "network"
NETWORK_DIR.mkdir(parents=True, exist_ok=True)

# Valores por defecto
DEFAULT_JSON_PATH = r"C:\PersonalProjects\vnnUNet\data\nnUNet\nnUNet_preprocessed\Dataset099_LungTest\nnUNetPlans.json"
DEFAULT_MODE = "3d_fullres"
DEFAULT_OUTPUT_PATH = NETWORK_DIR / "nnunet_extracted_architecture.py"

CLASS_PATH_MAPPING = {
    "PlainConvUNet": "dynamic_network_architectures/architectures/unet.py",
    "ResidualEncoderUNet": "dynamic_network_architectures/architectures/residual_unet.py",
}

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/MIC-DKFZ/dynamic-network-architectures/master/"

def descargar_dependencias(imports: list):
    """
    Descarga todos los archivos necesarios según los imports detectados en network/.
    """
    archivos_descargados = set()

    for imp in imports:
        if not imp.startswith("from dynamic_network_architectures"):
            continue

        partes = imp.split()
        if "import" not in partes:
            continue

        try:
            path = partes[1]
            archivo_py = path.split(".")
            if len(archivo_py) < 2:
                continue

            ruta_relativa = "/".join(archivo_py) + ".py"
            if ruta_relativa in archivos_descargados:
                continue

            url = GITHUB_RAW_BASE + ruta_relativa
            local_path = NETWORK_DIR / ruta_relativa
            local_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"⬇️  Descargando {ruta_relativa} ...")

            r = requests.get(url)
            if r.status_code == 200:
                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                archivos_descargados.add(ruta_relativa)
            else:
                print(f"[✗] No se pudo descargar {ruta_relativa} (status {r.status_code})")

        except Exception as e:
            print(f"[✗] Error procesando import '{imp}': {e}")

def obtener_codigo_y_imports(nombre_clase, ruta_archivo):
    url = GITHUB_RAW_BASE + ruta_archivo
    respuesta = requests.get(url)
    if respuesta.status_code != 200:
        print(f"[✗] Error al obtener el archivo desde {url}")
        return None, None

    lineas = respuesta.text.splitlines()

    # Paso 1: Extraer imports (multilínea incluido)
    imports_raw = []
    collecting = False
    current_import = []

    for linea in lineas:
        stripped = linea.strip()

        if stripped.startswith("from ") or stripped.startswith("import "):
            current_import = [linea]
            if stripped.endswith("("):
                collecting = True
                continue
            else:
                imports_raw.append(linea)

        elif collecting:
            current_import.append(linea)
            if ")" in stripped:
                collecting = False
                full_import = "\n".join(current_import)
                imports_raw.append(full_import)
                current_import = []

        if stripped.startswith(f"class {nombre_clase}"):
            break

    # Paso 2: Extraer cuerpo de la clase
    codigo = []
    dentro = False
    indent = None
    for linea in lineas:
        if linea.strip().startswith(f"class {nombre_clase}"):
            dentro = True
            indent = len(linea) - len(linea.lstrip())
            codigo.append(linea)
        elif dentro:
            if len(linea.strip()) > 0 and len(linea) - len(linea.lstrip()) <= indent:
                break
            codigo.append(linea)

    # Paso 3: Filtrar imports usados en el cuerpo de clase
    cuerpo_clase = "\n".join(codigo)
    imports_utiles = []

    for imp in imports_raw:
        identificadores = set()

        # Ejemplo: from torch import nn, functional => extrae 'nn', 'functional'
        if "import" in imp:
            partes = imp.split("import", 1)[1]
            partes = partes.replace("(", "").replace(")", "")
            for part in partes.split(","):
                part = part.strip().split(" as ")[-1].strip()
                if part:
                    identificadores.add(part)

        # Mantener si alguno se usa en el código
        if any(ident in cuerpo_clase for ident in identificadores):
            imports_utiles.append(imp)

    return imports_utiles, codigo

def main():
    json_path = input("Ruta al nnUNetPlans.json (Enter para usar valor por defecto): ").strip() or DEFAULT_JSON_PATH
    mode = input("Modo de red (2d, 3d_fullres, etc.) (Enter para usar valor por defecto): ").strip() or DEFAULT_MODE

    json_path = Path(json_path)
    if not json_path.exists():
        print(f"[✗] El archivo {json_path} no existe.")
        return

    with open(json_path, "r") as f:
        plans = json.load(f)

    try:
        network_class_path = plans["configurations"][mode]["architecture"]["network_class_name"]
        nombre_clase = network_class_path.split(".")[-1]
    except KeyError:
        print(f"[✗] No se encontró arquitectura para modo '{mode}' en el JSON.")
        return

    if nombre_clase not in CLASS_PATH_MAPPING:
        print(f"[✗] Clase '{nombre_clase}' no está mapeada.")
        return

    ruta_archivo = CLASS_PATH_MAPPING[nombre_clase]
    imports, codigo = obtener_codigo_y_imports(nombre_clase, ruta_archivo)
    if not codigo:
        return

    with open(DEFAULT_OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write('"""\nAuthor: Martin Cartajena, auto-generado desde nnUNetPlans.json\n"""\n\n')
        for imp in imports:
            f.write(imp + "\n")
        f.write("\n\n")
        for linea in codigo:
            f.write(linea + "\n")

    print(f"[✓] Clase '{nombre_clase}' exportada con imports a '{DEFAULT_OUTPUT_PATH}'")

    descargar_dependencias(imports)

    # Ejecutar generador de test
    generate_test_arquitecture(json_path, mode, nombre_clase)

if __name__ == "__main__":
    main()
