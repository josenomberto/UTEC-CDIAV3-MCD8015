#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Reproducible de Obtención de Datos desde Google Drive
Proyecto 3: Precio Dinámico de Alojamientos (CDMX)
Curso: Proyecto Integrador I - UTEC Postgrado (MCD8015)

Soporta dos modalidades:
  1. Copia directa desde Google Drive montado en Google Colab (/content/drive/MyDrive/...).
  2. Descarga vía ID de Google Drive (empleando gdown).
"""

import os
import sys
import shutil
import argparse

# Intentar importar gdown para descargas remotas
try:
    import gdown
    HAS_GDOWN = True
except ImportError:
    HAS_GDOWN = False

# ==============================================================================
# CONFIGURACIÓN DE ARCHIVOS Y RUTAS
# ==============================================================================
DATA_DIR = os.path.join("data", "raw")
FILES = [
    "airbnb_listings.parquet",
    "airbnb_calendar.parquet",
    "airbnb_reviews.parquet"
]

# IDs de Google Drive
GDRIVE_IDS = {
    "airbnb_listings.parquet": "17WNpcVw6qJY6fLn7zv4oEjNohSp0yhPv",
    "airbnb_calendar.parquet": "10RsoHmS2beTnBeh8xt7jI3Uhg578sZEY",
    "airbnb_reviews.parquet":  "1wlldpZxsqKI4DcspsBd14hqaBNgiEvQm"
}

# Ruta predeterminada en Google Colab si la carpeta está compartida en MyDrive
COLAB_DRIVE_PATH = "/content/drive/MyDrive/grupo_07_airbnb_cdmx/01_dataset"


def setup_data_dir():
    """Crea la carpeta data/raw si no existe."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"📁 Directorio de destino verificado: '{DATA_DIR}/'")


def verify_files():
    """Verifica la existencia y tamaño de los archivos parquet en data/raw."""
    missing = []
    print("\n🔍 Verificando archivos en data/raw/:")
    for f in FILES:
        path = os.path.join(DATA_DIR, f)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  [PASS] {f} ({size_mb:.2f} MB)")
        else:
            print(f"  [FAIL] {f} (No encontrado o vacío)")
            missing.append(f)
    return len(missing) == 0


def copy_from_mounted_drive(source_folder):
    """Copia los archivos directamente desde Google Drive montado en Colab o local."""
    print(f"\nIntentando copiar datos desde carpeta compartida: '{source_folder}'")
    copied = 0
    for f in FILES:
        src = os.path.join(source_folder, f)
        dst = os.path.join(DATA_DIR, f)
        if os.path.exists(src):
            print(f"  -> Copiando {f}...")
            shutil.copy(src, dst)
            copied += 1
        else:
            print(f"  -> [ADVERTENCIA] No se encontró {src}")
    return copied == len(FILES)


def download_from_gdrive():
    """Descarga los archivos desde Google Drive usando gdown."""
    if not HAS_GDOWN:
        print("Error: Librería 'gdown' no instalada. Ejecute 'pip install gdown'.")
        return False

    print("\n Descargando archivos Parquet desde Google Drive (gdown)...")
    for fname, fid in GDRIVE_IDS.items():
        dst = os.path.join(DATA_DIR, fname)
        if fid.startswith("COLOCAR_AQUI"):
            print(f" Ignorando {fname}: ID de Google Drive no configurado.")
            continue
        url = f"https://drive.google.com/uc?id={fid}"
        print(f"  -> Descargando {fname}...")
        gdown.download(url, dst, quiet=False)
    return verify_files()


def main():
    parser = argparse.ArgumentParser(description="Obtención de datos para EDA reproducible.")
    parser.add_argument("--drive-path", type=str, default=COLAB_DRIVE_PATH,
                        help="Ruta a la carpeta compartida en Google Drive (Colab).")
    args = parser.parse_args()

    setup_data_dir()

    # 1. Si ya existen, no descargar/copiar de nuevo
    if verify_files():
        print("\n Todos los archivos Parquet ya están disponibles en data/raw/. Listo para el EDA.")
        sys.exit(0)

    # 2. Intentar copiar desde Google Drive montado (Colab / Local)
    if os.path.exists(args.drive_path):
        if copy_from_mounted_drive(args.drive_path) and verify_files():
            print("\n Datos copiados exitosamente desde Google Drive.")
            sys.exit(0)

    # 3. Intentar descarga por gdown
    if download_from_gdrive():
        print("\n Datos descargados exitosamente vía gdown.")
        sys.exit(0)

    print("\n No se pudieron obtener los archivos automáticamente.")
    print("Asegúrate de haber montado Google Drive en Colab o de haber colocado los archivos en 'data/raw/'.")


if __name__ == "__main__":
    main()
