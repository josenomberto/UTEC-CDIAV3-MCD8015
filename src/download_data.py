#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script Reproducible de Obtención de Datos desde Google Drive
Proyecto 3: Precio Dinámico de Alojamientos (CDMX)
Curso: Proyecto Integrador I - UTEC Postgrado (MCD8015)

Soporta múltiples fuentes de obtención:
  1. Descarga desde AWS S3 Bucket (vía AWS CLI `aws s3 cp/sync` o `boto3`).
  2. Copia desde carpeta local / Google Drive montado (/content/drive/MyDrive/...).
  3. Descarga vía ID de Google Drive (empleando gdown).
"""

import os
import sys
import shutil
import argparse

# Intentar importar gdown para descargas remotas
#try:
#    import gdown
#    HAS_GDOWN = True
#except ImportError:
#    HAS_GDOWN = False

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

# URI por defecto de AWS S3 (Reemplazar con el bucket/URI exacto de AWS S3 de tu equipo)
DEFAULT_S3_URI = "s3://utec-cdiav3-mcd8015/data/raw/"

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


def download_from_s3(s3_uri):
    """Descarga los archivos desde un bucket de AWS S3 usando aws-cli o boto3."""
    print(f"\n☁️ Intentando descargar datos desde AWS S3: '{s3_uri}'")
    
    # 1. Intentar con AWS CLI (aws s3 cp / sync)
    try:
        if shutil.which("aws"):
            print("  -> Ejecutando descarga con AWS CLI (aws s3 cp)...")
            for f in FILES:
                src_path = os.path.join(s3_uri.rstrip('/'), f)
                dst_path = os.path.join(DATA_DIR, f)
                cmd = ["aws", "s3", "cp", src_path, dst_path, "--no-sign-request"]
                res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if res.returncode != 0:
                    # Reintentar sin --no-sign-request en caso de buckets privados con credenciales
                    cmd_cred = ["aws", "s3", "cp", src_path, dst_path]
                    subprocess.run(cmd_cred, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if verify_files():
                return True
    except Exception as e:
        print(f"  -> Advertencia AWS CLI: {e}")

    # 2. Intentar con Python boto3 si está instalado
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config

        print("  -> Ejecutando descarga con librería boto3...")
        clean_s3 = s3_uri.replace("s3://", "")
        parts = clean_s3.split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # Intento de descarga pública primero
        try:
            s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
            for f in FILES:
                key = os.path.join(prefix, f).lstrip('/')
                dst_path = os.path.join(DATA_DIR, f)
                print(f"  -> Descargando {f} desde s3://{bucket_name}/{key}...")
                s3_client.download_file(bucket_name, key, dst_path)
        except Exception:
            # Reintentar con cliente con credenciales autenticadas
            s3_client = boto3.client('s3')
            for f in FILES:
                key = os.path.join(prefix, f).lstrip('/')
                dst_path = os.path.join(DATA_DIR, f)
                print(f"  -> Descargando {f} (autenticado) desde s3://{bucket_name}/{key}...")
                s3_client.download_file(bucket_name, key, dst_path)

        if verify_files():
            return True
    except Exception as e:
        print(f"  -> Advertencia boto3: {e}")

    return False


def copy_from_local(source_folder):
    """Copia los archivos desde una carpeta local o Google Drive montado."""
    print(f"\n Intentando copiar datos desde carpeta local: '{source_folder}'")
    copied = 0
    for f in FILES:
        src = os.path.join(source_folder, f)
        dst = os.path.join(DATA_DIR, f)
        if os.path.exists(src):
            print(f"  -> Copiando {f}...")
            shutil.copy(src, dst)
            copied += 1
    return copied == len(FILES)


def main():
    parser = argparse.ArgumentParser(description="Obtención de datos para EDA reproducible.")
    parser.add_argument("--s3-uri", type=str, default=DEFAULT_S3_URI,
                        help="URI del bucket de AWS S3 (ej. s3://mi-bucket/data/raw/).")
    parser.add_argument("--drive-path", type=str, default=COLAB_DRIVE_PATH,
                        help="Ruta a la carpeta local o Google Drive.")
    args = parser.parse_args()

    setup_data_dir()

    # 1. Si los archivos ya existen localmente en data/raw/, no volver a descargar
    if verify_files():
        print("\n Todos los archivos Parquet ya están disponibles en data/raw/. Listo para el EDA.")
        sys.exit(0)

    # 2. Intentar descarga desde AWS S3
    if args.s3_uri and not args.s3_uri.startswith("s3://mi-bucket-airbnb-cdmx"):
        if download_from_s3(args.s3_uri):
            print("\n Datos descargados exitosamente desde AWS S3.")
            sys.exit(0)

    # 3. Intentar copia local / Google Drive montado
    if os.path.exists(args.drive_path):
        if copy_from_local(args.drive_path) and verify_files():
            print("\n Datos copiados exitosamente desde carpeta local/Drive.")
            sys.exit(0)

    # 4. Reintentar S3 por defecto si la ruta local no existe
    if download_from_s3(args.s3_uri):
        print("\n Datos descargados exitosamente desde AWS S3.")
        sys.exit(0)

    print("\n No se pudieron obtener los archivos automáticamente.")
    print("Coloca los archivos parquet en 'data/raw/' o especifica el URI exacto de AWS S3 ejecutando: python download_data.py --s3-uri s3://tu-bucket/carpeta/")


if __name__ == "__main__":
    main()
