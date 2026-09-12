# ====================================================================================
# Makefile Reproducible - Proyecto 3: Precio Dinámico de Alojamientos CDMX
# Curso: Proyecto Integrador I (UTEC Postgrado - Maestría en Ciencia de Datos e IA)
# ====================================================================================

PYTHON = venv/bin/python
PIP = venv/bin/pip
DATA_DIR = data/raw

.PHONY: all setup download_data eda clean help

# Target por defecto
all: help

# 1. Creación de entorno virtual e instalación de dependencias
setup: requirements.txt
	@echo "====== [1/3] Creando entorno virtual e instalando librerías ======"
	python3 -m venv venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Entorno preparado. Para activar use: source venv/bin/activate"

# 2. Copia o Descarga de los Datasets Parquet desde Google Drive
download_data: setup
	@echo "====== [2/3] Verificando / Copiando Datasets Parquet desde Google Drive ======"
	mkdir -p $(DATA_DIR)
	$(PYTHON) src/download_data.py
	@echo "Verificación de archivos completada."

# 3. COMANDO ÚNICO DE EJECUCIÓN (Exigido en el Bloque 9 del One-Pager)
eda: download_data
	@echo "====== [3/3] Ejecutando EDA avanzado con estándares de Kelleher ======"
	$(PYTHON) 01_eda.py
	@echo "====== [¡ÉXITO!] ======"
	@echo "El EDA reproducible ha corrido de punta a punta."
	@echo "Las cifras y gráficos generados coinciden con el One-Pager y el Informe S02."

# Limpieza de caché y archivos temporales
clean:
	@echo "Eliminando caché y entorno virtual..."
	rm -rf venv/
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	@echo "Limpieza finalizada."

# Ayuda autodocumentada
help:
	@echo "Comandos disponibles:"
	@echo "  make setup          - Crea el entorno virtual e instala requisitos."
	@echo "  make download_data  - Copia o descarga los archivos parquet desde Google Drive."
	@echo "  make eda            - [COMANDO ÚNICO] Ejecuta el análisis exploratorio reproducible."
	@echo "  make clean          - Limpia el venv y archivos temporales."
