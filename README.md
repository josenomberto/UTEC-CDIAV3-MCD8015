# Proyecto 3 — Sistema Inteligente de Scoring y Selección de Propiedades para Subarriendo Turístico

## Marco CRISP-DM · Etapas E1 (Comprensión del Negocio) y E2 (Comprensión de los Datos)

### UTEC Postgrado — Maestría en Ciencia de Datos e Inteligencia Artificial

**Curso**: Proyecto Integrador I
**Fecha de Cierre (S02)**: 12 de septiembre de 2026

---

### Integrantes y Roles

* **Product Lead**: David Jimenez (*Investigación de negocio, stakeholders, propuesta de valor y métricas*)
* **Data Scientist**: Herles Pinedo (*EDA, perfilado de calidad, baselines, modelos y NLP*)
* **ML Engineer**: José Carlos Nomberto (*Reproducibilidad, pipeline de datos, automatización y despliegue*)

**Repositorio Oficial Git**: [https://github.com/josenomberto/UTEC-CDIAV3-MCD8015](https://github.com/josenomberto/UTEC-CDIAV3-MCD8015)

---

### Instrucción Única de Ejecución (Bloque 9 — Repositorio Reproducible)

Para reproducir todo el Análisis Exploratorio de Datos (EDA) y verificar la consistencia de los datos del reporte S02 de forma 100% automática, ejecute en la terminal:

```bash
git clone https://github.com/josenomberto/UTEC-CDIAV3-MCD8015.git
cd UTEC-CDIAV3-MCD8015
make eda
```

#### ¿Qué ejecuta el comando `make eda` internamente?

1. **`make setup`**: Crea el entorno virtual `venv` e instala las dependencias exactas desde `requirements.txt`.
2. **`make download_data`**: Descarga de forma directa y limpia los datasets relacionales en formato Parquet a `data/raw/`.
3. **Ejecución del EDA**: Ejecuta el pipeline reproducible (`src/01_eda.py` / `notebooks/01_eda.ipynb`) fijando semillas aleatorias (`random_state=42`), generando las figuras e imprimiendo la matriz de calidad.

---

### Resumen de Datos Recolectados (Bloque 6)

| Tabla                                              | Registros | Variables | Periodo Cubierto              | Variable Objetivo / Clave              |  Estado del Acceso  |
| :------------------------------------------------- | :--------: | :-------: | :---------------------------- | :------------------------------------- | :------------------: |
| **Anuncios** (`airbnb_listings.parquet`)   |   31,430   |    90    | Corte a Junio 2026            | `price` (Tarifa por noche)           | **Disponible** |
| **Calendario** (`airbnb_calendar.parquet`) | 11,471,961 |     5     | Proyección 365 días futuros | `price` / `available` (Ocupación) | **Disponible** |
| **Reseñas** (`airbnb_reviews.parquet`)    | 1,719,730 |     6     | Histórico acumulado          | Comentarios de texto libre             | **Disponible** |

---

### Tres Hallazgos Críticos del EDA (Bloque 7)

1. **Formatos de Moneda, Asimetría Positiva y Outliers Defensivos**:

   * **Cifra**: El 100% de los precios ingresó como texto (`$2,913.69 MXN` promedio inicial), con asimetría severa y bloqueos defensivos de hasta \$1,141,520 MXN.
   * **Consecuencia Técnica**: Sanitización regex a `float32`, transformación logarítmica $\log(y)$ en E3 y recorte interpercentil P1–P99 (\$337.79 a \$20,292.43 MXN) para evitar la contaminación de varianza en los algoritmos de regresión.
   * **Evidencia**: `src/figures/eda_outliers_detection.png` y Sección E2.3 del notebook.
2. **Multimodalidad, Anuncios Fantasma y Filtro de Estadía Turística**:

   * **Cifra**: Mínimos de estadía de hasta 729 noches (0.82% excede 30 días) y bimodalidad extrema en disponibilidad anual (picos en 0 y 365 días).
   * **Consecuencia Técnica**: Filtro analítico estricto `minimum_nights <= 30` (construyendo un subset limpio de 28,981 anuncios activos de corta estancia) y aislamiento de propiedades inactivas (`availability_365 == 0`).
   * **Evidencia**: `src/figures/eda_minimum_nights.png` y `src/figures/eda_kelleher_histograms.png`.
3. **Sesgo Territorial y Ausencia Informativa (MAR)**:

   * **Cifra**: El 72.99% de la oferta se concentra en Cuauhtémoc (45.97%), Miguel Hidalgo (15.49%) y Benito Juárez (11.53%). Además, el 17.50% de nulos en calificaciones coincide en un 100.00% con anuncios sin reseñas (MAR).
   * **Consecuencia Técnica**: Creación del indicador `is_new_listing = 1` para modelar la falta de historial como señal, y codificación espacial agrupada (*Target Encoding*) para controlar la alta cardinalidad por alcaldía.
   * **Evidencia**: `src/figures/eda_missings.png` y `src/figures/eda_territorial_bias.png`.

---

### Calidad de Datos Declarada (Bloque 8)

* **Faltantes**: Quantificados por variable crítica (`review_scores_rating`: 17.50%, `bedrooms`: 16.98%, `price`: 5.70%).
* **Duplicados e Inconsistencias**: 0 duplicados en llaves primarias; 0 registros huérfanos entre tablas.
* **Atípicos**: Recorte P1–P99 en tarifas y cota máxima de 30 noches de estadía mínima.
* **Sesgo**: Muestra acotada al mercado de oferta turística publicada en Airbnb en Ciudad de México a junio de 2026.
* **Veredicto Final**: **Sí alcanzan con las mitigaciones anotadas**.

---

### Viabilidad Declarada (Bloque 10)

* **Técnica (Alta)**: Existencia de volumen de señal relacional (31k anuncios y 11.47M de noches) en formato Parquet optimizado.
* **Financiera (Alta)**: Tarificadores comerciales cobran ~1% de facturación; el incremento de ocupación del 15% paga sobradamente la solución SaaS.
* **Comercial (Alta)**: Cientos de miles de anfitriones en LatAm fijan tarifas a mano; mercado masivo de autoservicio web.

---

### Arquitectura del Repositorio

```text
.
├── Makefile                     # Automatización de tareas (setup, download_data, eda)
├── README.md                    # Portada verificable del Bloque 9
├── requirements.txt             # Dependencias estrictas del entorno
├── data/
│   └── raw/                     # Datasets descargados (.parquet)
└── src/
    ├── 01_eda.py                # Script ejecutable del EDA
│   └── 01_eda.ipynb             # Notebook ejecutable con auditoría completa  
    └── figures/                 # Gráficos vectoriales generados por el EDA
```
