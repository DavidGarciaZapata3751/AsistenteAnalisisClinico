# Asistente de Analisis Clinico: Riesgo COPD y Guias Medicas

Proyecto de IA que integra analisis de datos clinicos, modelos predictivos de machine learning (XGBoost, Random Forest, MLP) y un sistema RAG (Retrieval-Augmented Generation) basado en Llama 3 para consulta de guias medicas.

## Configuracion para Git y GitHub

Este repositorio cuenta con un archivo .gitignore configurado para excluir archivos locales temporales, credenciales y archivos pesados generados por el pipeline. Al subir o clonar este repositorio desde GitHub, debes tener en cuenta que ciertos archivos no estaran presentes y deberas configurarlos o generarlos manualmente.

### Archivos excluidos por .gitignore que debes recrear:

1. Archivo de configuracion .env (contiene tus credenciales privadas).
2. Datos procesados (se ubican en data/processed/ y se generan ejecutando el script de preprocesamiento).
3. Modelos entrenados (archivos .joblib y .pth en la carpeta models/).
4. Base de datos vectorial (carpeta chroma_db/, la cual se genera al indexar los PDFs).

---

## Requisitos y Configuracion Inicial

Sigue estos pasos para poner en marcha el proyecto despues de clonarlo:

### 1. Instalar dependencias
Asegurate de estar en un entorno virtual (recomendado Python 3.10+):
```bash
pip install -r requirements.txt
```

### 2. Archivo de entorno (.env)
Crea un archivo llamado `.env` en la raiz del proyecto y añade tu API Key de Groq:
```env
GROQ_API_KEY=tu_api_key_aqui
```
Puedes obtener una clave de API gratuita en console.groq.com.

### 3. Estructura de Datos y PDFs
Coloca los archivos de datos y guias medicas en sus respectivas carpetas:
- Base de datos en: `DB/230PatientsCOPD.xlsx`
- Guias PDF en: `PDF/*.pdf`

---

## Ejecucion del Proyecto

Debido a que los datos procesados, los modelos de aprendizaje automatico y la base de datos vectorial estan excluidos del control de versiones por rendimiento y privacidad, debes ejecutar el pipeline en el siguiente orden la primera vez:

### Bloque 1: Exploracion y Limpieza
Ejecuta los notebooks interactivos o scripts en orden:
- `notebooks/01_eda.py`: Realiza el analisis exploratorio profundo, visualizaciones y pruebas estadisticas.
- `notebooks/02_preprocessing.py`: Ejecuta el pipeline de limpieza, manejo de outliers e imputacion. Genera los archivos en `data/processed/`.

### Bloque 2: Modelado ML
- `notebooks/03_modeling.py`: Entrena los modelos de Random Forest, XGBoost y la red neuronal MLP en PyTorch. Guarda los pesos y modelos optimizados en la carpeta `models/`.

### Bloque 3: Ingesta de PDFs para el Sistema RAG
Antes de iniciar el chatbot, debes indexar los documentos PDF en la base de datos vectorial local (ChromaDB):
```bash
python ingest_pdfs.py
```
Este paso creara la carpeta `chroma_db/` que contiene la base de conocimiento estructurada.

### Bloque 4: Lanzar la Aplicacion Web
Una vez generados los datos, entrenados los modelos y construido el vector store, inicia la interfaz web interactiva con Streamlit:
```bash
streamlit run app/main.py
```

---

## Tecnologias Utilizadas

- Analisis de datos: pandas, seaborn, scikit-learn
- Aprendizaje Automatico y Deep Learning: xgboost, imbalanced-learn, PyTorch
- Arquitectura RAG: langchain, ChromaDB, HuggingFace (BAAI/bge-m3), Groq (Llama 3)
- Evaluacion de RAG: ragas, datasets
- Interfaz de usuario: Streamlit

---
## Link del Video Youtube
```bash
https://www.youtube.com/watch?v=3vyvfYfrvow
```

## Integrantes
David García Zapata (dgarciaz1@eafit.edu.co)
Samuel LLano Madrigal (sllanom@eafit.edu.co)

