# Asistente de Análisis Clínico: Riesgo COPD y Guías Médicas

Proyecto de IA que integra análisis de datos clínicos, modelos predictivos de machine learning (XGBoost/Random Forest/MLP) y un sistema RAG (Retrieval-Augmented Generation) basado en Llama 3 para consulta de guías médicas.

## 🚀 Requisitos y Configuración

1. **Instalar dependencias**:
   Asegúrate de estar en un entorno virtual (recomendado Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys**:
   Configura tu clave gratuita de Groq (para el modelo Llama 3) en el archivo `.env`:
   ```env
   GROQ_API_KEY=tu_api_key_aqui
   ```
   Puedes obtenerla en [console.groq.com](https://console.groq.com/).

3. **Estructura de Datos**:
   Asegúrate de que tus datos estén estructurados:
   - Base de datos en: `DB/230PatientsCOPD.xlsx`
   - Guías PDF en: `PDF/*.pdf`

## 📂 Ejecución del Proyecto

El proyecto está dividido en bloques modulares. Deben ejecutarse en orden la primera vez:

### Bloque 1: Exploración y Limpieza
Ejecuta los notebooks interactivos (soportados nativamente en VS Code):
- `notebooks/01_eda.py`: Análisis exploratorio profundo, visualizaciones y tests estadísticos.
- `notebooks/02_preprocessing.py`: Pipeline de limpieza, manejo de outliers y encoding. Genera los datos en `/data/processed/`.

### Bloque 2: Modelado ML
- `notebooks/03_modeling.py`: Entrena Random Forest, XGBoost y un MLP en PyTorch. Guarda el mejor modelo en `/models/`.

### Bloque 3: Sistema RAG & Aplicación
- `notebooks/04_llm_rag.py`: (Opcional) Demostración en notebook de ingesta de PDFs a ChromaDB y consultas con Llama 3.

**Lanzar la Interfaz Web:**
Para utilizar el sistema unificado, lanza la aplicación Streamlit:
```bash
streamlit run app/main.py
```

## 🛠️ Tecnologías
- **Análisis**: `pandas`, `seaborn`, `scikit-learn`
- **Machine Learning**: `xgboost`, `imbalanced-learn`, `PyTorch`
- **RAG Stack**: `langchain`, `ChromaDB`, `HuggingFace (BAAI/bge-m3)`, `Groq (Llama 3)`
- **UI**: `Streamlit`
