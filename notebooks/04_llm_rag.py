# %% [markdown]
# # 🤖 Notebook 04: Sistema RAG (Retrieval-Augmented Generation)
# ## Asistente de Consulta de Guías Médicas
#
# **Objetivo:** Construir y evaluar un sistema RAG para responder preguntas
# basándose en guías médicas en PDF. Usa PyMuPDF, ChromaDB, BAAI/bge-m3,
# y Llama 3 (vía Groq API). Finalmente se evalúa con RAGAS.

# %%
# ── Imports ─────────────────────────────────────────────────────────────────
import sys
import os
from pathlib import Path
import warnings

# Silenciar warnings de transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import (
    PDF_DIR,
    CHROMA_DIR,
    ingest_pdfs,
    create_vectorstore,
    load_vectorstore,
    get_embeddings,
    get_llm,
    build_rag_chain,
    query_rag,
    evaluate_rag
)

# Verificar API Key
from dotenv import load_dotenv
load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    print("❌ ADVERTENCIA: GROQ_API_KEY no configurada. Las consultas al LLM fallarán.")
else:
    print("✅ GROQ_API_KEY configurada")

print("✅ Librerías cargadas")

# %% [markdown]
# ## 1. Ingesta de PDFs y Chunking

# %%
print(f"📂 Buscando PDFs en {PDF_DIR}...")
pdf_files = list(PDF_DIR.glob("*.pdf"))

if not pdf_files:
    print(f"⚠️ No se encontraron PDFs. Asegúrate de tener archivos en {PDF_DIR}.")
else:
    for f in pdf_files:
        print(f"   • {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    
    print("\n🔄 Procesando y dividiendo en chunks (512 tokens)...")
    chunks = ingest_pdfs()
    
    # Mostrar ejemplos de chunks
    print("\n📄 Ejemplo de chunk (Chunk 0):")
    print(f"Fuente: {chunks[0].metadata['source']}")
    print("-" * 50)
    print(chunks[0].page_content[:200] + "...")
    print("-" * 50)

# %% [markdown]
# ## 2. Embeddings y Vector Store (ChromaDB)

# %%
print("🔄 Inicializando Vector Store...")

# Generar embeddings y crear base de datos (toma tiempo la primera vez)
try:
    if len(list(CHROMA_DIR.glob("*.sqlite3"))) > 0:
        print("📁 Se encontró una base de datos ChromaDB existente.")
        # Opcional: vectorstore = load_vectorstore()
        # Aquí la recreamos para asegurar que coincida con los chunks actuales
        vectorstore = create_vectorstore(chunks)
    else:
        print("📁 Creando nueva base de datos ChromaDB...")
        vectorstore = create_vectorstore(chunks)
        
except Exception as e:
    print(f"❌ Error al crear Vector Store: {e}")
    # Fallback si no hay chunks (para pruebas sin PDFs)
    if not pdf_files:
        print("   Ejecutando en modo mock por falta de PDFs.")

# %% [markdown]
# ## 3. Configuración del LLM y Chain

# %%
print("🔄 Inicializando LLM (Groq Llama 3)...")
try:
    llm = get_llm(model="llama-3.3-70b-versatile", temperature=0.1)
    
    # Construir la cadena RAG (top-k = 4)
    chain, retriever = build_rag_chain(vectorstore, llm, k=4)
    print("✅ Cadena RAG construida exitosamente")
    
except Exception as e:
    print(f"❌ Error al inicializar LLM: {e}")

# %% [markdown]
# ## 4. Pruebas de Consulta Interactivas

# %%
# Preguntas de prueba relevantes a COPD / Neumonía
preguntas_prueba = [
    "¿Cuáles son los criterios diagnósticos para la severidad COPD GOLD?",
    "¿Qué recomienda la guía de práctica clínica para el tratamiento ambulatorio de neumonía?",
    "¿Cuáles son los principales factores de riesgo ambientales y de estilo de vida para COPD?",
]

if 'chain' in locals():
    print("🩺 Probando el Asistente Médico:")
    for i, pregunta in enumerate(preguntas_prueba, 1):
        print(f"\n{'-'*60}")
        print(f"❓ Pregunta {i}: {pregunta}")
        print(f"{'-'*60}")
        
        try:
            resultado = query_rag(chain, retriever, pregunta)
            
            print("🤖 Respuesta:\n")
            print(resultado["answer"])
            
            print("\n📚 Fuentes Consultadas:")
            fuentes_vistas = set()
            for doc in resultado["source_documents"]:
                fuente = doc.metadata.get("source", "Desconocida")
                if fuente not in fuentes_vistas:
                    print(f"   • {fuente}")
                    fuentes_vistas.add(fuente)
                    
        except Exception as e:
            print(f"❌ Error en consulta: {e}")

# %% [markdown]
# ## 5. Evaluación Cuantitativa con RAGAS
# 
# (Nota: Ragas requiere llamadas intensivas al LLM y puede consumir rate limits.
# Si tienes una clave gratuita, este bloque puede fallar por exceso de requests.
# Se recomienda usar un subconjunto pequeño para pruebas).

# %%
# Definir dataset de evaluación
eval_questions = [
    {
        "question": "¿Cuáles son los grados de obstrucción del flujo aéreo en la clasificación GOLD de COPD?",
        "ground_truth": "En la clasificación GOLD, los grados de obstrucción del flujo aéreo basados en FEV1 post-broncodilatador son: GOLD 1 (Leve) FEV1 >= 80%, GOLD 2 (Moderada) 50% <= FEV1 < 80%, GOLD 3 (Grave) 30% <= FEV1 < 50%, y GOLD 4 (Muy Grave) FEV1 < 30%."
    },
    {
        "question": "¿Qué recomendaciones da la guía sobre el tratamiento de la neumonía adquirida en la comunidad?",
        "ground_truth": "El tratamiento inicial de la neumonía adquirida en la comunidad suele ser empírico, basado en antibióticos. Se recomienda iniciar la terapia lo antes posible, idealmente dentro de las primeras horas tras el diagnóstico, ajustando posteriormente según los resultados de los cultivos o pruebas microbiológicas si están disponibles."
    }
]

if 'chain' in locals() and len(pdf_files) > 0:
    print("\n📊 Iniciando evaluación RAGAS...")
    print("   (Puede tardar 1-2 minutos y consumir requests a la API)")
    
    try:
        # Ejecutar evaluación
        resultados_ragas = evaluate_rag(chain, retriever, eval_questions)
        
    except Exception as e:
        print(f"\n⚠️ La evaluación RAGAS falló. Esto es común con APIs gratuitas debido a Rate Limits.")
        print(f"Detalle del error: {e}")
        
        print("\nPara mitigar rate limits en evaluación:")
        print("1. Usar modelos más pequeños para el evaluator de Ragas")
        print("2. Aumentar el time.sleep() entre llamadas")
        print("3. Usar un proveedor LLM con mayor límite para evaluación (ej. OpenAI)")

# %%
print("\n" + "=" * 60)
print("  ✅ MÓDULO RAG COMPLETADO")
print("  → Ya puedes ejecutar la aplicación con: streamlit run app/main.py")
print("=" * 60)
