"""
Script para ingestar los PDFs en ChromaDB.
Extrae texto, genera chunks y crea el vector store.
"""
import sys
sys.path.insert(0, ".")
from src.rag import ingest_pdfs, get_embeddings, create_vectorstore

print("=" * 60)
print("  INGESTA DE PDFs EN CHROMADB")
print("=" * 60)

# 1. Extraer texto y crear chunks
print("\n[1/3] Extrayendo texto de los PDFs y creando chunks...")
chunks = ingest_pdfs()

# 2. Cargar modelo de embeddings
print("\n[2/3] Cargando modelo de embeddings...")
embeddings = get_embeddings()

# 3. Crear vector store
print("\n[3/3] Creando vector store en ChromaDB...")
vectorstore = create_vectorstore(chunks, embeddings)

# Verificar
col = vectorstore._collection
print(f"\nVerificacion: {col.count()} documentos indexados en ChromaDB")
print("=" * 60)
print("INGESTA COMPLETADA EXITOSAMENTE")
print("Reinicia la app de Streamlit para usar las nuevas fuentes.")
