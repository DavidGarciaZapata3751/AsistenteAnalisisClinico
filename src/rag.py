"""
src/rag.py
==========
Módulo RAG (Retrieval-Augmented Generation) para consulta de guías médicas.
Integra PyMuPDF para ingesta, BAAI/bge-m3 para embeddings,
ChromaDB como vector store, y Groq (Llama 3) para generación.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# Cargar variables de entorno
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "PDF"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# ── Ingesta de PDFs ─────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """
    Extrae texto de un PDF usando PyMuPDF (fitz).

    Parameters
    ----------
    pdf_path : str | Path
        Ruta al archivo PDF.

    Returns
    -------
    str
        Texto completo extraído del PDF.
    """
    doc = fitz.open(str(pdf_path))
    text = ""
    for page_num, page in enumerate(doc):
        page_text = page.get_text("text")
        text += f"\n--- Página {page_num + 1} ---\n{page_text}"
    doc.close()
    print(f"  Document: {Path(pdf_path).name}: {len(text):,} caracteres extraidos")
    return text


def ingest_pdfs(pdf_dir: str | Path = None) -> list[Document]:
    """
    Lee todos los PDFs de un directorio y los divide en chunks.

    Chunking: 512 tokens / 64 overlap (usando RecursiveCharacterTextSplitter).

    Parameters
    ----------
    pdf_dir : str | Path
        Directorio con PDFs. Default: PDF_DIR.

    Returns
    -------
    list[Document]
        Lista de documentos LangChain chunkeados.
    """
    if pdf_dir is None:
        pdf_dir = PDF_DIR

    pdf_dir = Path(pdf_dir)
    pdf_files = list(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No se encontraron PDFs en {pdf_dir}")

    print(f"Procesando {len(pdf_files)} PDFs desde {pdf_dir}:")

    # Extraer texto de cada PDF
    documents = []
    for pdf_path in pdf_files:
        text = extract_text_from_pdf(pdf_path)
        documents.append(
            Document(
                page_content=text,
                metadata={"source": pdf_path.name, "type": "guia_medica"},
            )
        )

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Total de chunks generados: {len(chunks)}")

    return chunks


# ── Embeddings & Vector Store ───────────────────────────────────────────────

def get_embeddings(model_name: str = "BAAI/bge-m3"):
    """
    Inicializa el modelo de embeddings de HuggingFace.

    Parameters
    ----------
    model_name : str
        Modelo de embeddings (default: BAAI/bge-m3).

    Returns
    -------
    HuggingFaceEmbeddings
    """
    print(f"Cargando modelo de embeddings: {model_name}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )
    print(f"Modelo de embeddings cargado")
    return embeddings


def create_vectorstore(chunks: list[Document],
                       embeddings=None,
                       persist_dir: str | Path = None) -> Chroma:
    """
    Crea (o recarga) un vector store en ChromaDB.

    Parameters
    ----------
    chunks : list[Document]
        Chunks de documentos a indexar.
    embeddings : HuggingFaceEmbeddings
    persist_dir : str | Path
        Directorio de persistencia.

    Returns
    -------
    Chroma
        Vector store listo para búsqueda.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    if persist_dir is None:
        persist_dir = str(CHROMA_DIR)

    print(f"Creando vector store con {len(chunks)} chunks...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir),
        collection_name="guias_medicas",
    )
    print(f"Vector store creado en {persist_dir}")
    return vectorstore


def load_vectorstore(embeddings=None,
                     persist_dir: str | Path = None) -> Chroma:
    """
    Carga un vector store existente desde disco.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    if persist_dir is None:
        persist_dir = str(CHROMA_DIR)

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
        collection_name="guias_medicas",
    )
    print(f"Vector store cargado desde {persist_dir}")
    return vectorstore


# ── LLM & RAG Chain ────────────────────────────────────────────────────────

def get_llm(model: str = "llama-3.3-70b-versatile",
            temperature: float = 0.1) -> ChatGroq:
    """
    Inicializa el LLM de Groq (Llama 3).

    Parameters
    ----------
    model : str
        Modelo de Groq a utilizar.
    temperature : float

    Returns
    -------
    ChatGroq
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY no encontrada en .env")

    llm = ChatGroq(
        model=model,
        temperature=temperature,
        api_key=api_key,
    )
    print(f"LLM inicializado: {model} (Groq)")
    return llm


# Prompt template para el RAG
RAG_PROMPT_TEMPLATE = """Eres un asistente médico especializado en enfermedades respiratorias (COPD, neumonía).
Responde la pregunta del usuario basándote ÚNICAMENTE en el contexto proporcionado.
Si la información no está en el contexto, indica claramente que no tienes suficiente información.
Responde siempre en español.

CONTEXTO:
{context}

PREGUNTA:
{question}

RESPUESTA:"""


def build_rag_chain(vectorstore: Chroma, llm: ChatGroq = None, k: int = 4):
    """
    Construye la cadena RAG completa.

    Pipeline: Query → Retriever (top-k cosine) → Prompt → LLM → Output

    Parameters
    ----------
    vectorstore : Chroma
    llm : ChatGroq
    k : int
        Número de documentos a recuperar (top-k).

    Returns
    -------
    tuple
        (chain, retriever) - La cadena RAG y el retriever.
    """
    if llm is None:
        llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join(
            f"[Fuente: {doc.metadata.get('source', 'N/A')}]\n{doc.page_content}"
            for doc in docs
        )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def query_rag(chain, retriever, question: str) -> dict:
    """
    Ejecuta una consulta contra el sistema RAG.

    Parameters
    ----------
    chain : Runnable
        Cadena RAG.
    retriever : Retriever
        Retriever para obtener contextos.
    question : str
        Pregunta del usuario.

    Returns
    -------
    dict
        {"question", "answer", "contexts", "source_documents"}
    """
    # Obtener documentos relevantes
    source_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in source_docs]

    # Generar respuesta
    answer = chain.invoke(question)

    return {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "source_documents": source_docs,
    }


# ── Evaluación con RAGAS ───────────────────────────────────────────────────

def evaluate_rag(chain, retriever, eval_questions: list[dict],
                 llm=None) -> dict:
    """
    Evalúa el sistema RAG con RAGAS (faithfulness, answer_relevancy).

    Parameters
    ----------
    chain : Runnable
    retriever : Retriever
    eval_questions : list[dict]
        Lista con {"question": str, "ground_truth": str}.
    llm : ChatGroq, optional
        LLM para RAGAS evaluator.

    Returns
    -------
    dict
        Resultados de evaluación RAGAS.
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    print("Generando respuestas para evaluacion...")
    for item in eval_questions:
        result = query_rag(chain, retriever, item["question"])
        questions.append(item["question"])
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        ground_truths.append(item["ground_truth"])

    # Crear dataset para RAGAS
    eval_dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("Ejecutando evaluacion RAGAS...")
    results = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, answer_relevancy],
    )

    print(f"\n{'='*50}")
    print(f"  Resultados RAGAS")
    print(f"{'='*50}")
    print(f"  Faithfulness:      {results['faithfulness']:.4f}")
    print(f"  Answer Relevancy:  {results['answer_relevancy']:.4f}")

    return results
