"""Debug script to check ChromaDB and retriever."""
import sys
sys.path.insert(0, ".")
from src.rag import load_vectorstore, get_embeddings, get_llm, build_rag_chain, query_rag

print("=" * 60)
print("1. Loading embeddings...")
emb = get_embeddings()

print("\n2. Loading vectorstore...")
vs = load_vectorstore(emb)

print("\n3. Checking collection...")
col = vs._collection
count = col.count()
print(f"   Total documents in ChromaDB: {count}")

if count > 0:
    results = col.peek(3)
    print(f"   Sample metadatas: {results['metadatas']}")
    print(f"   Sample content lengths: {[len(d) for d in results['documents']]}")
else:
    print("   WARNING: ChromaDB is EMPTY! Need to run ingestion first.")

print("\n4. Testing retriever...")
retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 4})
docs = retriever.invoke("que es la neumonia")
print(f"   Retriever returned {len(docs)} documents")
for i, doc in enumerate(docs):
    print(f"   Doc {i}: source={doc.metadata.get('source', 'N/A')}, len={len(doc.page_content)}")
    print(f"   Preview: {doc.page_content[:100]}...")

print("\n5. Testing full query_rag...")
try:
    llm = get_llm()
    chain, ret = build_rag_chain(vs, llm)
    result = query_rag(chain, ret, "que es la neumonia")
    print(f"   Answer length: {len(result['answer'])}")
    print(f"   Source docs count: {len(result['source_documents'])}")
    for i, doc in enumerate(result['source_documents']):
        print(f"   Source {i}: {doc.metadata}")
except Exception as e:
    print(f"   ERROR in query_rag: {e}")

print("\n" + "=" * 60)
print("DONE")
