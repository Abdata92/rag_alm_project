from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

def create_and_persist_vectorstore(
    chunks: List[Document], 
    embeddings, 
    persist_dir: Path
) -> FAISS:
    print(f"💾 Vectorisation et sauvegarde FAISS dans '{persist_dir.resolve()}'...")
    
    # Création du vectorstore en mémoire
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    
    # Sauvegarde sur disque dans le dossier vector_db
    persist_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(persist_dir))
    
    print("🎉 Base vectorielle FAISS créée et sauvegardée avec succès !")
    return vector_store