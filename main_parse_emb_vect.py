import config
from src.parser import load_and_split_pdfs
from src.embeddings import get_embedding_model
from src.vectorstore import create_and_persist_vectorstore

def main():
    print("🚀 Démarrage de l'Étape 1 : Preparation des données et Vectorisation")
    
    # 1. Parsing & Chunking
    chunks = load_and_split_pdfs(
        pdf_dir=config.PDF_DIR,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )
    
    # 2. Embedding Model
    embeddings = get_embedding_model(config.EMBEDDING_MODEL_NAME)
    
    # 3. Vector DB Creation
    create_and_persist_vectorstore(
        chunks=chunks,
        embeddings=embeddings,
        persist_dir=config.VECTOR_DB_DIR
    )

if __name__ == "__main__":
    main()