from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_and_split_pdfs(pdf_dir: Path, chunk_size: int, chunk_overlap: int) -> List[Document]:
    pdf_path = Path(pdf_dir)
    
    # Recherche récursive des .pdf
    pdf_files = list(pdf_path.rglob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"Aucun PDF trouvé dans {pdf_path.resolve()} (ni dans ses sous-dossiers)")
        
    print(f"📖 {len(pdf_files)} fichier(s) PDF trouvé(s). Chargement en cours...")
    
    # PyPDFDirectoryLoader de langchain_community supporte le globbing récursif
    loader = PyPDFDirectoryLoader(str(pdf_path), glob="**/*.pdf")
    documents = loader.load()
    print(f"✅ {len(documents)} page(s) chargée(s) au total.")

    print(f"✂️ Découpage des textes (Size: {chunk_size}, Overlap: {chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✅ {len(chunks)} chunks créés.")
    return chunks