from pathlib import Path

# Chemins
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
EVAL_DIR = DATA_DIR / "evaluation"
VECTOR_DB_DIR = BASE_DIR / "vector_db"

# Paramètres de Chunking
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

# Modèle d'Embeddings
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# Modèle LLM (Open Weights)
# Vous pouvez utiliser un modèle Hugging Face ou un serveur local (Ollama / vLLM)
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
OLLAMA_MODEL_NAME = "qwen2:1.5b" #"phi3" #"mistral"  # ou "llama3"

TOP_K_RETRIEVAL = 4  # Nombre de chunks à récupérer pour répondre