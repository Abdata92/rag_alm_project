import os

# FIX: Empêche l'explosion du nombre de threads OpenMP/PyTorch sur CPU
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["VECLIB_MAXIMUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import torch
torch.set_num_threads(2)

import config
from src.embeddings import get_embedding_model
from langchain_community.vectorstores import FAISS
from src.llm import load_local_llm
from src.rag_chain import build_rag_chain
from src.evaluator import run_evaluation

def main():
    print("🚀 Démarrage de l'Étape 3 : Évaluation du pipeline RAG")
    
    # 1. Charger FAISS & LLM Ollama
    embeddings = get_embedding_model(config.EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        str(config.VECTOR_DB_DIR), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    llm = load_local_llm(getattr(config, 'OLLAMA_MODEL_NAME', 'qwen2:1.5b'))
    
    # 2. Chaîne RAG
    rag_chain, _ = build_rag_chain(vector_store, llm)
    
    # 3. Lancer l'évaluation
    output_path = config.BASE_DIR / "evaluation_results.json"
    run_evaluation(rag_chain, config.EVAL_DIR, output_path)

if __name__ == "__main__":
    main()