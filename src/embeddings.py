import torch
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    # Détecte si CUDA est disponible, sinon bascule sur CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Chargement du modèle d'embeddings : {model_name} (Device: {device})...")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )