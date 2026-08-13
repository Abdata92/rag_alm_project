from langchain_ollama import OllamaLLM

def load_local_llm(model_name: str = "qwen2:1.5b") -> OllamaLLM:
    print(f"🤖 Connexion au LLM Ollama local ({model_name})...")
    
    llm = OllamaLLM(
        model=model_name,
        temperature=0.1,  # Précision financière
        num_ctx=2046,  # Réduit l'empreinte mémoire RAM de la fenêtre de contexte
        # num_gpu=0  # <-- Force l'utilisation du CPU pour éviter les erreurs Out-Of-Memory VRAM
    )
    return llm