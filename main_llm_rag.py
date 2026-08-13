import config
from src.embeddings import get_embedding_model
from langchain_community.vectorstores import FAISS
from src.llm import load_local_llm
from src.rag_chain import build_rag_chain

def main():
    print("🚀 Chargement de l'agent RAG...")
    
    # 1. Recharger FAISS
    embeddings = get_embedding_model(config.EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        str(config.VECTOR_DB_DIR), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # 2. Charger le LLM via Ollama
    llm = load_local_llm(getattr(config, 'OLLAMA_MODEL_NAME', 'mistral'))
    
    # 3. Initialiser la chaîne RAG et le retriever
    rag_chain, retriever = build_rag_chain(vector_store, llm)
    
    chat_history = []
    
    print("\n✅ Agent prêt ! (Tapez 'exit' pour quitter)\n" + "="*50)
    
    while True:
        query = input("\n👤 Question ALM : ")
        if query.lower() in ["exit", "quit"]:
            break
            
        # Récupération des documents pour le Sourcing
        source_docs = retriever.invoke(query)
        
        # Exécution de la chaîne RAG
        answer = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        print(f"\n🤖 Réponse : {answer}")
        
        # Affichage des sources utilisées
        if source_docs:
            print("\n📚 Sources utilisées :")
            sources = set([
                doc.metadata.get('source', doc.metadata.get('file_path', 'Inconnu')) 
                for doc in source_docs
            ])
            for src in sources:
                print(f"  - {src}")
        
        # Mise à jour de l'historique conversationnel
        chat_history.extend([
            ("human", query),
            ("ai", answer)
        ])

if __name__ == "__main__":
    main()