import streamlit as st
import config
from src.embeddings import get_embedding_model
from langchain_community.vectorstores import FAISS
from src.llm import load_local_llm
from src.rag_chain import build_rag_chain

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistant RAG ALM - Documentation Financière",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Assistant RAG ALM - Analyse des DIC")
st.caption("Posez vos questions financières sur les Documents d'Informations Clés (DIC).")

# Charger et mettre en cache le pipeline RAG pour éviter de le recharger à chaque interaction
@st.cache_resource
def init_rag_pipeline():
    embeddings = get_embedding_model(config.EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        str(config.VECTOR_DB_DIR), 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    llm = load_local_llm(getattr(config, 'OLLAMA_MODEL_NAME', 'qwen2:1.5b'))
    rag_chain, retriever = build_rag_chain(vector_store, llm)
    return rag_chain, retriever

with st.spinner("⏳ Chargement des modèles et de la base vectorielle..."):
    rag_chain, retriever = init_rag_pipeline()

# Historique de la session Streamlit
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Barre latérale pour effacer l'historique
with st.sidebar:
    st.header("⚙️ Options")
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

# Affichage des anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources consultées"):
                for src in message["sources"]:
                    st.write(f"- `{src}`")

# Saisie utilisateur
if prompt := st.chat_input("Ex: Quel est le niveau de risque de ce fonds ?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération de la réponse..."):
            # 1. Récupération des documents sources
            source_docs = retriever.invoke(prompt)
            sources = list(set([
                doc.metadata.get('source', doc.metadata.get('file_path', 'Fichier inconnu'))
                for doc in source_docs
            ]))

            # 2. Inférence RAG
            answer = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })

            # 3. Affichage
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources consultées"):
                    for src in sources:
                        st.write(f"- `{src}`")

    # Mise à jour de l'historique
    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "sources": sources
    })
    st.session_state.chat_history.extend([
        ("human", prompt),
        ("ai", answer)
    ])