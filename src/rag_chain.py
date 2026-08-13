from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    """Formate les chunks récupérés pour les injecter dans le prompt."""
    return "\n\n".join(f"[Source: {doc.metadata.get('source', 'Inconnu')}]\n{doc.page_content}" for doc in docs)

def build_rag_chain(vector_store: FAISS, llm):
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Prompt métier ALM
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "Tu es un assistant expert en finance et gestion Actif/Passif (ALM).\n"
         "Réponds à la question posée de manière précise en t'appuyant uniquement sur le contexte ci-dessous.\n"
         "Si l'information n'est pas contenue dans le contexte, réponds simplement que tu ne sais pas.\n\n"
         "Contexte :\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    # Chaîne LCEL moderne sans dépendance vers `langchain.chains`
    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["input"])),
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", [])
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever