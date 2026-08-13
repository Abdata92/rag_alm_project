# I. Create python environment

1- Create a python virtual environment with this command "py -m venv rag_venv"
2- Activate the environment ".\rag_venv\Scripts\activate"

## I.1. Install all dependencies with Poetry 
1- pip install poetry 
2- poetry init
3- poetry add pypdf pdfplumber langchain langchain-community sentence-transformers chromadb faiss-gpu bert-scorepip 

# II. Create first step architecture [ PARSING, EMBALDING AND VECTORIZE ]

rag_alm_project/
├── data/
│   ├── raw_pdfs/          # Tous vos fichiers DIC (.pdf)
│   └── evaluation/        # Les fichiers .json d'évaluation (corpus, queries, etc.)
├── vector_db/             # Dossier où ChromaDB ou FAISS sauvegardera l'index
├── src/
│   ├── parser.py          # Fonctions de lecture des PDF et chunking
│   ├── embeddings.py      # Chargement du modèle d'embedding local
│   └── vectorstore.py     # Création et sauvegarde de la base vectorielle
├── config.py              # Configurations (taille des chunks, overlap, chemins, etc.)
└── main_step1.py          # Script principal pour exécuter l'étape 1

## II.1. Extract data 
1- Create the python unzip function to unzip files dowloaded "unzip_data.py"
2- Execute this command "py .\unzip_data.py". Before execute command make sure that will choose the the correct input / output folders names. 
    - For example : Extract data/DIC.zip to  data/raw_pdfs/
    -  extract_zip_file(zip_name="DIC", target_folder="raw_pdfs", base_dir="./data")

## II.2 PARSING, EMBALDING AND VECTORIZE 
Go to the racin pipeline  and execute all commands 

1- to load parse.py you must install "poetry add langchain-pypdf"  
2- to load vectorize.py you must install "poetry add sentence-transformers langchain-huggingface"
3- install chromadb or FAISS from racin "poetry add chromadb langchain-chroma" , "poetry add faiss-cpu"

# III . Implement RAG pipeline with local LLM model ( Mistral or llama)

rag_alm_project/
├── src/
│   ├── ... (fichiers existants)
│   ├── llm.py         # Chargement du modèle Open Weight
│   └── rag_chain.py   # Assemblage de la chaîne RAG + Mémoire + Sourcing
└── main_step2.py      # Interface de test interactive

## Objectifs de cette étape :
- Integrate a Open Weights model (ex: Mistral-7B-Instruct ou Llama-3-8B-Instruct).
- Implémenter la recherche contextuelle : Le retriever cherche les chunks pertinents dans FAISS.
- Mettre en place le Sourcing : Indiquer à l'utilisateur d'où vient l'information (ex: nom du fichier DIC).
- Gérer la Mémoire Conversationnelle : Permettre au modèle de se souvenir des échanges précédents.

## Pré-requis
1 - installer Ollama localement "irm https://ollama.com/install.ps1 | iex" ou utiliser le lien "https://ollama.com/download/windows"
2- Ajouter langchain, langchain-community et langchain-ollama avec Poetry "poetry add langchain-ollama", "poetry add langchain langchain-community"
2- télécharger le modèle et ouvrir un chat direct dans le terminal "ollama run mistral"
3- Installer le package LangChain dédié avec "poetry add langchain-ollama"

**NB: Les versions récentes de LangChain ont progressivement déprécié et retiré le sous-module langchain.chains.La meilleure solution est d'utiliser LCEL (LangChain Expression Language) pur**


# IV. EVALUATE RAG (BERTScore) PIPELINE 
**'objectif final fixé par la direction ALM est de valider la pertinence des réponses en calculant le F1 BERTScore sur le dataset d'évaluation fourni (data/evaluation/) et d'obtenir un score supérieur à 60 %.**

1- Installer la bibliothèque d'évaluation "poetry add bert-score evaluate"
2- Installer le modèle "Mistral"  dans Ollama "ollama pull mistral"
- Créer le script d'évaluation
- Créer l'orchestrateur 

# V. MISE EN PRODUCTION 
## V.1. Création de l'application qui va gérer la pipeline 
## V.2. Installer l'environnement de production Streamlit 
    1- Lancer Streamlit "poetry add streamlit"
    2- lancer l'application web "poetry run streamlit run app.py"

# VI. Architecture du Pipeline CI/CD

[ Git Push / PR sur main ]
         │
         ▼
 1. CI : Integration Continu (GitHub Actions)
    ├── Installation de Python & Poetry
    ├── Linting & Formatting (Ruff / Flake8)
    └── Tests unitaires & de régression (Pytest)
         │
         ▼
 2. CD : Déploiement Continu
    ├── Build de l'Image Docker (Streamlit + Pipeline RAG)
    ├── Push vers GitHub Container Registry (GHCR) ou Docker Hub
    └── (Optionnel) Déploiement automatique sur serveur / Cloud Sandbox

## VI.1. Créer le Dockerfile pour Streamlit