# Assistant Conversationnel RAG sur la Documentation Financière (ALM)
## Résumé du Projet
Développement de A à Z d'un agent conversationnel intelligent basé sur une architecture RAG (Retrieval-Augmented Generation) pour un département ALM (Gestion Actif/Passif) dans le secteur de l'assurance-vie.
L'outil permet aux équipes financières d'interroger en langage naturel une masse de Documents d'Informations Clés (DIC) réglementaires afin d'accélérer la prise de décision et de synthétiser les caractéristiques et niveaux de risque des placements financiers.

## Problématique Métier (Business Value)
Les équipes ALM gèrent de très nombreux investissements chaque année et doivent analyser des documents harmonisés au niveau européen (les DIC) fournis par les gestionnaires de fonds. Face au volume et à la complexité des rapports financiers, la recherche manuelle d'informations est chronophage.

### Objectif : Mettre à disposition un assistant IA local, sécurisé et précis capable d'extraire instantanément les informations clés des DIC tout en citant systématiquement ses sources.

## Stack Technique & Architecture
* LLM Local (Open Weights) : qwen2:1.5b (via Ollama) garantissant la confidentialité des données financières internes.

* Embeddings : sentence-transformers/paraphrase-multilingual-mpnet-base-v2 adapté au traitement multilingue (français/anglais).

* Vector Store : FAISS (Facebook AI Similarity Search) pour l'indexation et la recherche vectorielle rapide.

* Orchestration RAG : LangChain (gestion du pipeline RAG, prompt engineering, gestion de la mémoire conversationnelle).

* Interface Utilisateur : Streamlit pour une application web interactive et ergonomique.

* Évaluation Métier : Calcul du F1 BERTScore sur un benchmark de 619 requêtes financières d'évaluation.

* Ingénierie & MLOps : * Gestionnaire de dépendances : Poetry

* Conteneurisation : Docker

* Pipeline CI/CD : GitHub Actions (tests unitaires automatisés et publication de l'image Docker sur GHCR).

## Fonctionnalités Clés
* Analyse & Chunking Intelligents des DIC : Découpage optimisé des documents PDF réglementaires pour préserver le contexte financier et le niveau de risque des produits.

* Recherche Sémantique & Sourcing : Récupération dynamique des passages pertinents avec affichage explicite des fichiers sources consultés pour chaque réponse.

* Mémoire Conversationnelle : Suivi du fil de la discussion pour permettre aux analystes de poser des questions de suivi (follow-up).

Interface Graphique Intuitive : Chatbot Streamlit permettant le suivi des échanges et le nettoyage de l'historique en un clic.

Validation Métier Amont/Aval : Benchmark d'évaluation automatisé garantissant un haut niveau de fidélité par rapport aux réponses de référence.

## Résultats & Performances (KPIs)
Précision & Pertinence : F1 BERTScore de 71,36 % atteint sur le dataset officiel de 619 requêtes (dépassant largement l'exigence minimale de 60 %).

Stabilité Technique : 100 % de succès d'inférence (0 erreur) lors du traitement par lots sur l'intégralité du benchmark.

Déploiement : Conteneurisation Docker complète et intégration CI/CD prêtes pour la mise en production.
