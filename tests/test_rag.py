from pathlib import Path
import os

def test_project_structure():
    """Vérifie que les dossiers et fichiers indispensables sont présents."""
    base_dir = Path(__file__).parent.parent
    assert (base_dir / "app.py").exists(), "app.py est introuvable"
    assert (base_dir / "pyproject.toml").exists(), "pyproject.toml est introuvable"

def test_evaluation_file_exists():
    """Vérifie que les résultats de l'évaluation RAG sont bien enregistrés."""
    base_dir = Path(__file__).parent.parent
    eval_file = base_dir / "evaluation_results.json"
    assert eval_file.exists(), "Le fichier evaluation_results.json est introuvable"