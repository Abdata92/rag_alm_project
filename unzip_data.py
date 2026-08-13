import zipfile
from pathlib import Path
from typing import Union

def extract_zip_file(
    zip_name: str, 
    target_folder: str, 
    base_dir: Union[str, Path] = "./data"
) -> None:
    """
    Extrait un fichier ZIP précis situé dans base_dir vers un dossier cible.
    
    Parameters:
    -----------
    zip_name : str
        Nom du fichier zip (avec ou sans .zip, ex: "DIC" ou "DIC.zip").
    target_folder : str
        Nom du dossier de destination (ex: "raw_pdfs").
    base_dir : str ou Path (défaut: "./data")
        Dossier racine contenant le fichier ZIP.
    """
    base_path = Path(base_dir)
    
    # S'assurer que le nom du zip a bien l'extension .zip
    zip_filename = zip_name if zip_name.endswith(".zip") else f"{zip_name}.zip"
    
    zip_path = base_path / zip_filename
    target_path = base_path / target_folder

    # 1. Vérifier que le fichier ZIP existe
    if not zip_path.exists():
        print(f"❌ Le fichier '{zip_path}' est introuvable.")
        return

    # 2. Créer le dossier cible s'il n'existe pas
    target_path.mkdir(parents=True, exist_ok=True)

    # 3. Extraction
    print(f"Extraction de '{zip_path.name}' vers '{target_path}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    print(f"✅ Extraction réussie dans : {target_path}")


if __name__ == "__main__":
    print("🚀 Début de l'extraction des données...")

    # Exemple 1 : Extrait data/DIC.zip vers data/raw_pdfs/

    extract_zip_file(zip_name="dataset_eval", target_folder="evaluation", base_dir="./data")