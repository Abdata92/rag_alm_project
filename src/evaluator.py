import json
import gc
import os
from pathlib import Path
import bert_score
from tqdm import tqdm

def run_evaluation(rag_chain, eval_dir: Path, output_file: Path):
    queries_file = eval_dir / "queries.json"
    answers_file = eval_dir / "answers.json"
    predictions_file = eval_dir / "generated_predictions.json"
    
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(answers_file, "r", encoding="utf-8") as f:
        reference_answers = json.load(f)

    # 1. Récupération ou Génération des réponses
    if predictions_file.exists():
        print(f"📦 Chargement des prédictions existantes depuis : {predictions_file.resolve()}")
        with open(predictions_file, "r", encoding="utf-8") as f:
            generated_data = json.load(f)
    else:
        print(f"🚀 Inférence RAG sur {len(queries)} requêtes d'évaluation...\n")
        generated_data = {}
        pbar = tqdm(queries.items(), total=len(queries), desc="Inférence RAG", unit="req")
        errors_count = 0

        for uuid, query in pbar:
            if uuid in reference_answers:
                try:
                    answer = rag_chain.invoke({"input": query, "chat_history": []})
                    if not answer or not str(answer).strip():
                        answer = "Information non disponible."
                except Exception as e:
                    tqdm.write(f"⚠️ Erreur [UUID: {uuid}] : {e}")
                    answer = "Erreur d'inférence RAG."
                    errors_count += 1

                generated_data[uuid] = str(answer)
                pbar.set_postfix({"Erreurs": errors_count})

        # Sauvegarde intermédiaire des réponses générées
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, ensure_ascii=False, indent=4)
        print(f"💾 Prédictions sauvegardées dans : {predictions_file.resolve()}")

    # 🧹 Nettoyage de la mémoire avant de charger BERTScore
    del rag_chain
    gc.collect()

    # 2. Préparation des listes pour BERTScore
    uuids = []
    cands = []
    refs = []

    for uuid, gen_ans in generated_data.items():
        if uuid in reference_answers:
            uuids.append(uuid)
            cands.append(gen_ans)
            refs.append(reference_answers[uuid])

    print("\n⚡ Calcul du F1 BERTScore (modèle multilingue)...")
    
    # Batch size à 8 et nthreads à 2 pour éviter les plantages de mémoire Rust
    P, R, F1 = bert_score.score(
        cands=cands, 
        refs=refs, 
        lang="fr", 
        batch_size=8,
        nthreads=2,
        verbose=True
    )

    mean_f1 = F1.mean().item() * 100
    print("\n==========================================")
    print(f"🎯 F1 BERTScore Moyen : {mean_f1:.2f} %")
    print("==========================================")

    if mean_f1 >= 60.0:
        print("✅ Objectif atteint ! F1 BERTScore >= 60 %.")
    else:
        print("⚠️ Score sous les 60 %. Ajustement du chunking ou du prompt recommandé.")

    # Enregistrement final des résultats
    results = {
        "mean_f1_score": mean_f1,
        "total_queries": len(uuids),
        "details": [
            {
                "uuid": uid, 
                "query": queries.get(uid, ""), 
                "generated": gen, 
                "reference": ref, 
                "f1": f1_val.item()
            }
            for uid, gen, ref, f1_val in zip(uuids, cands, refs, F1)
        ]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"💾 Résultats globaux enregistrés dans : {output_file.resolve()}")