import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging

# Zittiamo i warning di Hugging Face!
hf_logging.set_verbosity_error()
from datasets import load_dataset
from tqdm import tqdm

def evaluate_layer_wise():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- ESPERIMENTO 2: LAYER-WISE MERGING ---")
    print("Fonderemo solo gli Embeddings e i Layer da 0 a 5.")

    base_name = "roberta-base"
    model_a_name = "SamLowe/roberta-base-go_emotions"
    model_b_name = "cross-encoder/nli-roberta-base"

    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
    
    # Caricamento Modelli
    model_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2).to(device)
    model_a = AutoModelForSequenceClassification.from_pretrained(model_a_name).to(device)
    model_b = AutoModelForSequenceClassification.from_pretrained(model_b_name).to(device)

    dict_base = model_base.state_dict()
    dict_a = model_a.state_dict()
    dict_b = model_b.state_dict()
    
    # Il nostro dizionario ibrido parziale
    dict_merged = {}
    la, lb = 0.6, 0.4
    
    for key in dict_base.keys():
        if key.startswith("roberta."):
            is_early_layer = False
            
            # Fonde la base del vocabolario (Embeddings)
            if "embeddings" in key:
                is_early_layer = True
            # Fonde SOLO i primi 6 layer (da 0 a 5 inclusi)
            elif "encoder.layer." in key:
                layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
                if layer_num <= 5:
                    is_early_layer = True
            
            # Se è un layer "basso", applichiamo la Task Arithmetic
            if is_early_layer:
                task_vector_a = dict_a[key] - dict_base[key]
                task_vector_b = dict_b[key] - dict_base[key]
                dict_merged[key] = dict_base[key] + (la * task_vector_a) + (lb * task_vector_b)
            # NOTA: Per i layer da 6 a 11 NON mettiamo nulla nel dizionario dict_merged.

    print("\nFase 1: Impianto parziale completato.")
    # strict=False ci salva la vita qui!
    # PyTorch aggiornerà i layer 0-5 con quelli fusi, ma lascerà i layer 6-11 
    # esattamente come erano nel modello NLI originale, perché non li trova nel dict_merged!
    model_b.load_state_dict(dict_merged, strict=False)
    model_b.eval()

    # ==========================================
    # VALUTAZIONE NLI
    # ==========================================
    print("Fase 2: Valutazione su SNLI Dataset (500 samples)...")
    snli = load_dataset("snli", split="test[:500]")
    
    corrette_nli = 0
    totali_nli = 0

    for item in tqdm(snli, desc="Test NLI"):
        if item['label'] == -1: 
            continue
        
        inputs = tokenizer_b(item['premise'], item['hypothesis'], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = model_b(**inputs)
            
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        
        # Mappatura etichette
        mapped_pred = -1
        if pred_idx == 0: mapped_pred = 2
        elif pred_idx == 1: mapped_pred = 0
        elif pred_idx == 2: mapped_pred = 1

        if mapped_pred == item['label']:
            corrette_nli += 1
        totali_nli += 1

    acc_nli = (corrette_nli / totali_nli) * 100

    print("\n==================================================")
    print(f"RISULTATI: LAYER-WISE MERGING")
    print("==================================================")
    print(f"Modello: RoBERTa Ibrido (Solo L0-L5 fusi | L6-L11 Puri NLI)")
    print(f"Accuratezza Logica (SNLI): {acc_nli:.2f}% ({corrette_nli}/{totali_nli})")
    print("==================================================")

if __name__ == "__main__":
    evaluate_layer_wise()