import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as hf_logging # <-- AGGIUNGI QUESTA

# Zittiamo i warning di Hugging Face!
hf_logging.set_verbosity_error() # <-- AGGIUNGI QUESTA
from datasets import load_dataset
from tqdm import tqdm # Per la barra di caricamento

def evaluate_on_datasets():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- INIZIALIZZAZIONE VALUTAZIONE UFFICIALE ---")

    # Modelli e Tokenizer
    base_name = "roberta-base"
    model_a_name = "SamLowe/roberta-base-go_emotions"
    model_b_name = "cross-encoder/nli-roberta-base"

    tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
    
    model_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2).to(device)
    model_a = AutoModelForSequenceClassification.from_pretrained(model_a_name).to(device)
    model_b = AutoModelForSequenceClassification.from_pretrained(model_b_name).to(device)

    # Merging con la tua "Golden Ratio" (0.6 / 0.4)
    print("\nFase 1: Creazione Modello Ibrido (L_Emo = 0.6, L_Logica = 0.4)...")
    dict_base = model_base.state_dict()
    dict_a = model_a.state_dict()
    dict_b = model_b.state_dict()
    dict_merged = {}

    la, lb = 0.6, 0.4
    for key in dict_base.keys():
        if key.startswith("roberta."):
            task_vector_a = dict_a[key] - dict_base[key]
            task_vector_b = dict_b[key] - dict_base[key]
            dict_merged[key] = dict_base[key] + (la * task_vector_a) + (lb * task_vector_b)

    # ==========================================
    # VALUTAZIONE NLI (LOGICA)
    # ==========================================
    print("\nFase 2: Valutazione su SNLI Dataset (500 samples)...")
    model_b.load_state_dict(dict_merged, strict=False)
    model_b.eval()

    # Scarichiamo 500 frasi di test
    # Il dataset SNLI ha etichette: 0 (Entailment), 1 (Neutral), 2 (Contradiction)
    snli = load_dataset("snli", split="test[:500]")
    
    corrette_nli = 0
    totali_nli = 0

    for item in tqdm(snli, desc="Test NLI"):
        # Ignoriamo i campioni senza etichetta (-1)
        if item['label'] == -1: 
            continue
            
        inputs = tokenizer_b(item['premise'], item['hypothesis'], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            outputs = model_b(**inputs)
            
        pred_idx = torch.argmax(outputs.logits, dim=-1).item()
        
        # Mappatura etichette (le label originali SNLI e quelle del modello HuggingFace sono spesso invertite)
        # Il modello cross-encoder usa: 0=Contradiction, 1=Entailment, 2=Neutral
        # SNLI originale usa: 0=Entailment, 1=Neutral, 2=Contradiction
        # Facciamo la conversione:
        mapped_pred = -1
        if pred_idx == 0: mapped_pred = 2 # Contradiction
        elif pred_idx == 1: mapped_pred = 0 # Entailment
        elif pred_idx == 2: mapped_pred = 1 # Neutral

        if mapped_pred == item['label']:
            corrette_nli += 1
        totali_nli += 1

    acc_nli = (corrette_nli / totali_nli) * 100

    # ==========================================
    # VALUTAZIONE EMOZIONI (Opzionale per stasera, più complessa)
    # ==========================================
    # GoEmotions è un dataset "Multi-label" (più emozioni vere per frase), 
    # valutarlo matematicamente in modo rigoroso è più rognoso.
    # Per il momento stampiamo l'Accuracy del task logico che è il vero scoglio.

    print("\n==================================================")
    print(f"RISULTATI FINALI SUL DATASET")
    print("==================================================")
    print(f"Modello: RoBERTa-Ibrido (Emo: 0.6 | Log: 0.4)")
    print(f"Accuratezza Logica (SNLI): {acc_nli:.2f}% ({corrette_nli}/{totali_nli} frasi corrette)")
    print("==================================================")

if __name__ == "__main__":
    evaluate_on_datasets()