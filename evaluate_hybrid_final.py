import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

hf_logging.set_verbosity_error()

def evaluate_hybrid():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- VALUTAZIONE FINALE: LOGICA vs EMOZIONI ---")

    # Configurazione modelli
    model_a_name = "SamLowe/roberta-base-go_emotions" # Task A
    model_b_name = "cross-encoder/nli-roberta-base"   # Task B
    base_name = "roberta-base"

    tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
    
    model_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2).to(device)
    model_a = AutoModelForSequenceClassification.from_pretrained(model_a_name).to(device)
    model_b = AutoModelForSequenceClassification.from_pretrained(model_b_name).to(device)

    # 1. LAYER-WISE MERGING (L0-L5)
    print("\nFase 1: Merging Layer-wise (L0-L5)...")
    dict_base, dict_a, dict_b = model_base.state_dict(), model_a.state_dict(), model_b.state_dict()
    dict_merged = {}
    la, lb = 0.6, 0.4

    for key in dict_base.keys():
        if key.startswith("roberta."):
            is_early_layer = False
            # Fonde Embeddings
            if "embeddings" in key:
                is_early_layer = True
            # Fonde i layer 0-5
            elif "encoder.layer." in key:
                layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
                if layer_num <= 5:
                    is_early_layer = True
            
            if is_early_layer:
                dict_merged[key] = dict_base[key] + (la * (dict_a[key] - dict_base[key])) + (lb * (dict_b[key] - dict_base[key]))

    # 2. VALUTAZIONE LOGICA (NLI)
    print("\nFase 2: Valutazione Logica (SNLI)...")
    model_b.load_state_dict(dict_merged, strict=False)
    model_b.eval()
    snli = load_dataset("snli", split="test[:400]") # Ridotto a 400 per velocità
    
    corrette_nli, totali_nli = 0, 0
    for item in tqdm(snli, desc="SNLI"):
        if item['label'] == -1: continue
        inputs = tokenizer_b(item['premise'], item['hypothesis'], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            pred = torch.argmax(model_b(**inputs).logits, dim=-1).item()
        
        # Mapping etichette: 0=Contr, 1=Entail, 2=Neut
        mapped = {0: 2, 1: 0, 2: 1}[pred]
        if mapped == item['label']: corrette_nli += 1
        totali_nli += 1

    # 3. VALUTAZIONE EMOZIONI (GoEmotions)
    print("\nFase 3: Valutazione Emozioni (GoEmotions)...")
    model_a.load_state_dict(dict_merged, strict=False)
    model_a.eval()
    emotions_ds = load_dataset("go_emotions", "simplified", split="test[:400]")
    
    all_labels, all_preds = [], []
    for item in tqdm(emotions_ds, desc="GoEmotions"):
        inputs = tokenizer_a(item['text'], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            logits = model_a(**inputs).logits
            # Per il multi-label usiamo una soglia (0.5) dopo la sigmoide
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs > 0.3).astype(int) # Soglia 0.3 per essere più sensibili
        
        # Prepariamo le etichette reali (formato multi-hot)
        true_labels = np.zeros(28)
        for l in item['labels']: true_labels[l] = 1
        
        all_labels.append(true_labels)
        all_preds.append(preds)

    f1_emo = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "="*50)
    print("📊 REPORT FINALE MODELLO IBRIDO")
    print("="*50)
    print(f"1. LOGICA (NLI Accuracy):  { (corrette_nli/totali_nli)*100 :.2f}%")
    print(f"2. EMOZIONI (F1-Score):     { f1_emo*100 :.2f}")
    print("="*50)
    print("Interpretazione: F1-Score > 40 è ottimo per un merge a 28 classi.")

if __name__ == "__main__":
    evaluate_hybrid()