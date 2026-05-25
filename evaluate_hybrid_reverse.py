import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import f1_score

hf_logging.set_verbosity_error()

def evaluate_hybrid_reverse():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- ABLATION STUDY: REVERSE LAYER-WISE (L6-L11) ---")
    print("Ipotesi: La logica regge, ma le emozioni dovrebbero crollare.")

    # Configurazione modelli
    model_a_name = "SamLowe/roberta-base-go_emotions" # Task A (Emozioni)
    model_b_name = "cross-encoder/nli-roberta-base"   # Task B (Logica)
    base_name = "roberta-base"

    tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
    
    model_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2).to(device)
    model_a = AutoModelForSequenceClassification.from_pretrained(model_a_name).to(device)
    model_b = AutoModelForSequenceClassification.from_pretrained(model_b_name).to(device)

    # 1. REVERSE MERGING (L6-L11)
    print("\nFase 1: Merging REVERSE Layer-wise (L6-L11)...")
    dict_base, dict_a, dict_b = model_base.state_dict(), model_a.state_dict(), model_b.state_dict()
    dict_merged = {}
    la, lb = 0.6, 0.4

    for key in dict_base.keys():
        if key.startswith("roberta.encoder.layer."):
            layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
            # FONDERE SOLO I LAYER ALTI (6-11)
            if layer_num >= 6:
                dict_merged[key] = dict_base[key] + (la * (dict_a[key] - dict_base[key])) + (lb * (dict_b[key] - dict_base[key]))

    # 2. VALUTAZIONE LOGICA (NLI)
    print("\nFase 2: Valutazione Logica (SNLI)...")
    model_b.load_state_dict(dict_merged, strict=False)
    model_b.eval()
    snli = load_dataset("snli", split="test[:400]") 
    
    corrette_nli, totali_nli = 0, 0
    for item in tqdm(snli, desc="SNLI"):
        if item['label'] == -1: continue
        inputs = tokenizer_b(item['premise'], item['hypothesis'], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            pred = torch.argmax(model_b(**inputs).logits, dim=-1).item()
        
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
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            preds = (probs > 0.3).astype(int) 
        
        true_labels = np.zeros(28)
        for l in item['labels']: true_labels[l] = 1
        
        all_labels.append(true_labels)
        all_preds.append(preds)

    f1_emo = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "="*50)
    print("🚨 REPORT ABLATION STUDY (REVERSE MERGING)")
    print("="*50)
    print(f"1. LOGICA (NLI Accuracy):  { (corrette_nli/totali_nli)*100 :.2f}%")
    print(f"2. EMOZIONI (F1-Score):     { f1_emo*100 :.2f}")
    print("="*50)
    print("Se l'F1-Score crolla rispetto a 48.03, abbiamo dimostrato la nostra tesi.")

if __name__ == "__main__":
    evaluate_hybrid_reverse()