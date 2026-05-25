import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging as hf_logging

hf_logging.set_verbosity_error()
from datasets import load_dataset
from tqdm import tqdm

def evaluate_reverse():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- ESPERIMENTO 3: REVERSE LAYER-WISE MERGING ---")
    print("LOGICA: Fonderemo SOLO i Layer da 6 a 11 (Top Layers).")

    model_b_name = "cross-encoder/nli-roberta-base"
    model_a_name = "SamLowe/roberta-base-go_emotions"
    base_name = "roberta-base"

    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)
    model_base = AutoModelForSequenceClassification.from_pretrained(base_name, num_labels=2).to(device)
    model_a = AutoModelForSequenceClassification.from_pretrained(model_a_name).to(device)
    model_b = AutoModelForSequenceClassification.from_pretrained(model_b_name).to(device)

    dict_base, dict_a, dict_b = model_base.state_dict(), model_a.state_dict(), model_b.state_dict()
    dict_merged = {}
    la, lb = 0.6, 0.4 # Manteniamo la stessa ratio per coerenza

    for key in dict_base.keys():
        if key.startswith("roberta.encoder.layer."):
            layer_num = int(key.split("encoder.layer.")[1].split(".")[0])
            # FONDERE SOLO LAYER ALTI (6-11)
            if layer_num >= 6:
                task_vector_a = dict_a[key] - dict_base[key]
                task_vector_b = dict_b[key] - dict_base[key]
                dict_merged[key] = dict_base[key] + (la * task_vector_a) + (lb * task_vector_b)

    print("\nFase 1: Impianto 'Reverse' completato.")
    model_b.load_state_dict(dict_merged, strict=False)
    model_b.eval()

    print("Fase 2: Valutazione su SNLI Dataset (500 samples)...")
    snli = load_dataset("snli", split="test[:500]")
    
    corrette, totali = 0, 0
    for item in tqdm(snli):
        if item['label'] == -1: continue
        inputs = tokenizer_b(item['premise'], item['hypothesis'], return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            pred = torch.argmax(model_b(**inputs).logits, dim=-1).item()
        
        mapped = {0: 2, 1: 0, 2: 1}[pred]
        if mapped == item['label']: corrette += 1
        totali += 1

    print(f"\nACCURATEZZA REVERSE (L6-L11 fusi): {(corrette/totali)*100:.2f}%")

if __name__ == "__main__":
    evaluate_reverse()