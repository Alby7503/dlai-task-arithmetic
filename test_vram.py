import torch
from transformers import AutoModelForSequenceClassification

def check_vram_usage():
    # 1. Rilevamento hardware dinamico
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- ANALISI HARDWARE ---")
    print(f"Device in uso: {device}")

    total_vram = 0
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU Rilevata: {gpu_name}")
        print(f"VRAM Totale Disponibile: {total_vram:.2f} GB\n")
    else:
        print("ATTENZIONE: Nessuna GPU compatibile rilevata. Esecuzione su CPU.\n")

    # 2. Definizione dei modelli
    model_a_name = "SamLowe/roberta-base-go_emotions"
    model_b_name = "cross-encoder/nli-roberta-base"

    # 3. Caricamento Modello A
    print(f"Scaricando e caricando il Task A ({model_a_name})...")
    model_a = AutoModelForSequenceClassification.from_pretrained(model_a_name).to(device)
    
    if device.type == "cuda":
        vram_a = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"-> VRAM allocata: {vram_a:.2f} GB\n")

    # 4. Caricamento Modello B
    print(f"Scaricando e caricando il Task B ({model_b_name})...")
    model_b = AutoModelForSequenceClassification.from_pretrained(model_b_name).to(device)
    
    # 5. Report finale
    if device.type == "cuda":
        vram_total = torch.cuda.memory_allocated() / (1024 ** 3)
        vram_libera = total_vram - vram_total
        print(f"--- REPORT VRAM ---")
        print(f"VRAM Occupata dai modelli: {vram_total:.2f} GB")
        print(f"VRAM Rimanente per computazione: {vram_libera:.2f} GB")
        
        if vram_libera < 1.0:
            print("WARNING: Memoria residua molto bassa. Il merging potrebbe andare in Out Of Memory (OOM).")
        else:
            print("STATUS: Risorse sufficienti per procedere con il Task Arithmetic merging.")

if __name__ == "__main__":
    check_vram_usage()