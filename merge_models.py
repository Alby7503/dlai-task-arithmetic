import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def evaluate_sweep():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("--- INIZIALIZZAZIONE AMBIENTE ---")

    base_name = "roberta-base"
    model_a_name = "SamLowe/roberta-base-go_emotions"
    model_b_name = "cross-encoder/nli-roberta-base"

    print("Caricamento modelli in VRAM (solo la prima volta)...")
    tokenizer_a = AutoTokenizer.from_pretrained(model_a_name)
    tokenizer_b = AutoTokenizer.from_pretrained(model_b_name)

    model_base = AutoModelForSequenceClassification.from_pretrained(
        base_name, num_labels=2).to(device)
    model_a = AutoModelForSequenceClassification.from_pretrained(
        model_a_name).to(device)
    model_b = AutoModelForSequenceClassification.from_pretrained(
        model_b_name).to(device)

    # Estraiamo i dizionari originali
    dict_base = model_base.state_dict()
    dict_a = model_a.state_dict()
    dict_b = model_b.state_dict()

    # Pre-calcoliamo i Task Vectors (risparmia tempo nel ciclo)
    task_vectors_a = {k: dict_a[k] - dict_base[k]
                      for k in dict_base if k.startswith("roberta.")}
    task_vectors_b = {k: dict_b[k] - dict_base[k]
                      for k in dict_base if k.startswith("roberta.")}

    # Definiamo le combinazioni di Lambda da testare (Emozioni, Logica)
    # Esempio: (0.2, 0.8) significa 20% Emozioni, 80% Logica
    lambda_pairs = [
        (0.2, 0.8),
        (0.4, 0.6),
        (0.5, 0.5),  # La nostra baseline di ieri
        (0.6, 0.4),
        (0.8, 0.2)
    ]

    # Frasi di test
    text_emo = "I am so incredibly happy, but also a little bit nervous about this exam!"
    inputs_emo = tokenizer_a(
        text_emo, return_tensors="pt", truncation=True, max_length=512).to(device)

    premise = "A man is playing soccer in the park."
    hypothesis = "Someone is playing a sport outside."
    inputs_nli = tokenizer_b(
        premise, hypothesis, return_tensors="pt", truncation=True).to(device)

    print("\n==================================================")
    print("INIZIO LAMBDA SWEEP")
    print("==================================================\n")

    for la, lb in lambda_pairs:
        print(f">>> Test con Lambda: Emozioni={la:.1f} | Logica={lb:.1f}")

        # 1. Creiamo il dizionario fuso per questo ciclo
        dict_merged = {}
        for key in dict_base.keys():
            if key.startswith("roberta."):
                dict_merged[key] = dict_base[key] + \
                    (la * task_vectors_a[key]) + (lb * task_vectors_b[key])

        # 2. Test Emozioni
        model_a.load_state_dict(dict_merged, strict=False)
        model_a.eval()
        with torch.no_grad():
            out_emo = model_a(**inputs_emo)
        prob_emo = torch.nn.functional.softmax(out_emo.logits, dim=-1)
        top_emo_prob, top_emo_idx = torch.topk(prob_emo, 1)
        best_emo = model_a.config.id2label[top_emo_idx[0][0].item()]

        # 3. Test Logica (NLI)
        model_b.load_state_dict(dict_merged, strict=False)
        model_b.eval()
        with torch.no_grad():
            out_nli = model_b(**inputs_nli)
        prob_nli = torch.nn.functional.softmax(out_nli.logits, dim=-1)
        # Indice 1 di solito è Entailment
        entailment_prob = prob_nli[0][1].item() * 100

        # Stampiamo i risultati di questa configurazione
        print(
            f"    Emozione prevalente: {best_emo} ({top_emo_prob[0][0].item()*100:.1f}%)")
        print(f"    Sicurezza Logica (Entailment): {entailment_prob:.1f}%\n")


if __name__ == "__main__":
    evaluate_sweep()
