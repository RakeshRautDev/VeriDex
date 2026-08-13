import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import numpy as np
from tqdm import tqdm

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        dataset = load_dataset("mrm8488/fake-news", split="train")
    except Exception as e:
        return
        

    def map_label(lbl):
        if lbl == 1: return 0 
        else: return 1 
        
    dataset = dataset.shuffle(seed=42).select(range(min(500, len(dataset))))
    texts = dataset["text"]
    true_labels = [map_label(l) for l in dataset["label"]]
    
    model_dir = "fakeNewsModel"
    if not os.path.exists(model_dir):
        return
        
    model_bin = os.path.join(model_dir, "pytorch_model.bin")
    alt_bin = os.path.join(model_dir, "best_model_fake_news_v3.bin")
    if not os.path.exists(model_bin) and os.path.exists(alt_bin):
        os.rename(alt_bin, model_bin)
        
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    
    batch_size = 16
    preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Fake News Inference", unit="batch"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        batch_preds = [int(np.argmax(prob)) for prob in probs]
        preds.extend(batch_preds)
        
     
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    
    
    metrics = {
        "model": "Fake News Detection (RoBERTa)",
        "dataset": "mrm8488/fake-news",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "samples_evaluated": len(true_labels)
    }
    
    with open("fake_news_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        

if __name__ == "__main__":
    main()
