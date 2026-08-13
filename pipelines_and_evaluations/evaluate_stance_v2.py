import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json
import random
from tqdm import tqdm

class StanceModel(nn.Module):
    def __init__(self, model_name, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, num_labels),
        )

    def mean_pool(self, token_emb, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        summed = (token_emb * mask).sum(dim=1)
        count = mask.sum(dim=1).clamp(min=1e-9)
        return summed / count

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(out.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        dataset = load_dataset("tweet_eval", "stance_climate", split="test")
        
        
        
        filtered_dataset = [row for row in dataset if row["label"] in [1, 2]]
        random.seed(42)
        random.shuffle(filtered_dataset)
        filtered_dataset = filtered_dataset[:500]
        
    except Exception as e:
        return

    topics = ["Climate change is a real concern"] * len(filtered_dataset)
    arguments = [row["text"] for row in filtered_dataset]
    
    true_labels = [(0 if row["label"] == 1 else 1) for row in filtered_dataset]

    model_dir = "stanceModel"
    if not os.path.exists(model_dir):
        return
        
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    m = StanceModel(model_dir).to(device)
    m.classifier.load_state_dict(
        torch.load(os.path.join(model_dir, "classifier_head.pt"), map_location=device)
    )
    m.eval()
    
    batch_size = 16
    preds = []
    
    for i in tqdm(range(0, len(topics), batch_size), desc="Stance Inference", unit="batch"):
        batch_t = topics[i:i+batch_size]
        batch_a = arguments[i:i+batch_size]
        enc = tokenizer(
            batch_t, batch_a,
            max_length=192,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            logits = m(enc["input_ids"], enc["attention_mask"])
            batch_preds = logits.argmax(-1).cpu().numpy()
            preds.extend(batch_preds)
    
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    
    
    metrics = {
        "model": "Stance Detection (DeBERTa)",
        "dataset": "TweetEval (Stance Climate)",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "samples_evaluated": len(true_labels)
    }
    
    with open("stance_v2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        

if __name__ == "__main__":
    main()
