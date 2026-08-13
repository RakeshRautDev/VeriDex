import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json

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

class StancePredictor:
    def __init__(self, model, tokenizer, device, max_len=192):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_len = max_len

    @classmethod
    def from_saved(cls, save_dir, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tok = AutoTokenizer.from_pretrained(save_dir)
        m = StanceModel(save_dir).to(device)
        m.classifier.load_state_dict(
            torch.load(os.path.join(save_dir, "classifier_head.pt"), map_location=device)
        )
        m.eval()
        return cls(m, tok, device)

    def predict_batch(self, topics, arguments, batch_size=16):
        all_preds = []
        for i in range(0, len(topics), batch_size):
            batch_t = topics[i:i+batch_size]
            batch_a = arguments[i:i+batch_size]
            enc = self.tokenizer(
                batch_t, batch_a,
                max_length=self.max_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(enc["input_ids"], enc["attention_mask"])
                preds = logits.argmax(-1)
                preds_np = preds.cpu().numpy()
                all_preds.extend(preds_np)
                
            for t, a, p in zip(batch_t, batch_a, preds_np):
                label_str = "PRO" if p == 1 else "CON"
        return all_preds

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        dataset = load_dataset("NLP-Debater-Project/IBM-Debater-ArgKP", split="train")
        dataset = dataset.shuffle(seed=42).select(range(500))
    except Exception as e:
        return

    topics = dataset["topic"]
    arguments = dataset["argument"]
    
    true_labels = [(1 if s == 1 else 0) for s in dataset["stance"]]

    model_dir = "stanceModel"
    if not os.path.exists(model_dir):
        return
        
    predictor = StancePredictor.from_saved(model_dir, device)
    
    preds = predictor.predict_batch(topics, arguments, batch_size=32)
    
    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    
    
    metrics = {
        "model": "Stance Detection (DeBERTa)",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "samples_evaluated": len(true_labels)
    }
    
    with open("stance_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        

if __name__ == "__main__":
    main()
