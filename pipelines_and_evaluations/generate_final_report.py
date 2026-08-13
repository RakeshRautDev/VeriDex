import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random
import gc

def plot_confusion_matrix(y_true, y_pred, labels, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()




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
    
    master_results = {}
    
    
    
    
    model_dir = "fakeNewsModel"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    batch_size = 16
    
    
    dataset_old_fn = load_dataset("GonzaloA/fake_news", split="train")
    dataset_old_fn = dataset_old_fn.shuffle(seed=42).select(range(min(500, len(dataset_old_fn))))
    
    texts = dataset_old_fn["text"]
    true_labels = [label for label in dataset_old_fn["label"]] 
    preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds.extend([int(np.argmax(prob)) for prob in probs])
        
    acc = accuracy_score(true_labels, preds)
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    master_results["Fake_News_Old_GonzaloA"] = {"accuracy": acc, "precision": p, "recall": r, "f1_score": f}
    plot_confusion_matrix(true_labels, preds, ["Fake (0)", "Real (1)"], "Fake News Model - GonzaloA (Old Dataset)", "cm_fake_news_old.png")
    
    
    dataset_new_fn = load_dataset("mrm8488/fake-news", split="train")
    dataset_new_fn = dataset_new_fn.shuffle(seed=42).select(range(min(500, len(dataset_new_fn))))
    
    texts = dataset_new_fn["text"]
    def map_fn_new(lbl): return 0 if lbl == 1 else 1
    true_labels = [map_fn_new(l) for l in dataset_new_fn["label"]]
    preds = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors='pt', truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        preds.extend([int(np.argmax(prob)) for prob in probs])
        
    acc = accuracy_score(true_labels, preds)
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    master_results["Fake_News_New_mrm8488"] = {"accuracy": acc, "precision": p, "recall": r, "f1_score": f}
    plot_confusion_matrix(true_labels, preds, ["Fake (0)", "Real (1)"], "Fake News Model - mrm8488 (New Dataset)", "cm_fake_news_new.png")
    
    
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    
    
    
    model_dir = "stanceModel"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = StanceModel(model_dir).to(device)
    model.classifier.load_state_dict(torch.load(os.path.join(model_dir, "classifier_head.pt"), map_location=device))
    model.eval()
    
    
    dataset_old_st = load_dataset("NLP-Debater-Project/IBM-Debater-ArgKP", split="train")
    dataset_old_st = dataset_old_st.shuffle(seed=42).select(range(min(500, len(dataset_old_st))))
    
    topics = dataset_old_st["topic"]
    arguments = dataset_old_st["argument"]
    true_labels = [(1 if s == 1 else 0) for s in dataset_old_st["stance"]] 
    preds = []
    
    for i in tqdm(range(0, len(topics), batch_size), desc="Inferencing"):
        batch_t = topics[i:i+batch_size]
        batch_a = arguments[i:i+batch_size]
        enc = tokenizer(batch_t, batch_a, max_length=192, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            preds.extend(logits.argmax(-1).cpu().numpy())
            
    acc = accuracy_score(true_labels, preds)
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    master_results["Stance_Old_IBM"] = {"accuracy": acc, "precision": p, "recall": r, "f1_score": f}
    plot_confusion_matrix(true_labels, preds, ["CON (0)", "PRO (1)"], "Stance Model - IBM Debater (Old Dataset)", "cm_stance_old.png")
    
    
    dataset_new_st = load_dataset("tweet_eval", "stance_climate", split="test")
    filtered = [row for row in dataset_new_st if row["label"] in [1, 2]]
    random.seed(42)
    random.shuffle(filtered)
    filtered = filtered[:500]
    
    topics = ["Climate change is a real concern"] * len(filtered)
    arguments = [row["text"] for row in filtered]
    true_labels = [(0 if row["label"] == 1 else 1) for row in filtered]
    preds = []
    
    for i in tqdm(range(0, len(topics), batch_size), desc="Inferencing"):
        batch_t = topics[i:i+batch_size]
        batch_a = arguments[i:i+batch_size]
        enc = tokenizer(batch_t, batch_a, max_length=192, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(enc["input_ids"], enc["attention_mask"])
            preds.extend(logits.argmax(-1).cpu().numpy())
            
    acc = accuracy_score(true_labels, preds)
    p, r, f, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    master_results["Stance_New_TweetEval"] = {"accuracy": acc, "precision": p, "recall": r, "f1_score": f}
    plot_confusion_matrix(true_labels, preds, ["CON (0)", "PRO (1)"], "Stance Model - TweetEval (New Dataset)", "cm_stance_new.png")
    
    
    
    
    with open("final_project_compiled_metrics.json", "w") as f:
        json.dump(master_results, f, indent=4)
        

if __name__ == "__main__":
    main()
