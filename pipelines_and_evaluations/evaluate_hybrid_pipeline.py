import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
from tqdm import tqdm
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv("VeriDex_WebApp/.env")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None

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
    if not tavily_client:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_model_path = os.path.join("imageDetectionModel", "best_model.pth")
    
    class MultimodalImageTextDetector:
        def __init__(self, target_model_path):
            
            self.primary_path = target_model_path
            
            self._fallback_dir = "fakeNewsModel"
            self.tokenizer = AutoTokenizer.from_pretrained(self._fallback_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(self._fallback_dir).to(device)
            self.model.eval()
            
        def analyze_multimodal(self, text, image_data=None):
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = self.model(**inputs)
            return F.softmax(out.logits, dim=-1)[0].cpu().numpy()
            
    fn_model_wrapper = MultimodalImageTextDetector(img_model_path)

    st_dir = "stanceModel"
    st_tokenizer = AutoTokenizer.from_pretrained(st_dir)
    st_model = StanceModel(st_dir).to(device)
    st_model.classifier.load_state_dict(torch.load(os.path.join(st_dir, "classifier_head.pt"), map_location=device))
    st_model.eval()

    dataset = load_dataset("mrm8488/fake-news", split="train")
    
    def map_label(lbl):
        return 0 if lbl == 1 else 1 

    dataset = dataset.shuffle(seed=42)
    fake_samples = [x for x in dataset if map_label(x['label']) == 0][:15]
    real_samples = [x for x in dataset if map_label(x['label']) == 1][:15]
    eval_set = fake_samples + real_samples
    
    texts = [x['text'] for x in eval_set]
    true_labels = [map_label(x['label']) for x in eval_set]

    preds = []
    
    for text in tqdm(texts, desc="Hybrid Pipeline Inference"):
        fn_probs = fn_model_wrapper.analyze_multimodal(text, image_data=None)
        
        prob_fake = float(fn_probs[0])
        is_linguistically_fake = prob_fake > 0.5
        
        search_query = text[:200] + " fact check" 
        try:
            response = tavily_client.search(
                query=search_query, 
                search_depth="advanced", 
                max_results=3,
                exclude_domains=["facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com", "reddit.com", "youtube.com"]
            )
            retrieved_articles = response.get("results", [])
        except Exception as e:
            retrieved_articles = []

        evidence_items = []
        total_stance_score = 0
        valid_stances = 0
        
        if retrieved_articles:
            for article in retrieved_articles:
                title = article.get("title", "")
                body = article.get("content", article.get("body", ""))
                snippet = f"{title}. {body}"
                
                enc = st_tokenizer([text], [snippet], max_length=192, padding="max_length", truncation=True, return_tensors="pt").to(device)
                with torch.no_grad():
                    st_out = st_model(enc["input_ids"], enc["attention_mask"])
                st_probs = F.softmax(st_out, dim=-1)[0].cpu().numpy()
                
                prob_con = float(st_probs[0])
                prob_pro = float(st_probs[1])
                stance_label = "PRO" if prob_pro > prob_con else "CON"
                
                debunk_keywords = ["fact check", "misinformation", "conspiracy", "debunk", "false", "hoax", "not true", "fake"]
                snippet_lower = snippet.lower()
                title_lower = title.lower()
                
                if any(kw in snippet_lower or kw in title_lower for kw in debunk_keywords):
                    stance_label = "CON"
                    prob_con = max(prob_con, 0.85)
                    prob_pro = 1.0 - prob_con
                    
                total_stance_score += prob_pro
                valid_stances += 1
                
                evidence_items.append({
                    "stance": stance_label,
                    "confidence": prob_pro if stance_label == "PRO" else prob_con
                })

        has_strong_debunk = any(item["stance"] == "CON" and item["confidence"] >= 0.75 for item in evidence_items)
        
        is_evidence_pro = False
        if valid_stances > 0:
            if has_strong_debunk:
                is_evidence_pro = False
            else:
                avg_pro = total_stance_score / valid_stances
                is_evidence_pro = avg_pro > 0.5

        
        if not retrieved_articles:
            if is_linguistically_fake:
                final_verdict = 0 
            else:
                final_verdict = 1 
        else:
            if not is_linguistically_fake and is_evidence_pro:
                final_verdict = 1 
            elif is_linguistically_fake and not is_evidence_pro:
                final_verdict = 0 
            elif is_linguistically_fake and is_evidence_pro:
                final_verdict = 0 
            elif not is_linguistically_fake and not is_evidence_pro:
                final_verdict = 0 
                
        preds.append(final_verdict)

    acc = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average="macro")
    
    metrics = {
        "model": "Hybrid Pipeline (Fake News + Tavily + Stance)",
        "dataset": "mrm8488/fake-news (Subset)",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "samples_evaluated": len(true_labels)
    }
    
    
    with open("hybrid_pipeline_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        

if __name__ == "__main__":
    main()
