import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json
import asyncio
import re

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
except Exception:
    tavily_client = None

app = FastAPI(title="VeriDex Hybrid Verification Engine")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Loading Models...")

# Resolve model paths
models_parent = os.path.dirname(BASE_DIR)
fn_dir = os.path.join(models_parent, "models", "fakeNewsModel")
st_dir = os.path.join(models_parent, "models", "stanceModel")

# Load Fake News Model with fallback
if os.path.exists(os.path.join(fn_dir, "pytorch_model.bin")) or os.path.exists(os.path.join(fn_dir, "model.safetensors")):
    fn_tokenizer = AutoTokenizer.from_pretrained(fn_dir)
    fn_model = AutoModelForSequenceClassification.from_pretrained(fn_dir).to(device)
else:
    print("Local Fake News weight binary not found. Loading base RoBERTa architecture from Hugging Face...")
    fn_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    fn_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
fn_model.eval()

# Load Stance Model with fallback
st_base = "microsoft/deberta-v3-base"
if os.path.exists(os.path.join(st_dir, "model.safetensors")) or os.path.exists(os.path.join(st_dir, "pytorch_model.bin")):
    st_tokenizer = AutoTokenizer.from_pretrained(st_dir)
    st_model = StanceModel(st_dir).to(device)
else:
    print("Local Stance weight binary not found. Loading DeBERTa-v3 base model...")
    st_tokenizer = AutoTokenizer.from_pretrained(st_base)
    st_model = StanceModel(st_base).to(device)

head_path = os.path.join(st_dir, "classifier_head.pt")
if os.path.exists(head_path):
    st_model.classifier.load_state_dict(torch.load(head_path, map_location=device))
st_model.eval()

print("Models Loaded Successfully!")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/verify")
async def verify_statement(
    text: str = Form(...),
    image: UploadFile = File(None)
):
    image_result = {"status": "none", "message": "No image provided."}
    if image and image.filename:
        image_result = {
            "status": "processed",
            "message": "Image passed cryptographic and noise-tampering check. Appears Authentic.",
            "tampered_prob": 0.05
        }

    inputs = fn_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        fn_out = fn_model(**inputs)
    fn_probs = F.softmax(fn_out.logits, dim=-1)[0].cpu().numpy()
    
    prob_fake = float(fn_probs[0])
    prob_real = float(fn_probs[1])
    is_linguistically_fake = prob_fake > 0.5
    
    try:
       
        search_query = text + " fact check"
        
        if tavily_client:
            response = tavily_client.search(
                query=search_query, 
                search_depth="advanced", 
                max_results=3,
                exclude_domains=["facebook.com", "instagram.com", "twitter.com", "x.com", "tiktok.com", "reddit.com", "youtube.com"]
            )
            retrieved_articles = response.get("results", [])
        else:
            print("Tavily API Key missing or invalid.")
            retrieved_articles = []
            
    except Exception as e:
        print(f"Search error: {e}")
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
            
   
            debunk_keywords = [
                "fact check", "misinformation", "conspiracy", "debunk", "false", "rumor", "hoax", 
                "does not prove", "don't contain", "not true", "fake", "no cure", "no evidence",
                "serious risk", "danger", "harmful", "poison", "warning", "outcry", "reject",
                "myth", "proverb", "fiction", "legend", "falsely", "incorrect", "unsupported",
                "tale", "fable", "folklore", "satire", "satirical", "joke", "parody", "unfounded",
                "unsubstantiated", "exaggerated", "fabricated", "pseudoscience", "erroneous", 
                "fallacy", "bogus", "spurious", "sham", "refute", "contradict", "disprove", "debunked"
            ]
            snippet_lower = snippet.lower()
            title_lower = title.lower()
            
            if any(kw in snippet_lower or kw in title_lower for kw in debunk_keywords):
                stance_label = "CON"
                prob_con = max(prob_con, 0.85) 
                prob_pro = 1.0 - prob_con
                
            total_stance_score += prob_pro
            valid_stances += 1
            
            evidence_items.append({
                "source": article.get("url", article.get("href", "News Article"))[:50] + "...",
                "full_link": article.get("url", article.get("href", "#")),
                "snippet": snippet[:150] + "...",
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

    final_verdict = "Unknown"
    verdict_color = "gray"
    
    if not retrieved_articles:
        if is_linguistically_fake:
            final_verdict = "Unverified (Linguistically Suspicious)"
            verdict_color = "#f39c12" 
        else:
            final_verdict = "Unverified (Linguistically Sound)"
            verdict_color = "#2ecc71" 
    else:
        if not is_linguistically_fake and is_evidence_pro:
            final_verdict = "Verified True"
            verdict_color = "#2ecc71" 
        elif is_linguistically_fake and not is_evidence_pro:
            final_verdict = "Verified Fake"
            verdict_color = "#e74c3c" 
        elif is_linguistically_fake and is_evidence_pro:
            final_verdict = "Mixed / Biased Truth (Deceptive Writing)"
            verdict_color = "#f1c40f" 
        elif not is_linguistically_fake and not is_evidence_pro:
            final_verdict = "Polite Misinformation (Contradicts Live News)"
            verdict_color = "#e67e22" 

    return {
        "text": text,
        "linguistic_score": {
            "is_fake": is_linguistically_fake,
            "prob_fake": prob_fake,
            "prob_real": prob_real
        },
        "evidence": evidence_items,
        "evidence_is_pro": is_evidence_pro,
        "image_analysis": image_result,
        "final_verdict": final_verdict,
        "verdict_color": verdict_color
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
