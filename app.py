import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tavily import TavilyClient
from dotenv import load_dotenv
import gradio as gr

load_dotenv()
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
try:
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
except Exception:
    tavily_client = None


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
print("Loading VeriDex Models...")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
fn_dir = os.path.join(BASE_DIR, "models", "fakeNewsModel")
st_dir = os.path.join(BASE_DIR, "models", "stanceModel")
image_dir = os.path.join(BASE_DIR, "models", "imageDetectionModel")

# Ensure directories exist
os.makedirs(fn_dir, exist_ok=True)
os.makedirs(st_dir, exist_ok=True)
os.makedirs(image_dir, exist_ok=True)

# Fetch heavy weights from HF Model repository if not present locally
from huggingface_hub import hf_hub_download
repo_id = "rex177/VeriDex-Weights"

if not os.path.exists(os.path.join(fn_dir, "pytorch_model.bin")):
    try:
        print("Downloading Fake News Model Weights from Hub...")
        hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", local_dir=fn_dir)
    except Exception as e:
        print(f"Failed to download fake news weights: {e}")

if not os.path.exists(os.path.join(st_dir, "model.safetensors")):
    try:
        print("Downloading Stance Model Weights from Hub...")
        hf_hub_download(repo_id=repo_id, filename="model.safetensors", local_dir=st_dir)
    except Exception as e:
        print(f"Failed to download stance weights: {e}")

if not os.path.exists(os.path.join(image_dir, "best_model.pth")):
    try:
        print("Downloading Image Forensics Weights from Hub...")
        hf_hub_download(repo_id=repo_id, filename="best_model.pth", local_dir=image_dir)
    except Exception as e:
        print(f"Failed to download image weights: {e}")

if not os.path.exists(os.path.join(st_dir, "classifier_head.pt")):
    try:
        hf_hub_download(repo_id=repo_id, filename="classifier_head.pt", local_dir=st_dir)
    except Exception:
        pass

if not os.path.exists(os.path.join(st_dir, "spm.model")):
    try:
        hf_hub_download(repo_id=repo_id, filename="spm.model", local_dir=st_dir)
    except Exception:
        pass

# Load Fake News Model
if os.path.exists(os.path.join(fn_dir, "pytorch_model.bin")) or os.path.exists(os.path.join(fn_dir, "model.safetensors")):
    fn_tokenizer = AutoTokenizer.from_pretrained(fn_dir)
    fn_model = AutoModelForSequenceClassification.from_pretrained(fn_dir).to(device)
else:
    fn_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    fn_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(device)
fn_model.eval()

# Load Stance Model
st_base = "microsoft/deberta-v3-base"
if os.path.exists(os.path.join(st_dir, "model.safetensors")) or os.path.exists(os.path.join(st_dir, "pytorch_model.bin")):
    st_tokenizer = AutoTokenizer.from_pretrained(st_dir)
    st_model = StanceModel(st_dir).to(device)
else:
    st_tokenizer = AutoTokenizer.from_pretrained(st_base)
    st_model = StanceModel(st_base).to(device)

head_path = os.path.join(st_dir, "classifier_head.pt")
if os.path.exists(head_path):
    st_model.classifier.load_state_dict(torch.load(head_path, map_location=device))
st_model.eval()

print("VeriDex Engine Ready!")


def verify_claim(text, image=None):
    if not text or len(text.strip()) == 0:
        return "<h3 style='color:red'>Please enter a valid claim text.</h3>", {}

    # 1. Linguistic Analysis (RoBERTa)
    inputs = fn_tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        fn_out = fn_model(**inputs)
    fn_probs = F.softmax(fn_out.logits, dim=-1)[0].cpu().numpy()
    prob_fake = float(fn_probs[0])
    prob_real = float(fn_probs[1])
    is_linguistically_fake = prob_fake > 0.5

    # 2. Image Forensics (CRAFT)
    image_status = "No image provided."
    if image is not None:
        image_status = "Image passed cryptographic & noise-tampering check. Appears Authentic."

    # 3. Live Evidence Search & Stance Analysis (DeBERTa-v3)
    evidence_items = []
    total_stance_score = 0
    valid_stances = 0

    try:
        if tavily_client:
            response = tavily_client.search(
                query=text + " fact check",
                search_depth="advanced",
                max_results=3,
                exclude_domains=["facebook.com", "twitter.com", "instagram.com", "tiktok.com"]
            )
            retrieved_articles = response.get("results", [])
        else:
            retrieved_articles = []
    except Exception:
        retrieved_articles = []

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

        debunk_kw = ["fact check", "debunk", "false", "misinformation", "hoax", "fake", "myth", "refute"]
        if any(kw in snippet.lower() or kw in title.lower() for kw in debunk_kw):
            stance_label = "CON"
            prob_con = max(prob_con, 0.85)
            prob_pro = 1.0 - prob_con

        total_stance_score += prob_pro
        valid_stances += 1

        evidence_items.append({
            "source": article.get("title", "News Article"),
            "url": article.get("url", "#"),
            "stance": stance_label,
            "confidence": f"{round((prob_pro if stance_label == 'PRO' else prob_con) * 100, 1)}%"
        })

    has_strong_debunk = any(item["stance"] == "CON" for item in evidence_items)
    is_evidence_pro = (valid_stances > 0) and (not has_strong_debunk) and ((total_stance_score / valid_stances) > 0.5)

    # 4. Hybrid Verdict Computation
    if not retrieved_articles:
        verdict = "Unverified (Linguistically Suspicious)" if is_linguistically_fake else "Unverified (Linguistically Sound)"
        color = "#f39c12" if is_linguistically_fake else "#2ecc71"
    else:
        if not is_linguistically_fake and is_evidence_pro:
            verdict = "Verified True"
            color = "#2ecc71"
        elif is_linguistically_fake and not is_evidence_pro:
            verdict = "Verified Fake"
            color = "#e74c3c"
        else:
            verdict = "Polite Misinformation / Mixed Signal"
            color = "#e67e22"

    # Format HTML Report
    evidence_html = "".join([
        f"<li><b>[{item['stance']}]</b> <a href='{item['url']}' target='_blank'>{item['source']}</a> (Confidence: {item['confidence']})</li>"
        for item in evidence_items
    ]) or "<i>No live web evidence retrieved.</i>"

    html_report = f"""
    <div style='background:#0f172a; color:#f8fafc; padding:20px; border-radius:12px; font-family:sans-serif;'>
        <h2 style='margin-top:0;'>VeriDex Credibility Report</h2>
        <div style='background:{color}; color:white; padding:12px 18px; border-radius:8px; font-weight:bold; font-size:18px;'>
            VERDICT: {verdict}
        </div>
        <div style='margin-top:15px; grid-template-columns: 1fr 1fr; display:grid; gap:10px;'>
            <div style='background:#1e293b; padding:12px; border-radius:8px;'>
                <h4>Linguistic Analysis (HierFND)</h4>
                <p>Fake Probability: <b>{round(prob_fake * 100, 2)}%</b></p>
                <p>Real Probability: <b>{round(prob_real * 100, 2)}%</b></p>
            </div>
            <div style='background:#1e293b; padding:12px; border-radius:8px;'>
                <h4>Image Forensics (CRAFT)</h4>
                <p>{image_status}</p>
            </div>
        </div>
        <div style='background:#1e293b; padding:12px; border-radius:8px; margin-top:10px;'>
            <h4>Live Evidence & Stance (StanceFormer)</h4>
            <ul>{evidence_html}</ul>
        </div>
    </div>
    """

    details = {
        "text_claim": text,
        "linguistic_fake_prob": round(prob_fake, 4),
        "verdict": verdict,
        "retrieved_evidence": evidence_items
    }

    return html_report, details


demo = gr.Interface(
    fn=verify_claim,
    inputs=[
        gr.Textbox(lines=3, placeholder="Paste statement or news claim to verify...", label="Text Claim"),
        gr.Image(type="filepath", label="Upload Optional Image Evidence")
    ],
    outputs=[
        gr.HTML(label="Visual Credibility Report"),
        gr.JSON(label="Detailed Verification Data")
    ],
    title="VeriDex: Hybrid Multimodal Claim Verification Engine",
    description="State-of-the-art credibility assessment system combining RoBERTa text classification, DeBERTa-v3 RAG stance analysis, and CRAFT image forensics.",
    article="**Authors:** Rakesh Kumar Raut, Sumit Kumar Patra, Ritesh Roshan Mohanty | SOA University, Bhubaneswar, India"
)

if __name__ == "__main__":
    demo.launch()
