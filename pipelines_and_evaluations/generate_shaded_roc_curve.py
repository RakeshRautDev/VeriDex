import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

out_dir = "Ablation_Visuals"

def get_domain_weight(domain):
    credible = ["reuters.com", "apnews.com", "bbc.com", "politifact.com", "snopes.com", "factcheck.org"]
    unreliable = ["freedomtruthblog.net", "theonion.com", "randomnews.org", "infowars.com"]
    if any(d in domain for d in credible): return 1.5
    elif any(d in domain for d in unreliable): return 0.2
    return 1.0

def generate_shaded_roc():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    try:
        df = pd.read_csv("test_claims_dataset.csv")
    except FileNotFoundError:
        return
        
    y_true = df["true_label"].apply(lambda x: 1 if x == "Fake" else 0).values
    
    y_prob_text_only = []
    y_prob_hybrid = []
    y_prob_stance_only = []
    
    for _, row in df.iterrows():
        prob_fake_text = float(row["linguistic_prob_fake"])
        evidence_list = json.loads(row["evidence"])
        
        y_prob_text_only.append(prob_fake_text)
        
        risk_score = prob_fake_text * 100
        stance_risk_prob = 0.5
        
        if len(evidence_list) > 0:
            total_stance_score = 0
            total_weight = 0
            has_strong_debunk = False
            for ev in evidence_list:
                weight = get_domain_weight(ev["domain"])
                p_pro = ev["confidence"] if ev["stance"] == "PRO" else 1.0 - ev["confidence"]
                if weight >= 1.4 and ev["has_debunk_keywords"] and ev["confidence"] >= 0.75:
                    has_strong_debunk = True
                total_stance_score += p_pro * weight
                total_weight += weight
                
            if has_strong_debunk:
                risk_score = max(risk_score, 90.0)
                stance_risk_prob = 0.95
            else:
                avg_pro = total_stance_score / total_weight
                evidence_risk = (1.0 - avg_pro) * 100
                stance_risk_prob = (1.0 - avg_pro)
                risk_score = (risk_score * 0.4) + (evidence_risk * 0.6)
                
        y_prob_stance_only.append(stance_risk_prob)
        y_prob_hybrid.append(risk_score / 100.0)

    
    fpr_t, tpr_t, _ = roc_curve(y_true, y_prob_text_only)
    auc_t = auc(fpr_t, tpr_t)
    
    fpr_s, tpr_s, _ = roc_curve(y_true, y_prob_stance_only)
    auc_s = auc(fpr_s, tpr_s)
    
    fpr_h, tpr_h, _ = roc_curve(y_true, y_prob_hybrid)
    auc_h = auc(fpr_h, tpr_h)
    
    
    tpr_t_interp = np.interp(fpr_h, fpr_t, tpr_t)
    
    tpr_t_interp = np.minimum(tpr_t_interp, tpr_h)

    
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    
    
    plt.plot(fpr_s, tpr_s, label=f"Evidence-Only Baseline (AUC = {auc_s:.3f})", color="#27ae60", linestyle=":", linewidth=2.5)
    
    
    plt.plot(fpr_t, tpr_t, label=f"Text-Only Baseline (AUC = {auc_t:.3f})", color="#e67e22", linestyle="--", linewidth=2.5)
    
    
    plt.plot(fpr_h, tpr_h, label=f"VeriDex Hybrid (AUC = {auc_h:.3f})", color="#2980b9", linewidth=3.5)
    
    
    plt.fill_between(fpr_h, tpr_t_interp, tpr_h, color="#3498db", alpha=0.25, 
                     label="Discriminative Advantage of Retrieval")
    
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    
    plt.title("ROC Curves: Three-Way Comparison", fontsize=16, pad=15, fontweight="bold")
    plt.xlabel("False Positive Rate", fontsize=13, fontweight="bold")
    plt.ylabel("True Positive Rate", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11, frameon=True, shadow=True)
    
    plt.tight_layout()
    output_path = os.path.join(out_dir, "ROC_Shaded_Comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    

if __name__ == "__main__":
    generate_shaded_roc()
