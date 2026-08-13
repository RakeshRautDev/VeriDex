import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def get_domain_weight(domain):
    credible = ["reuters.com", "apnews.com", "bbc.com", "politifact.com", "snopes.com", "factcheck.org"]
    unreliable = ["freedomtruthblog.net", "theonion.com", "randomnews.org", "infowars.com"]
    if any(d in domain for d in credible):
        return 1.5
    elif any(d in domain for d in unreliable):
        return 0.2
    return 1.0

def run_advanced_evaluation():
    
    try:
        df = pd.read_csv("test_claims_dataset.csv")
    except FileNotFoundError:
        return
        
    y_true = df["true_label"].apply(lambda x: 1 if x == "Fake" else 0).values
    
    y_prob_text_only = []
    y_prob_hybrid = []
    y_prob_stance_only = []
    
    y_pred_hybrid = []
    
    
    for index, row in df.iterrows():
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
                if ev["stance"] == "PRO":
                    prob_pro = ev["confidence"]
                else:
                    prob_pro = 1.0 - ev["confidence"]
                    if weight >= 1.4 and ev["has_debunk_keywords"] and ev["confidence"] >= 0.75:
                        has_strong_debunk = True
                        
                total_stance_score += (prob_pro * weight)
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
        hybrid_prob = risk_score / 100.0
        y_prob_hybrid.append(hybrid_prob)
        
        y_pred_hybrid.append(1 if hybrid_prob > 0.5 else 0)

    
    cm = confusion_matrix(y_true, y_pred_hybrid)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Predicted Real", "Predicted Fake"], yticklabels=["Actual Real", "Actual Fake"])
    plt.title("Confusion Matrix: VeriDex Hybrid Model")
    plt.tight_layout()
    plt.savefig("cm_hybrid_model.png", dpi=300)
    
    plt.figure(figsize=(8, 6))
    
    fpr_t, tpr_t, _ = roc_curve(y_true, y_prob_text_only)
    auc_t = auc(fpr_t, tpr_t)
    plt.plot(fpr_t, tpr_t, label=f"Linguistic Model Only (AUC = {auc_t:.3f})", linestyle="--")
    
    fpr_s, tpr_s, _ = roc_curve(y_true, y_prob_stance_only)
    auc_s = auc(fpr_s, tpr_s)
    plt.plot(fpr_s, tpr_s, label=f"Stance Model Only (AUC = {auc_s:.3f})", linestyle=":")
    
    fpr_h, tpr_h, _ = roc_curve(y_true, y_prob_hybrid)
    auc_h = auc(fpr_h, tpr_h)
    plt.plot(fpr_h, tpr_h, label=f"VeriDex Hybrid System (AUC = {auc_h:.3f})", linewidth=2, color="blue")
    
    plt.plot([0, 1], [0, 1], 'k--') 
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison: Ablation Study")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("roc_curve_comparison.png", dpi=300)
    

if __name__ == "__main__":
    run_advanced_evaluation()
