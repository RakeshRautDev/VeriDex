import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import json

def get_domain_weight(domain):
    credible = ["reuters.com", "apnews.com", "bbc.com", "politifact.com", "snopes.com", "factcheck.org"]
    unreliable = ["freedomtruthblog.net", "theonion.com", "randomnews.org", "infowars.com"]
    if any(d in domain for d in credible): return 1.5
    elif any(d in domain for d in unreliable): return 0.2
    return 1.0

def main():
    try:
        df = pd.read_csv("test_claims_dataset.csv")
    except FileNotFoundError:
        return
        
    out_dir = "Ablation_Visuals"
    os.makedirs(out_dir, exist_ok=True)
    
    y_true = df["true_label"].apply(lambda x: 1 if x == "Fake" else 0).values
    y_pred = []
    
    for _, row in df.iterrows():
        prob_fake_text = float(row["linguistic_prob_fake"])
        evidence_list = json.loads(row["evidence"])
        
        risk_score = prob_fake_text * 100
        
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
            else:
                avg_pro = total_stance_score / total_weight
                evidence_risk = (1.0 - avg_pro) * 100
                risk_score = (risk_score * 0.4) + (evidence_risk * 0.6) 
                
        y_pred.append(1 if risk_score > 50 else 0)

    cm = confusion_matrix(y_true, y_pred)
    
    
    cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    labels = []
    for i in range(2):
        for j in range(2):
            labels.append(f"{cm[i,j]}\n({cm_perc[i,j]*100:.1f}%)")
    labels = np.asarray(labels).reshape(2,2)
    
    plt.figure(figsize=(8, 6))
    
    
    ax = sns.heatmap(cm, annot=labels, fmt="", cmap="Blues", 
                xticklabels=["Predicted: Real News", "Predicted: Fake News"], 
                yticklabels=["Actual: Real News", "Actual: Fake News"],
                annot_kws={"size": 15, "weight": "bold"}, cbar=False)
                
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        
    plt.title("Performance Matrix: Optimal 40/60 Hybrid Architecture", fontsize=16, pad=20, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "Ultimate_Confusion_Matrix.png"), dpi=300)
    plt.close()
    

if __name__ == "__main__":
    main()
