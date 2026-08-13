import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def get_domain_weight(domain):
    credible = ["reuters.com", "apnews.com", "bbc.com", "politifact.com", "snopes.com", "factcheck.org"]
    unreliable = ["freedomtruthblog.net", "theonion.com", "randomnews.org", "infowars.com"]
    if any(d in domain for d in credible):
        return 1.5
    elif any(d in domain for d in unreliable):
        return 0.2
    return 1.0

def evaluate_config(df, text_weight, evidence_weight, config_name, out_dir):
    y_true = df["true_label"].apply(lambda x: 1 if x == "Fake" else 0).values
    y_prob = []
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
            else:
                avg_pro = total_stance_score / total_weight
                evidence_risk = (1.0 - avg_pro) * 100
                risk_score = (risk_score * text_weight) + (evidence_risk * evidence_weight)
                
        hybrid_prob = risk_score / 100.0
        y_prob.append(hybrid_prob)
        y_pred.append(1 if hybrid_prob > 0.5 else 0)

    
    report = classification_report(y_true, y_pred, output_dict=True)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    
    
    color_map = "Blues" if "Optimal" in config_name else ("Reds" if "Text" in config_name else "Oranges")
    
    sns.heatmap(cm, annot=True, fmt="d", cmap=color_map, xticklabels=["Predicted Real", "Predicted Fake"], yticklabels=["Actual Real", "Actual Fake"])
    plt.title(f"Confusion Matrix:\n{config_name} Split")
    plt.tight_layout()
    
    safe_name = config_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    plt.savefig(os.path.join(out_dir, f"CM_{safe_name}.png"), dpi=300)
    plt.close()
    
    
    metrics = {
        "Configuration": config_name,
        "Text_Weight": text_weight,
        "Evidence_Weight": evidence_weight,
        "Accuracy": round(report["accuracy"], 4),
        "F1_Fake": round(report["1"]["f1-score"], 4),
        "F1_Real": round(report["0"]["f1-score"], 4),
        "AUC": round(roc_auc, 4)
    }
    
    with open(os.path.join(out_dir, f"Metrics_{safe_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    return fpr, tpr, roc_auc

def main():
    
    try:
        df = pd.read_csv("test_claims_dataset.csv")
    except FileNotFoundError:
        return

    out_dir = "Ablation_Visuals"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    
    configs = [
        {"name": "Text-Heavy (80-20)", "t_weight": 0.8, "e_weight": 0.2},
        {"name": "Evidence-Heavy (20-80)", "t_weight": 0.2, "e_weight": 0.8},
        {"name": "Optimal Hybrid (40-60)", "t_weight": 0.4, "e_weight": 0.6}
    ]
    
    plt.figure(figsize=(9, 7))
    
    for cfg in configs:
        fpr, tpr, roc_auc = evaluate_config(df, cfg["t_weight"], cfg["e_weight"], cfg["name"], out_dir)
        plt.plot(fpr, tpr, label=f'{cfg["name"]} (AUC = {roc_auc:.3f})', linewidth=2.5)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel("False Positive Rate (Higher = More False Alarms)", fontsize=11)
    plt.ylabel("True Positive Rate (Higher = Caught More Fake News)", fontsize=11)
    plt.title("Ablation Study: Why 40/60 is the Optimal Split", fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(out_dir, "ROC_Master_Comparison.png"), dpi=300)
    plt.close()
    

if __name__ == "__main__":
    main()
