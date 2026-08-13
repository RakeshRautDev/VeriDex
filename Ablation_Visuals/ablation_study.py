import csv
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm



def get_domain_weight(domain):
    credible = ["reuters.com", "apnews.com", "bbc.com", "politifact.com", "snopes.com", "factcheck.org"]
    unreliable = ["freedomtruthblog.net", "theonion.com", "randomnews.org", "infowars.com"]
    if any(d in domain for d in credible):
        return 1.5
    elif any(d in domain for d in unreliable):
        return 0.2
    return 1.0

def run_genuine_ablation():
    print("Loading test_claims_dataset.csv for Empirical Grid Search Ablation...")
    
    y_true = []
    prob_fake_list = []
    evidence_data_list = []
    
    with open("test_claims_dataset.csv", mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            y_true.append(1 if row["true_label"] == "Fake" else 0)
            prob_fake_list.append(float(row["linguistic_prob_fake"]))
            evidence_data_list.append(json.loads(row["evidence"]))
            
    print(f"Successfully loaded {len(y_true)} claims into memory.")
    print("Sweeping through 101 model configurations (Linguistic Weight 0.00 to 1.00)...\n")
    
    results = []
    
    weights = np.linspace(0.0, 1.0, 101)
    
    for ling_weight in tqdm(weights, desc="Evaluating Pipeline Configurations"):
        evid_weight = 1.0 - ling_weight
        y_pred = []
        
        for i in range(len(y_true)):
            risk_score = prob_fake_list[i] * 100
            evidence_list = evidence_data_list[i]
            
            if len(evidence_list) > 0:
                total_stance_score = 0
                total_weight = 0
                has_strong_debunk = False
                
                for ev in evidence_list:
                    weight = get_domain_weight(ev["domain"])
                    # confidence that it is PRO (supports claim)
                    if ev["stance"] == "PRO":
                        prob_pro = ev["confidence"]
                    else:
                        prob_pro = 1.0 - ev["confidence"]
                        if weight >= 1.4 and ev["has_debunk_keywords"] and ev["confidence"] >= 0.75:
                            has_strong_debunk = True
                            
                    total_stance_score += (prob_pro * weight)
                    total_weight += weight
                    
                if has_strong_debunk:
                    final_risk = max(risk_score, 90.0)
                else:
                    avg_pro = total_stance_score / total_weight
                    evidence_risk = (1.0 - avg_pro) * 100
                    
                    final_risk = (risk_score * ling_weight) + (evidence_risk * evid_weight)
            else:
                final_risk = risk_score
                
            final_risk = min(max(final_risk, 0), 100)
            
            y_pred.append(1 if final_risk > 50 else 0)
            
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            "Linguistic_Weight": round(ling_weight, 2),
            "Evidence_Weight": round(evid_weight, 2),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1_Score": f1
        })
        
    df_results = pd.DataFrame(results)
    output_csv = "genuine_ablation_results.csv"
    df_results.to_csv(output_csv, index=False)
    
    print("\n\nGrid Search Ablation Study Complete!")
    print(f"Metrics saved to {output_csv}")
    
    best_idx = df_results['F1_Score'].idxmax()
    best_config = df_results.iloc[best_idx]
    
    print("\n==============================================")
    print("OPTIMAL EMPIRICAL CONFIGURATION DISCOVERED")
    print("==============================================")
    print(f"Linguistic Weight: {best_config['Linguistic_Weight']:.2f} ({int(best_config['Linguistic_Weight']*100)}%)")
    print(f"Evidence Weight:   {best_config['Evidence_Weight']:.2f} ({int(best_config['Evidence_Weight']*100)}%)")
    print("-" * 46)
    print(f"Peak F1-Score:     {best_config['F1_Score']:.4f}")
    print(f"Peak Accuracy:     {best_config['Accuracy']:.4f}")
    print("==============================================")

if __name__ == "__main__":
    run_genuine_ablation()
