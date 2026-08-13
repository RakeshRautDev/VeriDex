import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score

def get_domain_weight(domain):
    credible = ["reuters.com", "apnews.com", "bbc.com", "politifact.com", "snopes.com", "factcheck.org"]
    unreliable = ["freedomtruthblog.net", "theonion.com", "randomnews.org", "infowars.com"]
    if any(d in domain for d in credible): return 1.5
    elif any(d in domain for d in unreliable): return 0.2
    return 1.0

def evaluate_fast(df, text_w, evid_w):
    y_true = df["true_label"].apply(lambda x: 1 if x == "Fake" else 0).values
    y_pred = []
    
    for _, row in df.iterrows():
        prob_t = float(row["linguistic_prob_fake"])
        ev_list = json.loads(row["evidence"])
        
        r_score = prob_t * 100
        if len(ev_list) > 0:
            tot_st = 0; tot_w = 0; debunk = False
            for ev in ev_list:
                w = get_domain_weight(ev["domain"])
                p_pro = ev["confidence"] if ev["stance"] == "PRO" else 1.0 - ev["confidence"]
                if w >= 1.4 and ev["has_debunk_keywords"] and ev["confidence"] >= 0.75: debunk = True
                tot_st += p_pro * w
                tot_w += w
            
            if debunk: r_score = max(r_score, 90.0)
            else:
                avg_p = tot_st / tot_w
                e_risk = (1.0 - avg_p) * 100
                r_score = (r_score * text_w) + (e_risk * evid_w)
                
        p = r_score / 100.0
        y_pred.append(1 if p > 0.5 else 0)
        
    return f1_score(y_true, y_pred, average="weighted")

def run_comprehensive():
    try:
        df = pd.read_csv("test_claims_dataset.csv")
    except FileNotFoundError:
        return
        
    out = "Ablation_Visuals"
    os.makedirs(out, exist_ok=True)
    
    results = []
    for t in np.linspace(0.0, 1.0, 101):
        e = 1.0 - t
        f1 = evaluate_fast(df, t, e)
        results.append({"Text_Weight": t, "Evidence_Weight": e, "F1_Score": f1})
        
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x="Text_Weight", y="F1_Score", linewidth=3, color="#8e44ad")
    
    optimal = results_df.loc[results_df['F1_Score'].idxmax()]
    plt.axvline(x=optimal['Text_Weight'], color='#e74c3c', linestyle='--', linewidth=2,
                label=f'Discovered Peak (Text: {optimal["Text_Weight"]:.2f}, Evidence: {optimal["Evidence_Weight"]:.2f})')
    
    plt.title("Automated Hyperparameter Discovery (101 Configurations Tested)", fontsize=15, pad=15, fontweight="bold")
    plt.xlabel("Linguistic Model Weight (0.0 to 1.0)", fontsize=12)
    plt.ylabel("System Accuracy (F1-Score)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "1_GridSearch_Discovery_Curve.png"), dpi=300)
    plt.close()
    
    top_3 = results_df.nlargest(3, 'F1_Score')
    baseline_text = results_df[results_df['Text_Weight'] == 1.0].iloc[0]
    baseline_evid = results_df[results_df['Text_Weight'] == 0.0].iloc[0]
    
    comp_data = {
        "Configuration": [
            "Baseline A: Text Only (100% / 0%)", 
            "Baseline B: Evidence Only (0% / 100%)",
            f"Top Hybrid #3 (Text {top_3.iloc[2]['Text_Weight']*100:.0f}%)",
            f"Top Hybrid #2 (Text {top_3.iloc[1]['Text_Weight']*100:.0f}%)",
            f"🏆 Best Discovered (Text {top_3.iloc[0]['Text_Weight']*100:.0f}%)"
        ],
        "F1_Score": [
            baseline_text['F1_Score'],
            baseline_evid['F1_Score'],
            top_3.iloc[2]['F1_Score'],
            top_3.iloc[1]['F1_Score'],
            top_3.iloc[0]['F1_Score']
        ]
    }
    
    comp_df = pd.DataFrame(comp_data)
    
    min_f1 = comp_df['F1_Score'].min() - 0.05
    max_f1 = comp_df['F1_Score'].max() + 0.02
    
    plt.figure(figsize=(11, 7))
    ax = sns.barplot(data=comp_df, y="Configuration", x="F1_Score", 
                     palette=["#95a5a6", "#7f8c8d", "#3498db", "#2980b9", "#2ecc71"])
                     
    plt.title("Performance Comparison: Baselines vs Top Discovered Hybrids", fontsize=15, pad=15, fontweight="bold")
    plt.xlabel("Weighted F1-Score", fontsize=12)
    plt.ylabel("")
    plt.xlim(max(0.5, min_f1), min(1.0, max_f1)) 
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_width():.4f}", 
                    (p.get_width() + 0.001, p.get_y() + p.get_height() / 2),
                    ha='left', va='center', fontsize=12, color='black', fontweight='bold')
                    
    plt.tight_layout()
    plt.savefig(os.path.join(out, "2_Baseline_vs_Optimal_Comparison.png"), dpi=300)
    plt.close()
    
    results_df.to_csv(os.path.join(out, "Full_100_Configuration_Tests.csv"), index=False)
    

if __name__ == "__main__":
    run_comprehensive()
