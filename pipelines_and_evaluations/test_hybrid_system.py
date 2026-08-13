import csv
import json

def get_domain_weight(domain):
    credible = ["reuters.com", "apnews.com", "bbc.com", "politifact.com", "snopes.com", "factcheck.org"]
    unreliable = ["freedomtruthblog.net", "theonion.com", "randomnews.org", "infowars.com"]
    if any(d in domain for d in credible):
        return 1.5
    elif any(d in domain for d in unreliable):
        return 0.2
    return 1.0

def run_evaluation():
    
    correct_predictions = 0
    total = 0
    false_positives = 0 
    false_negatives = 0 
    
    with open("test_claims_dataset.csv", mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            total += 1
            true_label = row["true_label"]
            prob_fake = float(row["linguistic_prob_fake"])
            evidence_list = json.loads(row["evidence"])
            
            
            risk_score = prob_fake * 100
            
            
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
                    
                    risk_score = (risk_score * 0.4) + (evidence_risk * 0.6)
                    
            risk_score = min(max(risk_score, 0), 100)
            
            
            predicted_label = "Fake" if risk_score > 50 else "Real"
            
            if predicted_label == true_label:
                correct_predictions += 1
            else:
                if predicted_label == "Fake" and true_label == "Real":
                    false_positives += 1
                elif predicted_label == "Real" and true_label == "Fake":
                    false_negatives += 1

    accuracy = (correct_predictions / total) * 100
    precision = (total - false_positives - false_negatives) / total 
    

if __name__ == "__main__":
    try:
        run_evaluation()
    except FileNotFoundError:
