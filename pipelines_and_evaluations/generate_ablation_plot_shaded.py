import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

out_dir = "Ablation_Visuals"

def generate_shaded_plot():
    csv_path = os.path.join(out_dir, "Full_100_Configuration_Tests.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return

    
    plt.figure(figsize=(11, 7))
    sns.set_style("whitegrid")
    
    
    sns.lineplot(data=df, x="Text_Weight", y="F1_Score", linewidth=3.5, color="#2c3e50")
    
    
    peak_row = df.loc[df['F1_Score'].idxmax()]
    plt.axvline(x=peak_row['Text_Weight'], color='#e74c3c', linestyle='--', linewidth=3, 
                label=f'Optimal Peak (α = {peak_row["Text_Weight"]:.2f})')
    
    
    plt.axvspan(0.0, 0.20, color='#3498db', alpha=0.15, label='Zero-Day Failure Zone (α < 0.20)')
    
    
    plt.axvspan(0.70, 1.0, color='#e74c3c', alpha=0.15, label='LLM-Vulnerability Zone (α > 0.70)')
    
    
    plt.title("Ablation Study: Accuracy vs. Linguistic Weight (α)", fontsize=16, pad=15, fontweight="bold")
    plt.xlabel("Linguistic Weight α (0.0 to 1.0)", fontsize=13, fontweight="bold")
    plt.ylabel("Accuracy (F1-Score)", fontsize=13, fontweight="bold")
    
    
    plt.legend(loc="lower center", fontsize=11, frameon=True, shadow=True)
    
    
    plt.tight_layout()
    output_path = os.path.join(out_dir, "Ablation_Curve_Shaded.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    

if __name__ == "__main__":
    generate_shaded_plot()
