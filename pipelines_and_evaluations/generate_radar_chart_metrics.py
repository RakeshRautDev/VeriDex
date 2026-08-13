import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = "Ablation_Visuals"

def generate_metric_radar_chart():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    
    labels = [
        'In-Distribution\nAccuracy', 
        'Cross-Domain\nAccuracy', 
        'Macro-F1\nScore', 
        'AUC-ROC', 
        'LLM-Fake Detection\nAccuracy', 
        'Cross-Generator\nGeneralisation'
    ]
    num_vars = len(labels)

    
    
    veridex = [0.95, 0.91, 0.95, 0.96, 0.94, 0.92]
    defend =  [0.93, 0.82, 0.91, 0.90, 0.65, 0.60] 
    fakebert = [0.90, 0.78, 0.88, 0.89, 0.55, 0.50]

    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    
    veridex += veridex[:1]
    defend += defend[:1]
    fakebert += fakebert[:1]
    angles += angles[:1]

    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    
    plt.xticks(angles[:-1], labels, color='black', size=12, fontweight='bold')
    
    
    ax.set_rlabel_position(30)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
    plt.ylim(0, 1.0)

    
    ax.plot(angles, veridex, linewidth=3.5, linestyle='solid', label='VeriDex (Hybrid)', color='#2980b9')
    ax.fill(angles, veridex, color='#3498db', alpha=0.25)

    
    ax.plot(angles, defend, linewidth=2.5, linestyle='dashed', label='dEFEND Baseline', color='#27ae60')
    ax.fill(angles, defend, color='#2ecc71', alpha=0.1)
    
    
    ax.plot(angles, fakebert, linewidth=2.5, linestyle=':', label='FakeBERT Baseline', color='#e67e22')
    ax.fill(angles, fakebert, color='#f39c12', alpha=0.1)

    
    plt.title('Radar Chart: Robustness and Classification Metrics', size=16, fontweight='bold', y=1.12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12, frameon=True, shadow=True)

    
    output_path = os.path.join(out_dir, "Radar_Chart_Quantitative.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    generate_metric_radar_chart()
