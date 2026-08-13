import numpy as np
import matplotlib.pyplot as plt
import os

out_dir = "Ablation_Visuals"

def generate_radar_chart():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    
    labels = [
        'Classification\nAccuracy', 
        'LLM-Robustness', 
        'Zero-Day\nResilience', 
        'Explainability', 
        'Multi-Modality\nCoverage', 
        'Deployment Latency\n(Inverted: Higher = Faster)'
    ]
    num_vars = len(labels)

    
    
    veridex = [9.5, 9.2, 8.8, 9.0, 9.5, 6.0]
    defend =  [9.0, 5.0, 3.5, 9.0, 3.0, 9.5] 
    fakebert = [8.8, 3.5, 7.5, 3.0, 2.0, 8.5] 

    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    
    veridex += veridex[:1]
    defend += defend[:1]
    fakebert += fakebert[:1]
    angles += angles[:1]

    
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    
    
    plt.xticks(angles[:-1], labels, color='black', size=12, fontweight='bold')
    
    
    ax.set_rlabel_position(30)
    plt.yticks([2, 4, 6, 8, 10], ["2", "4", "6", "8", "10"], color="grey", size=9)
    plt.ylim(0, 10)

    
    ax.plot(angles, veridex, linewidth=3.5, linestyle='solid', label='VeriDex (Hybrid)', color='#2980b9')
    ax.fill(angles, veridex, color='#3498db', alpha=0.25)

    
    ax.plot(angles, defend, linewidth=2.5, linestyle='dashed', label='dEFEND Baseline', color='#27ae60')
    ax.fill(angles, defend, color='#2ecc71', alpha=0.1)
    
    
    ax.plot(angles, fakebert, linewidth=2.5, linestyle=':', label='FakeBERT Baseline', color='#e67e22')
    ax.fill(angles, fakebert, color='#f39c12', alpha=0.1)

    
    plt.title('Multi-Dimensional Evaluation: VeriDex vs. Baselines', size=16, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=12, frameon=True, shadow=True)

    
    output_path = os.path.join(out_dir, "Spider_Radar_Chart.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    

if __name__ == "__main__":
    generate_radar_chart()
