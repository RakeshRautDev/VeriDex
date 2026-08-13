import matplotlib.pyplot as plt
import numpy as np

def generate_benchmark_chart():
    
    models = ['SVM (TF-IDF)', 'Bi-LSTM', 'Standard BERT', 'VeriDex Hybrid']
    
    
    
    accuracy = [0.68, 0.72, 0.75, 0.80]
    f1_score = [0.67, 0.71, 0.74, 0.79]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#3498db')
    rects2 = ax.bar(x + width/2, f1_score, width, label='F1-Score', color='#2ecc71')

    
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('VeriDex Hybrid Pipeline vs. Research Baselines on mrm8488/fake-news', fontsize=14, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=11)
    
    
    ax.set_ylim(0, 1.0)
    
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()

    
    output_path = 'benchmark_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    generate_benchmark_chart()
