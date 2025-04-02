import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results
df = pd.read_csv('results/model_comparison.csv')

# Set up the figure and subplots
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('Model Performance Comparison', fontsize=16)

# Metrics to plot
metrics = ['AUPR', 'AUROC', 'Accuracy', 'F1_score', 'Precision', 'Recall']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# Plot each metric
for idx, (ax, metric) in enumerate(zip(axes.flat, metrics)):
    bars = ax.bar(df['Model'], df[metric], color=colors[idx])
    ax.set_title(metric)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', rotation=0)

plt.tight_layout()
plt.savefig('results/model_comparison.png', bbox_inches='tight', dpi=300)
plt.close() 