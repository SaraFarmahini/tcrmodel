import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve, 
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import os

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

# Define the results directory
RESULTS_BASE = "/Users/sarafarmahinifarahani/Downloads/transformer model/results/"

def load_results():
    """Load all split results and combine them"""
    results = []
    for i in range(5):
        file_path = os.path.join(RESULTS_BASE, f"transformer.pep+cdr3b.only-neg-assays.hard-split.{i}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['split'] = i
            results.append(df)
    return pd.concat(results, ignore_index=True)

def plot_roc_curves(results_df):
    """Plot ROC curves for each split"""
    plt.figure(figsize=(10, 8))
    for split in range(5):
        split_data = results_df[results_df['split'] == split]
        fpr, tpr, _ = roc_curve(split_data['label'], split_data['prediction'])
        auc = np.trapz(tpr, fpr)
        plt.plot(fpr, tpr, label=f'Split {split} (AUC = {auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Split')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_BASE, 'roc_curves.png'))
    plt.close()

def plot_pr_curves(results_df):
    """Plot Precision-Recall curves for each split"""
    plt.figure(figsize=(10, 8))
    for split in range(5):
        split_data = results_df[results_df['split'] == split]
        precision, recall, _ = precision_recall_curve(split_data['label'], split_data['prediction'])
        pr_auc = np.trapz(precision, recall)
        plt.plot(recall, precision, label=f'Split {split} (PR-AUC = {pr_auc:.3f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Split')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(RESULTS_BASE, 'pr_curves.png'))
    plt.close()

def plot_confusion_matrices(results_df):
    """Plot confusion matrices for each split"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for split in range(5):
        split_data = results_df[results_df['split'] == split]
        cm = confusion_matrix(split_data['label'], (split_data['prediction'] > 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[split], cmap='Blues')
        axes[split].set_title(f'Split {split}')
        axes[split].set_xlabel('Predicted')
        axes[split].set_ylabel('Actual')
    
    # Remove the last subplot if it exists
    if len(axes) > 5:
        axes[5].remove()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_BASE, 'confusion_matrices.png'))
    plt.close()

def plot_metric_distribution(results_df):
    """Plot distribution of predictions for positive and negative cases"""
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results_df, x='prediction', hue='label', bins=50)
    plt.title('Distribution of Predictions')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.savefig(os.path.join(RESULTS_BASE, 'prediction_distribution.png'))
    plt.close()

def plot_metrics_boxplot(results_df):
    """Plot boxplot of metrics across splits"""
    metrics = ['AUROC', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'AUPR']
    plt.figure(figsize=(12, 6))
    
    # Calculate metrics for each split
    split_metrics = []
    for split in range(5):
        split_data = results_df[results_df['split'] == split]
        metrics_dict = {
            'AUROC': roc_auc_score(split_data['label'], split_data['prediction']),
            'Accuracy': accuracy_score(split_data['label'], (split_data['prediction'] > 0.5).astype(int)),
            'Precision': precision_score(split_data['label'], (split_data['prediction'] > 0.5).astype(int)),
            'Recall': recall_score(split_data['label'], (split_data['prediction'] > 0.5).astype(int)),
            'F1 score': f1_score(split_data['label'], (split_data['prediction'] > 0.5).astype(int)),
            'AUPR': np.trapz(*precision_recall_curve(split_data['label'], split_data['prediction'])[:2])
        }
        split_metrics.append(metrics_dict)
    
    # Create boxplot
    metrics_df = pd.DataFrame(split_metrics)
    sns.boxplot(data=metrics_df)
    plt.xticks(rotation=45)
    plt.title('Performance Metrics Across Splits')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_BASE, 'metrics_boxplot.png'))
    plt.close()

def main():
    # Load results
    print("Loading results...")
    results_df = load_results()
    
    # Create visualizations
    print("Creating visualizations...")
    plot_roc_curves(results_df)
    plot_pr_curves(results_df)
    plot_confusion_matrices(results_df)
    plot_metric_distribution(results_df)
    plot_metrics_boxplot(results_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(results_df.groupby('split').agg({
        'label': ['count', 'mean'],
        'prediction': ['mean', 'std']
    }))
    
    print("\nVisualizations have been saved to:", RESULTS_BASE)

if __name__ == "__main__":
    main() 