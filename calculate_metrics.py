import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, f1_score, precision_score, recall_score

def calculate_metrics_without_threshold(df):
    y_true = df['label']
    y_pred = df['prediction']
    
    # Calculate AUROC and AUPR
    auroc = roc_auc_score(y_true, y_pred)
    aupr = average_precision_score(y_true, y_pred)
    
    # Calculate precision-recall curve and find best F1
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    
    best_f1 = f1_scores[best_f1_idx]
    best_precision = precisions[best_f1_idx]
    best_recall = recalls[best_f1_idx]
    best_threshold = thresholds[best_f1_idx] if best_f1_idx < len(thresholds) else thresholds[-1]
    
    # Calculate accuracy at best F1 threshold
    y_pred_binary = (y_pred > best_threshold).astype(int)
    accuracy = np.mean(y_true == y_pred_binary)
    
    return {
        'AUROC': auroc,
        'AUPR': aupr,
        'Best F1': best_f1,
        'Best Precision': best_precision,
        'Best Recall': best_recall,
        'Best Threshold': best_threshold,
        'Accuracy': accuracy
    }

# Process all splits
all_metrics = []
for i in range(5):
    df = pd.read_csv(f'grazioli/transformer/results/transformer.pep+cdr3b.only-sampled-negs.hard-split.{i}.csv')
    metrics = calculate_metrics_without_threshold(df)
    metrics['Split'] = i
    all_metrics.append(metrics)

# Calculate average metrics
df_metrics = pd.DataFrame(all_metrics)
avg_metrics = df_metrics.drop('Split', axis=1).mean()
std_metrics = df_metrics.drop('Split', axis=1).std()

print("\nMetrics for each split:")
print(df_metrics)
print("\nAverage metrics across splits:")
for metric in avg_metrics.index:
    print(f"{metric}: {avg_metrics[metric]:.4f} ± {std_metrics[metric]:.4f}")

# Update the model comparison CSV
model_comparison = pd.read_csv('results/model_comparison.csv')
transformer_idx = model_comparison.index[model_comparison['Model'] == 'transformer_hard_split_random_negative'].item()
model_comparison.loc[transformer_idx, 'AUPR'] = avg_metrics['AUPR']
model_comparison.loc[transformer_idx, 'AUROC'] = avg_metrics['AUROC']
model_comparison.loc[transformer_idx, 'Accuracy'] = avg_metrics['Accuracy']
model_comparison.loc[transformer_idx, 'F1_score'] = avg_metrics['Best F1']
model_comparison.loc[transformer_idx, 'Precision'] = avg_metrics['Best Precision']
model_comparison.loc[transformer_idx, 'Recall'] = avg_metrics['Best Recall']
model_comparison.to_csv('results/model_comparison.csv', index=False) 