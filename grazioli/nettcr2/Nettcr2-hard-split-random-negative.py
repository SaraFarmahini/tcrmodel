#NetTCR-2.0 - Hard split (Test: only randomized negatives) - Train: only randomized negatives
# 
import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import trange
import random
import math
from scipy.interpolate import interp1d
import statistics 
import os

from tcrmodels.nettcr2.model import NetTCR2
from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc

from matplotlib import collections
from matplotlib import colors
from numpy.random import normal

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

metrics = [
    'AUROC',
    'Accuracy',
    'Recall',
    'Precision',
    'F1 score',
    'AUPR'
]

def pr_auc(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    return pr_auc

def get_scores(y_true, y_prob, y_pred):
    """
    Compute a df with all classification metrics and respective scores.
    """
    
    scores = [
        roc_auc_score(y_true, y_prob),
        accuracy_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        pr_auc(y_true, y_prob)
    ]
    
    df = pd.DataFrame(data={'score': scores, 'metrics': metrics})
    return df

def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    
# Update paths to use local directories
DATA_BASE = "data"
RESULTS_BASE = "results"

results_nettcr2 = []

for i in tqdm(range(5)):
    print(f"Processing split {i}...")
    
    df_train = pd.read_csv(os.path.join(DATA_BASE, "train", f"train-{i}.csv"))
    df_test = pd.read_csv(os.path.join(DATA_BASE, "test", f"test-{i}.csv"))
    
    print(f"Loaded {len(df_train)} training and {len(df_test)} test samples")
    
    max_pep_len = max(df_train["antigen.epitope"].str.len().max(), df_test["antigen.epitope"].str.len().max())
    max_cdr3b_len = max(df_train["cdr3.beta"].str.len().max(), df_test["cdr3.beta"].str.len().max())

    model = NetTCR2(
        architecture="b", 
        single_chain_column='cdr3.beta',
        peptide_column='antigen.epitope',
        label_column='label',
        max_pep_len=max_pep_len, 
        max_cdr3_len=max_cdr3b_len
    )
    model.train(df_train, epochs=5)

    prediction_df = model.test(df_test)

    scores_df = get_scores(
        y_true=prediction_df['label'].to_numpy(), 
        y_prob=prediction_df['prediction'].to_numpy(),
        y_pred=prediction_df['prediction'].to_numpy().round(),
    )
    scores_df['experiment'] = i
    results_nettcr2.append(scores_df)
    df_test['prediction'] = prediction_df['prediction']
    df_test.to_csv(os.path.join(RESULTS_BASE, f"nettcr2.pep+cdr3b.only-sampled-negs.hard-split.{i}.csv"), index=False)

results_nettcr2_df = pd.concat(results_nettcr2)
results_nettcr2_df.to_csv(os.path.join(RESULTS_BASE, "nettcr2.pep+cdr3b.only-sampled-negs.hard-split.csv"), index=False)