#ERGO II - Hard split (Test: only randomized negatives) - Train: only randomized negatives



import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import trange
import random
import math
from scipy.interpolate import interp1d
import statistics 
import os
import torch

from tcrmodels.ergo2.model import ERGO2
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

def make_ergo_train_df(df):
    if "negative.source" in df.columns and  "mhc.a" in df.columns and 'mhc.source' in df.columns:
        df = df.drop(columns=["negative.source", "mhc.a", 'mhc.source']).reset_index(drop=True)
    
    map_keys = {
    'cdr3.alpha': 'tcra',
    'cdr3.beta': 'tcrb',
    'antigen.epitope': 'peptide',
    'mhc.seq': 'mhc',
    'label': 'sign'
    }
    df = df.rename(columns={c: map_keys[c] for c in df.columns})

    # the ERGO II implementation expected the following columns to be preset in the dataframe
    # even if they are not used
    df['va'] = pd.NA
    df['vb'] = pd.NA
    df['ja'] = pd.NA
    df['jb'] = pd.NA
    df['t_cell_type'] = pd.NA
    df['protein'] = pd.NA

    df['tcra'] = "UNK"  
    df['tcrb'] = df['tcrb'].str.replace('O','X')
    df['peptide'] = df['peptide'].str.replace('O','X')

    return df

def make_ergo_test_df(df):
    if "negative.source" in df.columns and  "mhc.a" in df.columns and 'mhc.source' in df.columns:
        df = df.drop(columns=["negative.source", "mhc.a", 'mhc.source']).reset_index(drop=True)
    
    map_keys = {
    'cdr3.alpha': 'TRA',
    'cdr3.beta': 'TRB',
    'antigen.epitope': 'Peptide',
    'mhc.seq': 'MHC',
    'label': 'sign'
    }
    df = df.rename(columns={c: map_keys[c] for c in df.columns})

    # the ERGO II implementation expected the following columns to be preset in the dataframe
    # even if they are not used
    df['TRAV'] = pd.NA
    df['TRBV'] = pd.NA
    df['TRAJ'] = pd.NA
    df['TRBJ'] = pd.NA
    df['T-Cell-Type'] = pd.NA
    df['Protein'] = pd.NA

    df['TRA'] = "UNK"   
    df['TRB'] = df['TRB'].str.replace('O','X')
    df['Peptide'] = df['Peptide'].str.replace('O','X')

    return df

def main():
    results_ergo2 = []

    for i in tqdm(range(5)):
        print(f"Processing split {i}...")
        
        df_train = make_ergo_train_df(
            pd.read_csv(os.path.join(DATA_BASE, "train", f"train-{i}.csv"), 
                       usecols=['cdr3.alpha', 'cdr3.beta', 'antigen.epitope', 'label', 'mhc.seq'])
        )
        
        df_test = make_ergo_test_df(
            pd.read_csv(os.path.join(DATA_BASE, "test", f"test-{i}.csv"), 
                       usecols=['cdr3.alpha', 'cdr3.beta', 'antigen.epitope', 'label', 'mhc.seq'])
        )
        
        print(f"Loaded {len(df_train)} training and {len(df_test)} test samples")
        
        # Use CPU training
        model = ERGO2(
            gpu=None,  # Use CPU
            use_alpha=False,
            random_seed=i,
            train_val_ratio=.2,
        )
        
        model.train(df_train, epochs=5)
        prediction_df = model.test(df_test)

        scores_df = get_scores(
            y_true=prediction_df['sign'].to_numpy(), 
            y_prob=prediction_df['prediction'].to_numpy(),
            y_pred=prediction_df['prediction'].to_numpy().round(),
        )
        scores_df['experiment'] = i
        results_ergo2.append(scores_df)
        df_test['prediction'] = prediction_df['prediction']
        df_test.to_csv(os.path.join(RESULTS_BASE, f"ergo2.pep+cdr3b.only-sampled-negs.hard-split.{i}.csv"), index=False)

    results_ergo2 = pd.concat(results_ergo2)
    results_ergo2.to_csv(os.path.join(RESULTS_BASE, "ergo2.pep+cdr3b.only-sampled-negs.hard-split.csv"), index=False)

if __name__ == '__main__':
    main()