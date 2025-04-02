import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import trange
import random
import math
from scipy.interpolate import interp1d
import statistics 
import os
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam
import sys
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc

# Add transformer model path to system path
transformer_path = '/Users/sarafarmahinifarahani/Downloads/ERGO-update/tcrmodels/transformer2/transformer model'
sys.path.append(transformer_path)

from transformer import build_transformer_model
import utils

from matplotlib import collections
from matplotlib import colors
from numpy.random import normal

# Update paths to use local directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_BASE = os.path.join(BASE_DIR, "data/train/only-sampled-negs")
TEST_BASE = os.path.join(BASE_DIR, "data/test/only-sampled-negs")
RESULTS_BASE = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

# Create directories if they don't exist
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

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
    tf.random.set_seed(random_seed)

def encode_sequences(df, chain, cdr3a_column, cdr3b_column, peptide_column, label_column, max_pep_len=9, max_cdr3_len=30):
    """Encode sequences using BLOSUM50 encoding"""
    if chain == 'ab':
        tcra = utils.enc_list_bl_max_len(df[cdr3a_column], utils.blosum50_20aa, max_cdr3_len)
        tcrb = utils.enc_list_bl_max_len(df[cdr3b_column], utils.blosum50_20aa, max_cdr3_len)
        pep = utils.enc_list_bl_max_len(df[peptide_column], utils.blosum50_20aa, max_pep_len)
        x = {'cdr3a_input': tcra, 'cdr3b_input': tcrb, 'peptide_input': pep}
    else:
        tcrb = utils.enc_list_bl_max_len(df[cdr3b_column], utils.blosum50_20aa, max_cdr3_len)
        pep = utils.enc_list_bl_max_len(df[peptide_column], utils.blosum50_20aa, max_pep_len)
        x = {'cdr3b_input': tcrb, 'peptide_input': pep}
    
    if label_column is not None:
        y = np.array(df[label_column])
        return x, y
    return x

def main():
    results_transformer = []

    for i in tqdm(range(5)):
        print(f"Processing split {i}...")
        
        # Load data with proper dtype handling
        df_train = pd.read_csv(os.path.join(DATA_BASE, f"train-{i}.csv"), low_memory=False)
        df_test = pd.read_csv(os.path.join(TEST_BASE, f"test-{i}.csv"), low_memory=False)
        
        print(f"Loaded {len(df_train)} training and {len(df_test)} test samples")
        
        # Sample a smaller subset if needed
        if len(df_train) > 50000:
            df_train = df_train.sample(n=50000, random_state=i)
            print(f"Sampled {len(df_train)} training samples")
        
        # Get max sequence lengths
        max_pep_len = max(df_train["antigen.epitope"].str.len().max(), df_test["antigen.epitope"].str.len().max())
        max_cdr3b_len = max(df_train["cdr3.beta"].str.len().max(), df_test["cdr3.beta"].str.len().max())
        
        # Encode sequences
        X_train, y_train = encode_sequences(
            df_train, 
            chain='b',
            cdr3a_column='cdr3.alpha',
            cdr3b_column='cdr3.beta',
            peptide_column='antigen.epitope',
            label_column='label',
            max_pep_len=max_pep_len,
            max_cdr3_len=max_cdr3b_len
        )
        
        X_test, y_test = encode_sequences(
            df_test,
            chain='b',
            cdr3a_column='cdr3.alpha',
            cdr3b_column='cdr3.beta',
            peptide_column='antigen.epitope',
            label_column='label',
            max_pep_len=max_pep_len,
            max_cdr3_len=max_cdr3b_len
        )
        
        # Build and compile model
        model = build_transformer_model(
            max_pep_len=max_pep_len,
            max_cdr3_len=max_cdr3b_len,
            d_model=128,
            num_heads=8,
            dff=512,
            num_encoder_layers=6,
            dropout_rate=0.1
        )
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Set up callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(CHECKPOINTS_DIR, f'transformer_random_neg_{i}.h5'),
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=2,
                verbose=1
            )
        ]
        
        # Train model
        history = model.fit(
            {'peptide_input': X_train['peptide_input'], 'cdr3_input': X_train['cdr3b_input']},
            y_train,
            batch_size=32,
            epochs=5,
            validation_split=0.15,
            callbacks=callbacks,
            verbose=1
        )
        
        # Make predictions
        y_pred = model.predict({'peptide_input': X_test['peptide_input'], 'cdr3_input': X_test['cdr3b_input']})
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        scores_df = get_scores(
            y_true=y_test,
            y_prob=y_pred.flatten(),
            y_pred=y_pred_binary.flatten()
        )
        scores_df['experiment'] = i
        results_transformer.append(scores_df)
        
        # Save predictions
        df_test['prediction'] = y_pred.flatten()
        df_test.to_csv(os.path.join(RESULTS_BASE, f"transformer.pep+cdr3b.only-sampled-negs.hard-split.{i}.csv"), index=False)
    
    # Save all results
    results_transformer_df = pd.concat(results_transformer)
    results_transformer_df.to_csv(os.path.join(RESULTS_BASE, "transformer.pep+cdr3b.only-sampled-negs.hard-split.csv"), index=False)

if __name__ == '__main__':
    main() 