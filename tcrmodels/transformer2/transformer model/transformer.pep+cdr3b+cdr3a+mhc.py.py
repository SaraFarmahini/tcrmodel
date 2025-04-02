#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm
from tqdm import trange
import random
import math
from scipy import interp
import statistics 
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Add, Concatenate, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
import utils
import sys

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc

from matplotlib import collections
from matplotlib import colors
from numpy.random import normal

# Import transformer model components
from transformer import build_transformer_model, encode_sequences

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# In[ ]:


def clean_sequences(df, peptide_column, cdr3b_column):
    """Clean sequences by removing rows with invalid amino acids"""
    # Define valid amino acids
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    
    # Filter out rows with invalid amino acids
    mask = df[peptide_column].apply(lambda x: all(aa in valid_aas for aa in x)) & \
           df[cdr3b_column].apply(lambda x: all(aa in valid_aas for aa in x))
    
    return df[mask]

def pad_sequences(sequences, max_len):
    """Pad sequences to a fixed length using 'A' as padding"""
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            padded.append(seq + 'A' * (max_len - len(seq)))
        else:
            padded.append(seq[:max_len])
    return padded

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


# In[ ]:


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)


# In[ ]:


def calculate_class_weights(y):
    """Calculate balanced class weights for imbalanced dataset"""
    unique_classes = np.unique(y)
    class_weights = {}
    total = len(y)
    n_classes = len(unique_classes)
    
    for cls in unique_classes:
        class_weights[int(cls)] = total / (n_classes * np.sum(y == cls))
    
    print(f"Calculated class weights: {class_weights}")
    return class_weights


# In[ ]:


login = os.getlogin( )
DATA_BASE = f"/Users/sarafarmahinifarahani/Downloads/transformer model/data/tc-hard/ds.hard-splits/pep+cdr3b/"
RESULTS_BASE = f"/Users/sarafarmahinifarahani/Downloads/transformer model/results/"

# Create results directory if it doesn't exist
os.makedirs(RESULTS_BASE, exist_ok=True)
os.makedirs(os.path.join(RESULTS_BASE, "checkpoints"), exist_ok=True)

# Set random seeds for reproducibility
set_random_seed(42)

results_transformer = []

# Fixed sequence lengths
MAX_PEP_LEN = 9  # Maximum peptide length (fixed for the model)
MAX_CDR3_LEN = 30  # Maximum CDR3 length

for i in tqdm(range(5)):
    try:
        print(f"\nProcessing split {i}...")
        
        # Load data
        train_path = os.path.join(DATA_BASE, "train/only-neg-assays", f"train-{i}.csv")
        test_path = os.path.join(DATA_BASE, "test/only-neg-assays", f"test-{i}.csv")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            print(f"Error: Data files not found for split {i}")
            continue
            
        df_train = pd.read_csv(train_path, low_memory=False)
        df_test = pd.read_csv(test_path, low_memory=False)
        
        print(f"Loaded {len(df_train)} training and {len(df_test)} test samples")
        
        # Clean sequences
        df_train = clean_sequences(df_train, 'antigen.epitope', 'cdr3.beta')
        df_test = clean_sequences(df_test, 'antigen.epitope', 'cdr3.beta')
        
        if len(df_train) == 0 or len(df_test) == 0:
            print(f"Warning: No valid sequences found in split {i}")
            continue
            
        print(f"After cleaning: {len(df_train)} training and {len(df_test)} test samples")
        
        # Pad sequences to fixed length
        df_train['antigen.epitope'] = pad_sequences(df_train['antigen.epitope'], MAX_PEP_LEN)
        df_test['antigen.epitope'] = pad_sequences(df_test['antigen.epitope'], MAX_PEP_LEN)
        df_train['cdr3.beta'] = pad_sequences(df_train['cdr3.beta'], MAX_CDR3_LEN)
        df_test['cdr3.beta'] = pad_sequences(df_test['cdr3.beta'], MAX_CDR3_LEN)
        
        # Encode sequences
        print("Encoding sequences...")
        x_train, y_train = encode_sequences(
            df_train, 
            chain='b', 
            cdr3a_column='cdr3.alpha', 
            cdr3b_column='cdr3.beta', 
            peptide_column='antigen.epitope', 
            label_column='label',
            max_pep_len=MAX_PEP_LEN,
            max_cdr3_len=MAX_CDR3_LEN
        )
        
        x_test = encode_sequences(
            df_test, 
            chain='b', 
            cdr3a_column='cdr3.alpha', 
            cdr3b_column='cdr3.beta', 
            peptide_column='antigen.epitope', 
            label_column=None,
            max_pep_len=MAX_PEP_LEN,
            max_cdr3_len=MAX_CDR3_LEN
        )
        
        # Calculate class weights
        class_weights = calculate_class_weights(y_train)
        print("Class weights:", class_weights)
        
        # Build and train model
        print("Building model...")
        model = build_transformer_model(
            seq_length=MAX_CDR3_LEN,
            embed_dim=64,
            num_heads=4,
            ff_dim=128,
            num_blocks=2,
            chain='b'
        )
        
        # Compile with class weights
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['AUC', 'accuracy']
        )
        
        # Add callbacks
        checkpoint_path = os.path.join(RESULTS_BASE, "checkpoints", f"model_checkpoint_{i}.h5")
        
        early_stopping = EarlyStopping(
            monitor='val_AUC',
            patience=5,
            restore_best_weights=True,
            mode='max'
        )
        
        model_checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_AUC',
            save_best_only=True,
            mode='max'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
        
        # Train model with callbacks and class weights
        print("Training model...")
        history = model.fit(
            x_train, 
            y_train, 
            epochs=30,  # Increased epochs for better convergence
            batch_size=64,  # Increased batch size for more stable gradients
            validation_split=0.15,  # Increased validation split
            callbacks=[
                early_stopping,
                model_checkpoint,
                reduce_lr,
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(RESULTS_BASE, f'logs/split_{i}'),
                    histogram_freq=1
                )
            ],
            class_weight=class_weights,
            verbose=1
        )
        
        # Make predictions
        print("Making predictions...")
        predictions = model.predict(x_test)
        
        # Apply temperature scaling for calibration
        temperature = 0.5  # Adjust this value based on validation performance
        predictions = 1 / (1 + np.exp(-np.log(predictions / (1 - predictions)) / temperature))
        
        # Create prediction dataframe
        prediction_df = df_test.copy()
        prediction_df['prediction'] = predictions
        
        # Calculate scores
        scores_df = get_scores(
            y_true=prediction_df['label'].to_numpy(),
            y_prob=prediction_df['prediction'].to_numpy(),
            y_pred=prediction_df['prediction'].to_numpy().round()
        )
        scores_df['experiment'] = i
        results_transformer.append(scores_df)
        
        # Save predictions
        prediction_path = os.path.join(RESULTS_BASE, f"transformer.pep+cdr3b.only-neg-assays.hard-split.{i}.csv")
        prediction_df.to_csv(prediction_path, index=False)
        print(f"Saved predictions to {prediction_path}")
        
        # Clear memory
        tf.keras.backend.clear_session()
        del model
        del x_train, y_train, x_test
        import gc
        gc.collect()
        
    except Exception as e:
        print(f"Error in split {i}: {str(e)}")
        import traceback
        traceback.print_exc()
        continue

if results_transformer:
    # Combine and save results
    results_transformer_df = pd.concat(results_transformer)
    results_transformer_df.to_csv(RESULTS_BASE+"transformer.pep+cdr3b.only-neg-assays.hard-split.csv", index=False)
    
    # Print average results
    print("\nAverage Results:")
    print(results_transformer_df.groupby('metrics')['score'].mean())
else:
    print("No successful experiments completed.")

# Comment out all other sections
# # # NetTCR-2.0 - Hard split (Test: only randomized negatives) - Train: only randomized negatives
# # In[ ]:
# results_nettcr2 = []
# for i in tqdm(range(5)):
#     df_train = pd.read_csv(DATA_BASE+"train/only-sampled-negs/"+f"train-{i}.csv")
#     df_test = pd.read_csv(DATA_BASE+"test/only-sampled-negs/"+f"test-{i}.csv")
#     ...

# # # NetTCR-2.0 - Hard split (Test: only randomized negatives) - Train: negative assays + randomized negatives
# # In[ ]:
# results_nettcr2 = []
# for i in tqdm(range(5)):
#     df_train = pd.read_csv(DATA_BASE+"train/only-sampled-negs.full/"+f"train-{i}.csv")
#     df_test = pd.read_csv(DATA_BASE+"test/only-sampled-negs/"+f"test-{i}.csv")
#     ...

# # # NetTCR-2.0 - Random split - Train and test: only negative assays
# # In[ ]:
# results_nettcr2 = []
# for i in tqdm(range(5)):
#     df = pd.read_csv(f"/Users/sarafarmahinifarahani/Downloads/transformer model/data/tc-hard/ds.csv")
#     ...

