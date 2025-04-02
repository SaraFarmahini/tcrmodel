import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import utils

def build_transformer_model(
    max_pep_len,
    max_cdr3_len,
    d_model=128,
    num_heads=8,
    dff=512,
    num_encoder_layers=4,
    dropout_rate=0.1
):
    """Build a transformer model for TCR-peptide binding prediction"""
    
    # Input layers
    pep_input = layers.Input(shape=(max_pep_len, 20), name='peptide_input')
    cdr3_input = layers.Input(shape=(max_cdr3_len, 20), name='cdr3_input')
    
    # Initial dense projection
    pep_embedding = layers.Dense(d_model)(pep_input)
    cdr3_embedding = layers.Dense(d_model)(cdr3_input)
    
    # Positional encoding
    pep_pos_encoding = positional_encoding(max_pep_len, d_model)
    cdr3_pos_encoding = positional_encoding(max_cdr3_len, d_model)
    
    pep_embedding = pep_embedding + pep_pos_encoding
    cdr3_embedding = cdr3_embedding + cdr3_pos_encoding
    
    # Transformer encoder layers
    for _ in range(num_encoder_layers):
        # Self-attention for peptide
        pep_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(pep_embedding, pep_embedding)
        pep_attention = layers.Dropout(dropout_rate)(pep_attention)
        pep_attention = layers.LayerNormalization()(pep_embedding + pep_attention)
        
        # Feed forward for peptide
        pep_ffn = point_wise_feed_forward_network(d_model, dff)
        pep_output = pep_ffn(pep_attention)
        pep_output = layers.Dropout(dropout_rate)(pep_output)
        pep_embedding = layers.LayerNormalization()(pep_attention + pep_output)
        
        # Self-attention for CDR3
        cdr3_attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model
        )(cdr3_embedding, cdr3_embedding)
        cdr3_attention = layers.Dropout(dropout_rate)(cdr3_attention)
        cdr3_attention = layers.LayerNormalization()(cdr3_embedding + cdr3_attention)
        
        # Feed forward for CDR3
        cdr3_ffn = point_wise_feed_forward_network(d_model, dff)
        cdr3_output = cdr3_ffn(cdr3_attention)
        cdr3_output = layers.Dropout(dropout_rate)(cdr3_output)
        cdr3_embedding = layers.LayerNormalization()(cdr3_attention + cdr3_output)
    
    # Cross-attention between peptide and CDR3
    cross_attention = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=d_model
    )(cdr3_embedding, pep_embedding)
    
    # Global average pooling
    pooled = layers.GlobalAveragePooling1D()(cross_attention)
    
    # Final dense layers
    dense = layers.Dense(256, activation='relu')(pooled)
    dense = layers.Dropout(dropout_rate)(dense)
    output = layers.Dense(1, activation='sigmoid')(dense)
    
    model = tf.keras.Model(inputs=[pep_input, cdr3_input], outputs=output)
    return model

def positional_encoding(length, depth):
    """Create positional encoding for transformer"""
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

def point_wise_feed_forward_network(d_model, dff):
    """Create a point-wise feed forward network"""
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

def encode_sequences(df, chain='b', cdr3a_column=None, cdr3b_column=None,
                    peptide_column=None, label_column=None,
                    max_pep_len=None, max_cdr3_len=None):
    """Encode sequences using BLOSUM50 encoding"""
    
    # Encode peptide sequences
    peptide_seqs = df[peptide_column].values
    peptide_encoded = utils.enc_list_bl_max_len(peptide_seqs, utils.blosum50_20aa, max_pep_len)
    
    # Encode CDR3 sequences based on chain type
    if chain == 'b':
        cdr3_seqs = df[cdr3b_column].values
    elif chain == 'a':
        cdr3_seqs = df[cdr3a_column].values
    else:
        raise ValueError("chain must be either 'a' or 'b'")
    
    cdr3_encoded = utils.enc_list_bl_max_len(cdr3_seqs, utils.blosum50_20aa, max_cdr3_len)
    
    # Prepare inputs
    x = {'peptide_input': peptide_encoded, 'cdr3b_input': cdr3_encoded} if chain == 'b' else {'peptide_input': peptide_encoded, 'cdr3a_input': cdr3_encoded}
    
    # Prepare labels if provided
    if label_column is not None:
        y = df[label_column].values
        return x, y
    return x 