import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, Dropout, Add, Concatenate, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import numpy as np
import pandas as pd
import utils

_ENCODING = utils.blosum50_20aa

# Load and Encode Data
def encode_sequences(df, chain, cdr3a_column, cdr3b_column, peptide_column, label_column, max_pep_len=9, max_cdr3_len=30):
    if chain == 'ab':
        tcra = utils.enc_list_bl_max_len(df[cdr3a_column], _ENCODING, max_cdr3_len)
        tcrb = utils.enc_list_bl_max_len(df[cdr3b_column], _ENCODING, max_cdr3_len)
        pep = utils.enc_list_bl_max_len(df[peptide_column], _ENCODING, max_pep_len)
        x = {'cdr3a_input': tcra, 'cdr3b_input': tcrb, 'peptide_input': pep}
    else:
        tcrb = utils.enc_list_bl_max_len(df[cdr3b_column], _ENCODING, max_cdr3_len)
        pep = utils.enc_list_bl_max_len(df[peptide_column], _ENCODING, max_pep_len)
        x = {'cdr3b_input': tcrb, 'peptide_input': pep}
    
    if label_column is not None:
        y = np.array(df[label_column])
        return x, y
    return x

# Transformer Block
class TransformerBlock(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        
        # Input projection
        self.proj = Dense(embed_dim)
        
        # Multi-head attention with regularization
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att_dropout = Dropout(rate)
        self.att_layernorm = LayerNormalization(epsilon=1e-6)
        
        # Feed-forward network with regularization
        self.ffn = keras.Sequential([
            Dense(ff_dim, activation="relu", kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
            Dense(embed_dim),
        ])
        self.ffn_dropout = Dropout(rate)
        self.ffn_layernorm = LayerNormalization(epsilon=1e-6)
        
        # Add layers for residual connections
        self.add1 = Add()
        self.add2 = Add()

    def call(self, inputs, training=False):
        x = self.proj(inputs)
        
        attn_output = self.att(x, x)
        attn_output = self.att_dropout(attn_output, training=training)
        out1 = self.att_layernorm(self.add1([x, attn_output]))
        
        ffn_output = self.ffn(out1)
        ffn_output = self.ffn_dropout(ffn_output, training=training)
        return self.ffn_layernorm(self.add2([out1, ffn_output]))

# Focal Loss to handle class imbalance
def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fn(y_true, y_pred):
        epsilon = 1e-7
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        p_t = tf.where(y_true == 1, y_pred, 1 - y_pred)
        focal_loss = -alpha * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    return focal_loss_fn

# Transformer Model
def build_transformer_model(seq_length, embed_dim, num_heads, ff_dim, num_blocks, chain, num_classes=1):
    if chain == 'ab':
        cdr3a_input = Input(shape=(seq_length, 20), name='cdr3a_input')
        cdr3b_input = Input(shape=(seq_length, 20), name='cdr3b_input')
        peptide_input = Input(shape=(9, 20), name='peptide_input')
        x_cdr3a, x_cdr3b, x_peptide = cdr3a_input, cdr3b_input, peptide_input
    else:
        cdr3b_input = Input(shape=(seq_length, 20), name='cdr3b_input')
        peptide_input = Input(shape=(9, 20), name='peptide_input')
        x_cdr3b, x_peptide = cdr3b_input, peptide_input
    
    for _ in range(num_blocks):
        if chain == 'ab':
            x_cdr3a = TransformerBlock(embed_dim, num_heads, ff_dim)(x_cdr3a)
        x_cdr3b = TransformerBlock(embed_dim, num_heads, ff_dim)(x_cdr3b)
        x_peptide = TransformerBlock(embed_dim, num_heads, ff_dim)(x_peptide)
    
    if chain == 'ab':
        x_cdr3a = GlobalAveragePooling1D()(x_cdr3a)
    x_cdr3b = GlobalAveragePooling1D()(x_cdr3b)
    x_peptide = GlobalAveragePooling1D()(x_peptide)
    
    if chain == 'ab':
        x = Concatenate()([x_cdr3a, x_cdr3b, x_peptide])
    else:
        x = Concatenate()([x_cdr3b, x_peptide])
    
    x = Dense(embed_dim, activation='gelu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(num_classes, activation='sigmoid')(x)
    
    model = Model(inputs=[cdr3b_input, peptide_input], outputs=outputs) if chain == 'b' else Model(inputs=[cdr3a_input, cdr3b_input, peptide_input], outputs=outputs)
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['AUC', 'accuracy']
    )
    return model

# Load Data
df_train = pd.read_csv("sample_train.csv")
df_test = pd.read_csv("sample_test.csv")

x_train, y_train = encode_sequences(df_train, chain='b', cdr3a_column='CDR3a', cdr3b_column='CDR3b', peptide_column='peptide', label_column='binder')
x_test = encode_sequences(df_test, chain='b', cdr3a_column='CDR3a', cdr3b_column='CDR3b', peptide_column='peptide', label_column=None)

# Build & Summarize Model
model = build_transformer_model(seq_length=30, embed_dim=64, num_heads=4, ff_dim=128, num_blocks=2, chain='b')
model.summary()

# Train Model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=1)
