import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import os
from tcrmodels.ergo2.model import ERGO2
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, auc
import torch
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Force CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_grad_enabled(False)

def verify_gpu():
    print("\nGPU Information:")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Current device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}\n")


def pr_auc(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    return auc(recall, precision)


def get_scores(y_true, y_prob, y_pred):
    metrics = ['AUROC', 'Accuracy', 'Recall', 'Precision', 'F1 score', 'AUPR']
    scores = [
        roc_auc_score(y_true, y_prob),
        accuracy_score(y_true, y_pred),
        recall_score(y_true, y_pred),
        precision_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        pr_auc(y_true, y_prob)
    ]
    return pd.DataFrame({'score': scores, 'metrics': metrics})
    

def make_ergo_train_df(df):
    columns_to_drop = ["negative.source", "mhc.a", "mhc.source", "license", "v.alpha", "j.alpha", "v.beta", "d.beta", "j.beta"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore').reset_index(drop=True)
    
    df = df.rename(columns={
        'cdr3.alpha': 'tcra',
        'cdr3.beta': 'tcrb',
        'antigen.epitope': 'peptide',
        'mhc.seq': 'mhc',
        'label': 'sign'
    })

    df['va'] = pd.NA
    df['vb'] = pd.NA
    df['ja'] = pd.NA
    df['jb'] = pd.NA
    df['t_cell_type'] = pd.NA
    df['protein'] = pd.NA
    df['tcra'] = "UNK"  
    df['tcrb'] = df['tcrb'].astype(str).str.replace('O', 'X')
    df['peptide'] = df['peptide'].astype(str).str.replace('O', 'X')
    return df


def make_ergo_test_df(df):
    columns_to_drop = ["negative.source", "mhc.a", "mhc.source", "license", "v.alpha", "j.alpha", "v.beta", "d.beta", "j.beta"]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore').reset_index(drop=True)
    
    df = df.rename(columns={
        'cdr3.alpha': 'TRA',
        'cdr3.beta': 'TRB',
        'antigen.epitope': 'Peptide',
        'mhc.seq': 'MHC',
        'label': 'sign'
    })

    df['TRAV'] = pd.NA
    df['TRBV'] = pd.NA
    df['TRAJ'] = pd.NA
    df['TRBJ'] = pd.NA
    df['T-Cell-Type'] = pd.NA
    df['Protein'] = pd.NA
    df['TRA'] = "UNK"   
    df['TRB'] = df['TRB'].astype(str).str.replace('O', 'X')
    df['Peptide'] = df['Peptide'].astype(str).str.replace('O', 'X')
    return df


def save_metrics(metrics, experiment, epoch, RESULTS_BASE):
    metrics_file = os.path.join(RESULTS_BASE, f"ergo2_hard_split_only_neg_assays_exp_{experiment}_epoch_{epoch}_metrics.json")
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved for experiment {experiment}, epoch {epoch}")


class MetricsCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.current_experiment = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        try:
            print("\nSaving validation metrics...")
            metrics = {
                'epoch': trainer.current_epoch,
                'val_auroc': float(trainer.callback_metrics.get('val_auroc', 0.0)),
                'val_accuracy': float(trainer.callback_metrics.get('val_accuracy', 0.0)),
                'val_loss': float(trainer.callback_metrics.get('val_loss', 0.0))
            }
            
            metrics_file = f'results/ergo2_hard_split_only_neg_assays_exp_{self.current_experiment}_epoch_{trainer.current_epoch}_metrics.json'
            print(f"Saving metrics to {metrics_file}")
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print("Metrics saved successfully!")
            
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_test_epoch_end(self, trainer, pl_module):
        try:
            print("\nSaving test metrics...")
            metrics = {
                'epoch': trainer.current_epoch,
                'test_auroc': float(trainer.callback_metrics.get('test_auroc', 0.0)),
                'test_accuracy': float(trainer.callback_metrics.get('test_accuracy', 0.0)),
                'test_loss': float(trainer.callback_metrics.get('test_loss', 0.0))
            }
            
            metrics_file = f'results/ergo2_hard_split_only_neg_assays_exp_{self.current_experiment}_test_metrics.json'
            print(f"Saving test metrics to {metrics_file}")
            
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            print("Test metrics saved successfully!")
            
        except Exception as e:
            print(f"Error saving test metrics: {str(e)}")
            import traceback
            traceback.print_exc()

    def on_train_start(self, trainer, pl_module):
        self.current_experiment = getattr(pl_module, 'current_experiment', 0)


class LSTM_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
    
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch_size, seq_len, hidden_dim*2)
        # Take the last output
        return lstm_out[:, -1, :]  # (batch_size, hidden_dim*2)


class ERGO2Model(pl.LightningModule):
    def __init__(self, tcrb_vocab_size, pep_vocab_size, embedding_dim=32, hidden_dim=256, num_layers=2, dropout=0.1, current_experiment=0):
        super().__init__()
        self.save_hyperparameters()
        
        # TCRB encoder
        self.tcrb_encoder = LSTM_Encoder(
            vocab_size=tcrb_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Peptide encoder
        self.pep_encoder = LSTM_Encoder(
            vocab_size=pep_vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Hidden layers (note: hidden_dim * 4 because we have bidirectional LSTM * 2 encoders)
        self.hidden_layer1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.relu = nn.LeakyReLU()
        self.output_layer1 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Store current experiment
        self.current_experiment = current_experiment
        
        # Store validation outputs
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, tcrb, pep):
        # Get embeddings
        tcrb_embed = self.tcrb_encoder(tcrb)  # [batch_size, hidden_dim]
        pep_embed = self.pep_encoder(pep)     # [batch_size, hidden_dim]
        
        # Concatenate embeddings
        combined = torch.cat([tcrb_embed, pep_embed], dim=1)  # [batch_size, 2*hidden_dim]
        
        # Pass through hidden layers
        hidden = self.hidden_layer1(combined)  # [batch_size, hidden_dim]
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        
        # Output layer
        output = self.output_layer1(hidden)  # [batch_size, 1]
        return output.squeeze(-1)  # [batch_size]

    def training_step(self, batch, batch_idx):
        tcrb, pep, labels = batch
        outputs = self(tcrb, pep)
        loss = self.criterion(outputs, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tcrb, pep, labels = batch
        outputs = self(tcrb, pep)
        loss = self.criterion(outputs, labels)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Debug prints
        if batch_idx == 0:  # Only print for first batch
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            print(f"\nValidation batch {batch_idx} predictions distribution:")
            print(f"Predictions: {predictions.sum().item()}/{len(predictions)} positive")
            print(f"Actual labels: {labels.sum().item()}/{len(labels)} positive")
        
        self.validation_step_outputs.append({'val_loss': loss, 'outputs': outputs, 'labels': labels})
        return {'val_loss': loss, 'outputs': outputs, 'labels': labels}

    def on_validation_epoch_end(self):
        all_outputs = torch.cat([x['outputs'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # Calculate metrics
        auroc = self.calculate_auroc(all_outputs, all_labels)
        accuracy = self.calculate_accuracy(all_outputs, all_labels)
        
        # Log metrics
        self.log('val_auroc', auroc, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        # Save metrics
        metrics = {
            'epoch': self.current_epoch,
            'val_auroc': float(auroc),
            'val_accuracy': float(accuracy),
            'val_loss': float(self.trainer.callback_metrics.get('val_loss', 0.0))
        }
        
        metrics_file = f'results/ergo2_hard_split_only_neg_assays_exp_{self.current_experiment}_epoch_{self.current_epoch}_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Clear validation outputs
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        tcrb, pep, labels = batch
        outputs = self(tcrb, pep)
        loss = self.criterion(outputs, labels)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.test_step_outputs.append({'test_loss': loss, 'outputs': outputs, 'labels': labels})
        return {'test_loss': loss, 'outputs': outputs, 'labels': labels}

    def on_test_epoch_end(self):
        all_outputs = torch.cat([x['outputs'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        # Calculate metrics
        auroc = self.calculate_auroc(all_outputs, all_labels)
        accuracy = self.calculate_accuracy(all_outputs, all_labels)
        
        # Log metrics
        self.log('test_auroc', auroc, on_epoch=True, prog_bar=True)
        self.log('test_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        # Save metrics
        metrics = {
            'epoch': self.current_epoch,
            'test_auroc': float(auroc),
            'test_accuracy': float(accuracy),
            'test_loss': float(self.trainer.callback_metrics.get('test_loss', 0.0))
        }
        
        metrics_file = f'results/ergo2_hard_split_only_neg_assays_exp_{self.current_experiment}_test_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Clear test outputs
        self.test_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def calculate_auroc(self, outputs, labels):
        with torch.no_grad():
            outputs = torch.sigmoid(outputs)
            return torch.tensor(roc_auc_score(labels.cpu().numpy(), outputs.cpu().numpy()))

    def calculate_accuracy(self, outputs, labels):
        with torch.no_grad():
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            return (predictions == labels).float().mean()


class ERGO2Tokenizer:
    def __init__(self):
        # Initialize vocabularies
        self.tcrb_vocab = {'<pad>': 0, '<unk>': 1}
        self.pep_vocab = {'<pad>': 0, '<unk>': 1}
        
        # Load vocabularies from files if they exist
        if os.path.exists('data/vocab/tcrb_vocab.txt'):
            with open('data/vocab/tcrb_vocab.txt', 'r') as f:
                for line in f:
                    token = line.strip()
                    if token not in self.tcrb_vocab:
                        self.tcrb_vocab[token] = len(self.tcrb_vocab)
        
        if os.path.exists('data/vocab/pep_vocab.txt'):
            with open('data/vocab/pep_vocab.txt', 'r') as f:
                for line in f:
                    token = line.strip()
                    if token not in self.pep_vocab:
                        self.pep_vocab[token] = len(self.pep_vocab)
    
    def tokenize_tcrb(self, sequence):
        # Split sequence into tokens (you can modify this based on your needs)
        tokens = list(sequence)
        return [self.tcrb_vocab.get(token, self.tcrb_vocab['<unk>']) for token in tokens]
    
    def tokenize_pep(self, sequence):
        # Split sequence into tokens (you can modify this based on your needs)
        tokens = list(sequence)
        return [self.pep_vocab.get(token, self.pep_vocab['<unk>']) for token in tokens]


class ERGO2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length=30):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Determine if this is training or test data based on column names
        self.is_training = 'tcrb' in data.columns
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Get sequence data based on whether this is training or test data
        if self.is_training:
            tcrb_seq = row['tcrb']
            pep_seq = row['peptide']
            label = row['sign']
        else:
            tcrb_seq = row['TRB']
            pep_seq = row['Peptide']
            label = row['sign']
        
        # Tokenize sequences
        tcrb_tokens = self.tokenizer.tokenize_tcrb(tcrb_seq)
        pep_tokens = self.tokenizer.tokenize_pep(pep_seq)
        
        # Pad sequences
        tcrb_padded = tcrb_tokens + [self.tokenizer.tcrb_vocab['<pad>']] * (self.max_length - len(tcrb_tokens))
        pep_padded = pep_tokens + [self.tokenizer.pep_vocab['<pad>']] * (self.max_length - len(pep_tokens))
        
        # Convert to tensors
        tcrb_tensor = torch.tensor(tcrb_padded[:self.max_length], dtype=torch.long)
        pep_tensor = torch.tensor(pep_padded[:self.max_length], dtype=torch.long)
        label = torch.tensor(label, dtype=torch.float32)
        
        return tcrb_tensor, pep_tensor, label


def main():
    try:
        print("\nStarting experiments...")
        
        # Initialize tokenizer
        print("\nInitializing tokenizer...")
        tokenizer = ERGO2Tokenizer()
        
        for exp_idx in range(5):
            print(f"\nExperiment {exp_idx + 1}/5")
            
            # Load data
            print(f"Loading train data from data/train/train-{exp_idx}.csv")
            train_data = pd.read_csv(f'data/train/train-{exp_idx}.csv')
            print(f"Loading test data from data/test/test-{exp_idx}.csv")
            test_data = pd.read_csv(f'data/test/test-{exp_idx}.csv')
            
            print(f"Train set loaded: {len(train_data)} samples")
            print(f"Test set loaded: {len(test_data)} samples")
            
            # Preprocess data
            print("\nPreprocessing training data...")
            train_data = make_ergo_train_df(train_data)
            train_dataset = ERGO2Dataset(train_data, tokenizer)
            
            print("\nPreprocessing test data...")
            test_data = make_ergo_test_df(test_data)
            test_dataset = ERGO2Dataset(test_data, tokenizer)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=160, shuffle=True, num_workers=10)
            test_loader = DataLoader(test_dataset, batch_size=160, shuffle=False, num_workers=10)
            
            # Initialize model
            print("\nInitializing ERGO2 model...")
            model = ERGO2Model(
                tcrb_vocab_size=len(tokenizer.tcrb_vocab),
                pep_vocab_size=len(tokenizer.pep_vocab),
                embedding_dim=32,
                hidden_dim=256,
                num_layers=2,
                dropout=0.1,
                current_experiment=exp_idx
            )
            
            # Initialize trainer
            print("\nInitializing trainer...")
            trainer = pl.Trainer(
                max_epochs=5,
                accelerator='cpu',
                devices=1,
                callbacks=[
                    ModelCheckpoint(
                        dirpath='checkpoints',
                        filename='ergo2-{epoch:02d}-{val_loss:.2f}',
                        monitor='val_loss',
                        mode='min',
                        save_top_k=3
                    )
                ]
            )
            
            # Train model
            print("\nStarting training...")
            print("=" * 50)
            trainer.fit(model, train_loader, test_loader)
            
            print(f"\nExperiment {exp_idx + 1} completed!")
            print("=" * 50)
            
    except Exception as e:
        print(f"Error in main: {str(e)}")
        import traceback
        traceback.print_exc()
        raise e


if __name__ == '__main__':
    main() 
