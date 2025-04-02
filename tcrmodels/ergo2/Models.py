import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class PaddingAutoencoder(nn.Module):
    def __init__(self, input_len, input_dim, encoding_dim):
        super(PaddingAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.input_len = input_len
        self.encoding_dim = encoding_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_len * self.input_dim, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, self.encoding_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.encoding_dim, 100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 300),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(300, self.input_len * self.input_dim)
        )

    def forward(self, padded_input):
        concat = padded_input.view(-1, self.input_len * self.input_dim)
        encoded = self.encoder(concat)
        decoded = self.decoder(encoded)
        decoding = decoded.view(-1, self.input_len, self.input_dim)
        decoding = F.softmax(decoding, dim=2)
        return decoding


class LSTM_Encoder(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, dropout):
        super(LSTM_Encoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.embedding = nn.Embedding(21 + 1, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, num_layers=2, batch_first=True, dropout=dropout)

    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(2, batch_size, self.lstm_dim, device=device),
            torch.zeros(2, batch_size, self.lstm_dim, device=device)
        )

    def lstm_pass(self, lstm, padded_embeds, lengths):
        device = padded_embeds.device
        lengths, perm_idx = lengths.sort(0, descending=True)
        padded_embeds = padded_embeds[perm_idx]
        packed = nn.utils.rnn.pack_padded_sequence(padded_embeds, lengths.cpu(), batch_first=True)
        hidden = self.init_hidden(len(lengths), device)
        lstm.flatten_parameters()
        lstm_out, _ = lstm(packed, hidden)
        unpacked, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        _, unperm_idx = perm_idx.sort(0)
        unpacked = unpacked[unperm_idx]
        lengths = lengths[unperm_idx]
        return unpacked, lengths

    def forward(self, seq, lengths):
        embeds = self.embedding(seq)
        lstm_out, lengths = self.lstm_pass(self.lstm, embeds, lengths)
        last_cell = torch.stack([lstm_out[i, l.item() - 1] for i, l in enumerate(lengths)])
        return last_cell


class AE_Encoder(nn.Module):
    def __init__(self, encoding_dim, tcr_type, input_dim=21, max_len=28, train_ae=True):
        super(AE_Encoder, self).__init__()
        self.encoding_dim = encoding_dim
        self.tcr_type = tcr_type
        self.input_dim = input_dim
        self.max_len = max_len
        self.autoencoder = PaddingAutoencoder(max_len, input_dim, encoding_dim)
        self.init_ae_params(train_ae)

    def init_ae_params(self, train_ae=True):
        ae_dir = 'TCR_Autoencoder'
        if self.tcr_type == 'alpha':
            ae_file = os.path.join(ae_dir, f'tcra_ae_dim_{self.encoding_dim}.pt')
        else:
            ae_file = os.path.join(ae_dir, f'tcrb_ae_dim_{self.encoding_dim}.pt')

        if not os.path.exists(ae_file):
            raise FileNotFoundError(f"Autoencoder checkpoint not found: {ae_file}")

        checkpoint = torch.load(ae_file, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.autoencoder.load_state_dict(checkpoint['model_state_dict'])

        if not train_ae:
            for param in self.autoencoder.parameters():
                param.requires_grad = False
            self.autoencoder.eval()

    def forward(self, padded_tcrs):
        concat = padded_tcrs.view(-1, self.max_len * self.input_dim)
        encoded = self.autoencoder.encoder(concat)
        return encoded


# Legacy class (not used in Lightning version)
class ERGO(nn.Module):
    def __init__(self, tcr_encoding_model, embedding_dim, lstm_dim, encoding_dim, dropout=0.1):
        super(ERGO, self).__init__()
        self.tcr_encoding_model = tcr_encoding_model
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.encoding_dim = encoding_dim

        if self.tcr_encoding_model == 'AE':
            self.tcr_encoder = AE_Encoder(encoding_dim=encoding_dim, tcr_type='beta')
        else:
            self.tcr_encoder = LSTM_Encoder(embedding_dim, lstm_dim, dropout)
            self.encoding_dim = lstm_dim

        self.pep_encoder = LSTM_Encoder(embedding_dim, lstm_dim, dropout)
        self.hidden_layer = nn.Linear(self.lstm_dim + self.encoding_dim, int((self.lstm_dim + self.encoding_dim)**0.5))
        self.relu = nn.LeakyReLU()
        self.output_layer = nn.Linear(int((self.lstm_dim + self.encoding_dim)**0.5), 1)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tcr_batch, peps, pep_lens):
        tcr_encoding = self.tcr_encoder(*tcr_batch)
        pep_encoding = self.pep_encoder(peps, pep_lens)
        concat = torch.cat([tcr_encoding, pep_encoding], dim=1)
        hidden = self.dropout(self.relu(self.hidden_layer(concat)))
        output = self.output_layer(hidden)
        return torch.sigmoid(output)