import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dff, dropout):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dff = dff
        self.dropout = dropout
        self.encoderlayer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dff, dropout=self.dropout, batch_first=True)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        out = self.encoderlayer(src)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.num_layer = num_layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
    
    def forward(self, src):
        out = self.transformer_encoder(src)
        return out
    
    
class TransformerModel(nn.Module):
    def __init__(self, encoder, linear):
        super().__init__()
        self.encoder = encoder
        self.linear = linear

    def forward(self, src):
        out = F.relu(self.linear(
            self.encoder(src)))
        return out


def create_transformer(d_model, nhead, dff, num_layers, dropout, l_win):
    linear = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(dropout),
        nn.Linear(d_model*l_win, 1)
    )
    model = TransformerModel(TransformerEncoder(
        TransformerEncoderLayer(d_model, nhead, dff, dropout), num_layers),
                             linear)
    return model