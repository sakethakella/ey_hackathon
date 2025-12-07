import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        memory = self.encoder(z)
        out = self.decoder(z, memory)
        recon = self.output_proj(out)
        return recon

# LOADING CHECKPOINT 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(__file__).resolve().parent.parent / "transformer_ae_v1.pth"
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

input_dim   = ckpt["input_dim"]
d_model     = ckpt["d_model"]
nhead       = ckpt["nhead"]
num_layers  = ckpt["num_layers"]
WINDOW_SIZE = ckpt["window_size"]
FEATURE_COLS = ckpt["feature_cols"]   

model = TransformerAutoencoder(
    input_dim=input_dim,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
)
model.load_state_dict(ckpt["model_state_dict"])
model.to(DEVICE)
model.eval()

ANOMALY_THRESHOLD = ckpt.get("threshold", 0.4)  

def detect_from_window(window_readings):
    if len(window_readings) < WINDOW_SIZE:
        return False, 0.0
    window_readings = window_readings[-WINDOW_SIZE:]

    rows = []
    for r in window_readings:
        rows.append([r[col] for col in FEATURE_COLS])

    arr = np.array(rows, dtype=np.float32)         
    x = torch.from_numpy(arr).unsqueeze(0).to(DEVICE) 

    with torch.no_grad():
        recon = model(x)
        mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()

    is_anom = mse > ANOMALY_THRESHOLD
    return is_anom, mse
