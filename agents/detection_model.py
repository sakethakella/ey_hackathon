import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_PATH = "synthetic_vehicle_maintenance.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "transformer_ae_v1.pth")

BATCH_SIZE = 64
WINDOW_SIZE = 50    
STEP = 10           
N_EPOCHS = 10
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_COLS = [
    "speed",
    "rpm",
    "engine_temp",
    "coolant_temp",
    "fuel_rate",
    "vibration",
    "battery_voltage",
    "tyre_pressure_fl",
    "tyre_pressure_fr",
    "tyre_pressure_rl",
    "tyre_pressure_rr",
    "brake_temp",
]

LABEL_COL = "label" 

class TimeSeriesDataset(Dataset):
    """
    Sliding-window dataset for (vehicle_id, trip_id) grouped time-series.
    """
    def __init__(self, df, feature_cols, window_size=50, step=10, only_normal=True):
        self.sequences = []
        self.labels = []

        for (_, trip_df) in df.groupby(["vehicle_id", "trip_id"]):
            if len(trip_df) < window_size:
                continue

            feats = trip_df[feature_cols].values.astype(np.float32)
            labs  = trip_df[LABEL_COL].values.astype(np.int64)

            T = len(trip_df)
            for start in range(0, T - window_size + 1, step):
                end = start + window_size
                window = feats[start:end]
                window_labels = labs[start:end]

                if only_normal and window_labels.max() != 0:
                    continue

                self.sequences.append(window)
                self.labels.append(int(window_labels.max()))

        self.sequences = np.stack(self.sequences, axis=0)
        self.labels = np.array(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)  # (T,F)
        y = self.labels[idx]
        return x, y

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

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        memory = self.encoder(z)
        out = self.decoder(z, memory)
        recon = self.output_proj(out)
        return recon

def train_transformer_ae(model, dataloader, n_epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        n_samples = 0

        for batch_x, _ in dataloader:  
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            n_samples += batch_x.size(0)

        avg_loss = total_loss / n_samples
        print(f"[Epoch {epoch+1}/{n_epochs}] Train Loss: {avg_loss:.4f}")


def eval_reconstruction_model(model, dataloader, device="cpu"):
    model.to(device)
    model.eval()

    all_errors = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            recon = model(batch_x)
            mse = torch.mean((recon - batch_x) ** 2, dim=(1,2))  
            all_errors.extend(mse.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    return np.array(all_errors), np.array(all_labels)

def main():
    print("Loading data from", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    print("Class balance (row-wise):")
    print(df[LABEL_COL].value_counts())

    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[FEATURE_COLS] = scaler.fit_transform(df_scaled[FEATURE_COLS])

    print("Building datasets...")
    train_df = df_scaled  

    train_dataset = TimeSeriesDataset(
        train_df,
        feature_cols=FEATURE_COLS,
        window_size=WINDOW_SIZE,
        step=STEP,
        only_normal=True,   
    )

    test_dataset = TimeSeriesDataset(
        df_scaled,
        feature_cols=FEATURE_COLS,
        window_size=WINDOW_SIZE,
        step=STEP,
        only_normal=False,  
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train windows: {len(train_dataset)}, Test windows: {len(test_dataset)}")

    input_dim = len(FEATURE_COLS)
    model = TransformerAutoencoder(input_dim=input_dim)

    print("Training Transformer Autoencoder on normal windows...")
    train_transformer_ae(model, train_loader, n_epochs=N_EPOCHS, lr=LR, device=DEVICE)

    print("Evaluating on normal + anomalous windows...")
    errors, labels = eval_reconstruction_model(model, test_loader, device=DEVICE)

    normal_errors = errors[labels == 0]

    anomaly_threshold = np.percentile(normal_errors, 95)

    sev_low    = np.percentile(normal_errors, 95)   
    sev_medium = np.percentile(normal_errors, 99)
    sev_high   = np.percentile(normal_errors, 99.5)

    print("\n=== Thresholds ===")
    print("Anomaly threshold (base):", anomaly_threshold)
    print("Severity thresholds:")
    print("  low   :", sev_low)
    print("  medium:", sev_medium)
    print("  high  :", sev_high)

    preds = (errors > anomaly_threshold).astype(int)

    print("\n=== Transformer AE – Sequence-level anomaly detection ===")
    print("Threshold (95th percentile of normal errors):", anomaly_threshold)
    print(classification_report(labels, preds))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))
    print("ROC-AUC (higher=more anomalous):", roc_auc_score(labels, errors))

    os.makedirs(MODEL_DIR, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_type": "TransformerAE",
        "input_dim": input_dim,
        "d_model": model.d_model,
        "nhead": model.nhead,
        "num_layers": model.num_layers,
        "feature_cols": FEATURE_COLS,
        "window_size": WINDOW_SIZE,
        "step": STEP,
        "anomaly_threshold": float(anomaly_threshold),
        "severity_thresholds": {
            "low": float(sev_low),
            "medium": float(sev_medium),
            "high": float(sev_high),
        },
    }
    torch.save(checkpoint, MODEL_PATH)
    print(f"\nSaved model checkpoint to {MODEL_PATH}")


if __name__ == "__main__":
    main()
