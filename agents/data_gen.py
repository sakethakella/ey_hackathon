"""
Synthetic dataset generator for predictive maintenance
for vehicles (Hero bikes, Mahindra cars).

Includes:
1) Physics-inspired sensor simulation
2) Anomaly injection
3) VAE and GAN for time-series generation (PyTorch)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
!pip install torch

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# 1. CONFIG
# ============================================================

@dataclass
class VehicleConfig:
    vehicle_type: str  # "bike" or "car"
    n_vehicles: int
    n_trips_per_vehicle: int
    trip_length: int  # time steps per trip
    dt: float = 1.0   # time step in seconds

# Example sensor list (you can add/remove later)
SENSOR_COLUMNS = [
    "speed",            # km/h
    "rpm",              # engine revolutions per minute
    "engine_temp",      # °C
    "coolant_temp",     # °C
    "fuel_rate",        # L/h
    "vibration",        # arbitrary amplitude
    "battery_voltage",  # V
    "tyre_pressure_fl", # front-left (for bikes, use single tyre_pressure)
    "tyre_pressure_fr", # front-right
    "tyre_pressure_rl", # rear-left
    "tyre_pressure_rr", # rear-right
    "brake_temp"        # °C
]

ANOMALY_TYPES = [
    "engine_overheat",
    "misfire_vibration",
    "battery_failure",
    "tyre_leak",
    "brake_overheat"
]

# 2. PHYSICS-INSPIRED NORMAL BEHAVIOUR

def simulate_normal_trip(config: VehicleConfig, vehicle_id: int, trip_id: int) -> pd.DataFrame:
    """
    Simulate one normal trip for a single vehicle using simple
    physics-inspired rules.
    """

    T = config.trip_length
    dt = config.dt

    # Time vector
    t = np.arange(T) * dt

    
    # 2.1 Driver behaviour: speed profile
    # Simple pattern: accelerate, cruise, decelerate with noise
    max_speed = 60 if config.vehicle_type == "bike" else 100
    accel_phase = int(T * 0.2)
    cruise_phase = int(T * 0.6)

    speed = np.zeros(T)
    # accelerate
    speed[:accel_phase] = np.linspace(0, max_speed, accel_phase)
    # cruise
    speed[accel_phase:accel_phase+cruise_phase] = max_speed + np.random.normal(0, 3, cruise_phase)
    # decelerate
    remaining = T - (accel_phase + cruise_phase)
    if remaining > 0:
        speed[-remaining:] = np.linspace(max_speed, 0, remaining)

    speed = np.clip(speed + np.random.normal(0, 2, T), 0, None)


    # 2.2 Engine RPM (simple proportional relation to speed)
    
    gear_ratio = 60 if config.vehicle_type == "bike" else 45
    rpm = speed * gear_ratio + np.random.normal(0, 50, T)
    rpm = np.clip(rpm, 800, None)  # idle ~800 rpm

    
    # 2.3 Engine / coolant temperature (first-order dynamics)
    
    ambient_temp = 30  # °C, assume Indian city
    engine_temp = np.zeros(T)
    coolant_temp = np.zeros(T)

    engine_temp[0] = ambient_temp
    coolant_temp[0] = ambient_temp

    for i in range(1, T):
        load = speed[i] / max_speed + rpm[i] / (gear_ratio * max_speed)  # simple load
        load = np.clip(load, 0, 2)

        # engine heats up with load, cools otherwise
        engine_temp[i] = engine_temp[i-1] + 0.3*load - 0.05*(engine_temp[i-1]-ambient_temp)
        # coolant temp lags engine temp
        coolant_temp[i] = coolant_temp[i-1] + 0.2*(engine_temp[i] - coolant_temp[i-1]) - 0.03*(coolant_temp[i-1]-ambient_temp)

    engine_temp += np.random.normal(0, 0.5, T)
    coolant_temp += np.random.normal(0, 0.5, T)


    # 2.4 Fuel rate (higher when speed & rpm high)
    
    base_fuel_rate = 1.5 if config.vehicle_type == "bike" else 4.0  # L/h baseline
    fuel_rate = base_fuel_rate + 0.02 * speed + 0.0005 * (rpm - 800)
    fuel_rate = np.clip(fuel_rate + np.random.normal(0, 0.05, T), 0, None)

    
    # 2.5 Vibration

    # Combine baseline + speed^2 relation (more speed → more vibration)
    vibration = 0.1 + 0.0005 * speed**2 + np.random.normal(0, 0.02, T)

    
    # 2.6 Battery voltage
    
    # Slightly fluctuating, around 12.8V (bike) / 13.5V (car)
    nominal_voltage = 12.8 if config.vehicle_type == "bike" else 13.5
    battery_voltage = nominal_voltage + np.random.normal(0, 0.05, T)

    
    # 2.7 Tyre pressure (car) / approximated for bike
    
    if config.vehicle_type == "bike":
        # treat front-left as main, others dummy copy
        base_pressure = 32  # psi
        tyre_pressure_fl = base_pressure + np.random.normal(0, 0.1, T)
        tyre_pressure_fr = tyre_pressure_fl.copy()
        tyre_pressure_rl = tyre_pressure_fl.copy()
        tyre_pressure_rr = tyre_pressure_fl.copy()
    else:
        # Mahindra car example
        base_pressure_front = 32
        base_pressure_rear = 34
        tyre_pressure_fl = base_pressure_front + np.random.normal(0, 0.1, T)
        tyre_pressure_fr = base_pressure_front + np.random.normal(0, 0.1, T)
        tyre_pressure_rl = base_pressure_rear + np.random.normal(0, 0.1, T)
        tyre_pressure_rr = base_pressure_rear + np.random.normal(0, 0.1, T)

    
    # 2.8 Brake temperature
    
    brake_temp = np.zeros(T)
    brake_temp[0] = ambient_temp
    for i in range(1, T):
        braking = max(0, speed[i-1] - speed[i])  # deceleration
        brake_temp[i] = brake_temp[i-1] + 0.2*braking - 0.1*(brake_temp[i-1] - ambient_temp)
    brake_temp += np.random.normal(0, 0.5, T)


    # Build dataframe
    
    df = pd.DataFrame({
        "time": t,
        "vehicle_id": vehicle_id,
        "trip_id": trip_id,
        "vehicle_type": config.vehicle_type,
        "speed": speed,
        "rpm": rpm,
        "engine_temp": engine_temp,
        "coolant_temp": coolant_temp,
        "fuel_rate": fuel_rate,
        "vibration": vibration,
        "battery_voltage": battery_voltage,
        "tyre_pressure_fl": tyre_pressure_fl,
        "tyre_pressure_fr": tyre_pressure_fr,
        "tyre_pressure_rl": tyre_pressure_rl,
        "tyre_pressure_rr": tyre_pressure_rr,
        "brake_temp": brake_temp,
    })

    # Labels for anomaly detection
    df["label"] = 0  # 0 = normal, 1 = anomaly
    df["anomaly_type"] = "none"

    return df


# 3. ANOMALY INJECTION


def inject_engine_overheat(df: pd.DataFrame):
    T = len(df)
    start = random.randint(int(T*0.4), int(T*0.7))
    duration = random.randint(int(T*0.1), int(T*0.2))
    end = min(T, start + duration)

    # Gradually push engine / coolant temp upward
    for i in range(start, end):
        df.loc[i, "engine_temp"] += 0.5 * (i - start)
        df.loc[i, "coolant_temp"] += 0.3 * (i - start)

    df.loc[start:end, "label"] = 1
    df.loc[start:end, "anomaly_type"] = "engine_overheat"
    return df


def inject_misfire_vibration(df: pd.DataFrame):
    T = len(df)
    start = random.randint(int(T*0.3), int(T*0.6))
    duration = random.randint(int(T*0.05), int(T*0.15))
    end = min(T, start + duration)

    # Oscillatory rpm + spikes in vibration
    for i in range(start, end):
        df.loc[i, "rpm"] += 300 * np.sin(0.5*(i-start)) + np.random.normal(0, 50)
        df.loc[i, "vibration"] += np.random.uniform(0.3, 0.8)

    df.loc[start:end, "label"] = 1
    df.loc[start:end, "anomaly_type"] = "misfire_vibration"
    return df


def inject_battery_failure(df: pd.DataFrame):
    T = len(df)
    start = random.randint(int(T*0.2), int(T*0.5))
    duration = random.randint(int(T*0.1), int(T*0.3))
    end = min(T, start + duration)

    # Voltage sag
    volt_drop = np.linspace(0, 3.0, duration)  # drop ~3V
    for idx, i in enumerate(range(start, end)):
        df.loc[i, "battery_voltage"] -= volt_drop[idx] + np.random.normal(0, 0.1)

    df.loc[start:end, "label"] = 1
    df.loc[start:end, "anomaly_type"] = "battery_failure"
    return df


def inject_tyre_leak(df: pd.DataFrame):
    T = len(df)
    start = random.randint(int(T*0.3), int(T*0.5))
    duration = random.randint(int(T*0.2), int(T*0.4))
    end = min(T, start + duration)

    # Smooth pressure drop in one tyre
    pressure_drop = np.linspace(0, 10.0, end-start)  # drop ~10 psi
    tyre_col = random.choice(["tyre_pressure_fl", "tyre_pressure_fr",
                              "tyre_pressure_rl", "tyre_pressure_rr"])
    for idx, i in enumerate(range(start, end)):
        df.loc[i, tyre_col] -= pressure_drop[idx] + np.random.normal(0, 0.2)

    df.loc[start:end, "label"] = 1
    df.loc[start:end, "anomaly_type"] = "tyre_leak"
    return df


def inject_brake_overheat(df: pd.DataFrame):
    T = len(df)
    start = random.randint(int(T*0.4), int(T*0.7))
    duration = random.randint(int(T*0.1), int(T*0.2))
    end = min(T, start + duration)

    # Heat up brakes
    for i in range(start, end):
        df.loc[i, "brake_temp"] += 2.0 * (i - start)

    df.loc[start:end, "label"] = 1
    df.loc[start:end, "anomaly_type"] = "brake_overheat"
    return df


ANOMALY_FUNCTIONS = {
    "engine_overheat": inject_engine_overheat,
    "misfire_vibration": inject_misfire_vibration,
    "battery_failure": inject_battery_failure,
    "tyre_leak": inject_tyre_leak,
    "brake_overheat": inject_brake_overheat,
}


def inject_random_anomalies(df: pd.DataFrame, p_anomaly: float = 0.5) -> pd.DataFrame:
    """
    With probability p_anomaly, inject a random anomaly into this trip.
    Can also extend to multiple anomalies per trip if needed.
    """
    if random.random() < p_anomaly:
        anomaly_type = random.choice(ANOMALY_TYPES)
        df = ANOMALY_FUNCTIONS[anomaly_type](df)
    return df


# 4. GENERATE FULL DATASET


def generate_dataset(
    bike_config: VehicleConfig,
    car_config: VehicleConfig,
    p_anomaly: float = 0.5
) -> pd.DataFrame:

    all_dfs = []

    # Bikes
    for v in range(bike_config.n_vehicles):
        for trip in range(bike_config.n_trips_per_vehicle):
            df = simulate_normal_trip(bike_config, vehicle_id=f"bike_{v}", trip_id=trip)
            df = inject_random_anomalies(df, p_anomaly=p_anomaly)
            all_dfs.append(df)

    # Cars
    for v in range(car_config.n_vehicles):
        for trip in range(car_config.n_trips_per_vehicle):
            df = simulate_normal_trip(car_config, vehicle_id=f"car_{v}", trip_id=trip)
            df = inject_random_anomalies(df, p_anomaly=p_anomaly)
            all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    return full_df


# 5. PREP DATA FOR SEQUENCE MODELS (GAN/VAE)


class TimeSeriesDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        window_size: int,
        step: int = 1,
        only_normal: bool = True
    ):
        """
        Converts time-series per trip into sliding windows.
        """
        self.sequences = []

        for (_, trip_df) in df.groupby(["vehicle_id", "trip_id"]):
            if only_normal:
                trip_df = trip_df[trip_df["label"] == 0]

            values = trip_df[feature_cols].values.astype(np.float32)
            T = len(values)
            for start in range(0, T - window_size + 1, step):
                window = values[start:start+window_size]
                self.sequences.append(window)

        self.sequences = np.stack(self.sequences, axis=0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx])



# 6. VAE FOR TIME-SERIES GENERATION


class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()

        # Encoder (LSTM -> mean & logvar)
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

        # Decoder (latent -> LSTM -> output)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        # x: (B, T, F)
        _, (h_n, _) = self.encoder_lstm(x)  # h_n: (1, B, H)
        h = h_n.squeeze(0)                  # (B, H)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, seq_len):
        # z: (B, latent_dim)
        # repeat latent over time
        z_seq = z.unsqueeze(1).repeat(1, seq_len, 1)  # (B, T, latent_dim)
        out, _ = self.decoder_lstm(z_seq)
        x_recon = self.output_layer(out)
        return x_recon

    def forward(self, x):
        B, T, F = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, T)
        return x_recon, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    # KL divergence
    kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + 0.001 * kld


def train_vae(model, dataloader, n_epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(batch)
            loss = vae_loss_function(recon_batch, batch, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"[VAE] Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")


def sample_from_vae(model, n_samples: int, seq_len: int, device="cpu") -> np.ndarray:
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.mu_layer.out_features).to(device)
        x_gen = model.decode(z, seq_len)  # (B, T, F)
        return x_gen.cpu().numpy()


# 7. GAN FOR TIME-SERIES GENERATION (SIMPLE LSTM GAN)


class TimeSeriesGenerator(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (B, T, latent_dim)
        out, _ = self.lstm(z)
        x = self.fc(out)
        return x


class TimeSeriesDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # use last hidden state
        h_last = out[:, -1, :]
        logits = self.fc(h_last)
        prob = self.sigmoid(logits)
        return prob


def train_gan(
    G: TimeSeriesGenerator,
    D: TimeSeriesDiscriminator,
    dataloader: DataLoader,
    latent_dim: int,
    n_epochs: int = 10,
    lr: float = 1e-4,
    device: str = "cpu"
):
    G.to(device)
    D.to(device)

    criterion = nn.BCELoss()
    optimizerD = torch.optim.Adam(D.parameters(), lr=lr)
    optimizerG = torch.optim.Adam(G.parameters(), lr=lr)

    for epoch in range(n_epochs):
        total_d_loss = 0.0
        total_g_loss = 0.0

        for real_seq in dataloader:
            real_seq = real_seq.to(device)
            B, T, F = real_seq.shape

            
            # Train Discriminator
            
            optimizerD.zero_grad()
            # Real
            real_labels = torch.ones(B, 1).to(device)
            fake_labels = torch.zeros(B, 1).to(device)

            real_preds = D(real_seq)
            d_loss_real = criterion(real_preds, real_labels)

            # Fake
            z = torch.randn(B, T, latent_dim).to(device)
            fake_seq = G(z)
            fake_preds = D(fake_seq.detach())
            d_loss_fake = criterion(fake_preds, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizerD.step()

            
            # Train Generator
            
            optimizerG.zero_grad()
            z = torch.randn(B, T, latent_dim).to(device)
            fake_seq = G(z)
            preds = D(fake_seq)
            g_loss = criterion(preds, real_labels)  # want discriminator to think they're real
            g_loss.backward()
            optimizerG.step()

            total_d_loss += d_loss.item() * B
            total_g_loss += g_loss.item() * B

        avg_d_loss = total_d_loss / len(dataloader.dataset)
        avg_g_loss = total_g_loss / len(dataloader.dataset)
        print(f"[GAN] Epoch {epoch+1}/{n_epochs}, D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")


def sample_from_gan(G: TimeSeriesGenerator, n_samples: int, seq_len: int, latent_dim: int, device="cpu"):
    G.to(device)
    G.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, seq_len, latent_dim).to(device)
        samples = G(z)
        return samples.cpu().numpy()


# 8. MAIN EXAMPLE


if __name__ == "__main__":
    
    # 8.1 Generate synthetic dataset
    
    bike_cfg = VehicleConfig(
        vehicle_type="bike",
        n_vehicles=10,
        n_trips_per_vehicle=20,
        trip_length=300
    )

    car_cfg = VehicleConfig(
        vehicle_type="car",
        n_vehicles=10,
        n_trips_per_vehicle=20,
        trip_length=300
    )

    df = generate_dataset(bike_cfg, car_cfg, p_anomaly=0.6)
    print("Synthetic dataset shape:", df.shape)
    print(df.head())

    # Save CSV if you want
    df.to_csv("synthetic_vehicle_maintenance.csv", index=False)

    
    # 8.2 Prepare data for VAE/GAN
    
    feature_cols = [
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
        "brake_temp"
    ]

    window_size = 50
    dataset = TimeSeriesDataset(
        df,
        feature_cols=feature_cols,
        window_size=window_size,
        step=10,
        only_normal=True  # train generative models on normal patterns
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    
    # 8.3 Train VAE
    
    input_dim = len(feature_cols)
    vae = TimeSeriesVAE(input_dim=input_dim, hidden_dim=64, latent_dim=16)
    train_vae(vae, dataloader, n_epochs=5, lr=1e-3, device=device)

    # Sample from VAE
    vae_samples = sample_from_vae(vae, n_samples=10, seq_len=window_size, device=device)
    print("VAE generated samples shape:", vae_samples.shape)

    
    # 8.4 Train GAN
    
    latent_dim = 16
    G = TimeSeriesGenerator(latent_dim=latent_dim, hidden_dim=64, output_dim=input_dim)
    D = TimeSeriesDiscriminator(input_dim=input_dim, hidden_dim=64)

    train_gan(G, D, dataloader, latent_dim=latent_dim, n_epochs=5, lr=1e-4, device=device)

    gan_samples = sample_from_gan(G, n_samples=10, seq_len=window_size, latent_dim=latent_dim, device=device)
    print("GAN generated samples shape:", gan_samples.shape)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# 1. Load synthetic dataset
df = pd.read_csv("synthetic_vehicle_maintenance.csv")
print(df.head())
print(df["label"].value_counts(), "\n")

# Feature columns (same as generator)
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
    "brake_temp"
]

# 3. TimeSeriesDataset

class TimeSeriesDataset(Dataset):
    def __init__(self, df, feature_cols, window_size=50, step=10, only_normal=True):
        self.sequences = []

        # group by vehicle + trip to preserve temporal order
        for (_, trip_df) in df.groupby(["vehicle_id", "trip_id"]):
            if only_normal:
                trip_df = trip_df[trip_df["label"] == 0]

            if len(trip_df) < window_size:
                continue

            values = trip_df[feature_cols].values.astype(np.float32)
            T = len(values)
            for start in range(0, T - window_size + 1, step):
                window = values[start:start+window_size]
                self.sequences.append(window)

        self.sequences = np.stack(self.sequences, axis=0)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx])

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (B,T,D)
        T = x.size(1)
        return x + self.pe[:, :T, :]
    

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
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
        # x: (B,T,F)
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        memory = self.encoder(z)
        # teacher-forcing style: use same inputs as queries
        out = self.decoder(z, memory)
        recon = self.output_proj(out)
        return recon

def train_transformer_ae(model, dataloader, n_epochs=10, lr=1e-3, device="cpu"):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for epoch in range(n_epochs):
        total = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optim.zero_grad()
            recon = model(batch)
            loss = crit(recon, batch)
            loss.backward()
            optim.step()
            total += loss.item() * batch.size(0)
        print(f"[Transformer AE] Epoch {epoch+1}/{n_epochs}, Loss: {total/len(dataloader.dataset):.4f}")

def eval_reconstruction_model(model, dataloader, labels, name="Model", device="cpu"):
    model.to(device)
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            recon = model(batch)
            mse = torch.mean((recon - batch)**2, dim=(1,2))
            errors.extend(mse.cpu().numpy())
    errors = np.array(errors)
    normal_errors = errors[labels == 0]
    thr = np.percentile(normal_errors, 95)
    preds = (errors > thr).astype(int)
    print(f"\n=== {name} sequence-level report ===")
    print("Threshold:", thr)
    print(classification_report(labels, preds))
    print("Confusion matrix:\n", confusion_matrix(labels, preds))
    print("ROC-AUC (higher=more anomalous):", roc_auc_score(labels, errors))
    return errors

trans_ae = TransformerAutoencoder(input_dim)
train_transformer_ae(trans_ae, train_seq_loader, n_epochs=10, lr=1e-3, device=device)
trans_errors = eval_reconstruction_model(trans_ae, test_seq_loader, window_labels,
                                         name="Transformer AE", device=device)

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve

def plot_roc_curve(y_true, y_score, model_name="Model"):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7,6))
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
    plt.plot([0,1], [0,1], 'k--', label="Random Baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pr_curve(y_true, y_score, model_name="Model"):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    plt.figure(figsize=(7,6))
    plt.plot(recall, precision, label=model_name, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision–Recall Curve - {model_name}")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_error_distribution(errors, y_true, model_name="Model"):
    normal_errors = errors[y_true == 0]
    anomaly_errors = errors[y_true == 1]

    plt.figure(figsize=(7,6))
    sns.kdeplot(normal_errors, label="Normal", linewidth=2)
    sns.kdeplot(anomaly_errors, label="Anomaly", linewidth=2, color='red')
    plt.title(f"Reconstruction Error Distribution - {model_name}")
    plt.xlabel("Error")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()

# 1. Compute threshold
normal_trans = trans_errors[window_labels == 0]
thr_trans = np.percentile(normal_trans, 95)
print("Transformer threshold:", thr_trans)

# 2. Get binary predictions
pred_trans = (trans_errors > thr_trans).astype(int)

# 3. Generate plots
plot_roc_curve(window_labels, trans_errors, "Transformer AE")
plot_pr_curve(window_labels, trans_errors, "Transformer AE")
plot_error_distribution(trans_errors, window_labels, "Transformer AE")

