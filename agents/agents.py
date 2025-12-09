import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted
import time
import torch
import os
from collections import deque
import joblib
import numpy as np
import requests
import pandas as pd


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  
        self.register_buffer("pe", pe)

    def forward(self, x):
        T = x.size(1)
        return x + self.pe[:, :T, :]

class TransformerAutoencoder(torch.nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.decoder = torch.nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = torch.nn.Linear(d_model, input_dim)

    def forward(self, x):
        z = self.input_proj(x)
        z = self.pos_encoder(z)
        memory = self.encoder(z)
        # Target input for the decoder is usually the input sequence itself for an autoencoder
        out = self.decoder(z, memory) 
        recon = self.output_proj(out)
        return recon

genai.configure(api_key="")
model_geniai = genai.GenerativeModel("gemini-2.0-flash")
MODEL_DIR = "models"
MODEL_FILENAME = "transformer_ae_v1.pth"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
test_df=pd.read_csv('syn_data_100.csv')
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
scaler = joblib.load(SCALER_PATH)
vehicle_id=12
input_dim = checkpoint.get("input_dim")
d_model = checkpoint.get("d_model")
nhead = checkpoint.get("nhead")
num_layers = checkpoint.get("num_layers")
anomaly_threshold = checkpoint.get("anomaly_threshold")
feature_cols = checkpoint.get("feature_cols")
window_size = checkpoint.get("window_size")
model = TransformerAutoencoder(input_dim=input_dim,d_model=d_model,nhead=nhead,num_layers=num_layers,)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

def get_gemini_response(prompt):
    for attempt in range(3):
        try:
            time.sleep(1.2)  # slow down to avoid soft limits
            response = model_geniai.generate_content(prompt)
            return response.text.strip()
        except ResourceExhausted:
            print("Hit rate limit. Retrying...")
            time.sleep(3)
    return "LLM quota exceeded. Cannot generate report."

def anomaly_readings(raw_data_window):
    if len(raw_data_window) < window_size:
        print(f"Window size ({len(raw_data_window)}) is less than required ({window_size}). Skipping.")
        return 0, 0.0
    
    X = np.array(
        [[row[col] for col in feature_cols] for row in raw_data_window],
        dtype=np.float32
    ) 
    X_scaled = scaler.transform(X)
    x = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(0).to(model.parameters().__next__().device)

    with torch.no_grad():
        recon = model(x)
        mse = torch.mean((recon - x) ** 2, dim=(1, 2)).item()
    is_anomaly = int(mse > anomaly_threshold)
    
    print(f"Reconstruction Error (MSE): {mse:.6f}")
    print(f"Anomaly Detected: {'Yes' if is_anomaly else 'No'} (Threshold: {anomaly_threshold:.6f})")
    
    return is_anomaly, mse

def generate_anomaly_report(sensor_window, mse_value, threshold):
    prompt = f"""
    You are an industrial anomaly-analysis expert.

    Below is a sequence of {len(sensor_window)} sensor readings. 
    Each reading includes the following features: {feature_cols}.

    The autoencoder anomaly model produced:
    - Reconstruction Error (MSE): {mse_value:.4f}
    - Threshold: {threshold:.4f}

    Based on the readings and the error score, write a clear diagnostic
    explanation (50–100 words) describing:
    - What kind of abnormal behaviour might be happening
    - What component/system might be affected
    - How serious the situation is (minor / moderate / critical)
    - What immediate action should be taken

    Sensor Window Data:
    {sensor_window}

    Produce only the 50–100 word explanation. Do NOT include headings.
    """
    response = get_gemini_response(prompt)
    return response

def get_sensor_data(i):
    row=test_df.iloc[i]
    return row

def main():
    windowww=[]
    for i in range(window_size):
        data = get_sensor_data(i)
        values = data.to_dict()
        values["timeStamp"] = i
        values["vehicle_id"] = vehicle_id
        try:
                resp = requests.post(
                    "http://localhost:5000/api/v1/vehicles/telemetry/latest",
                    json=values,
                    timeout=5
                )
                print("POST status:", resp.status_code)
                print("Response:", resp.text)
        except Exception as e:
                print("Error while sending request:", e)
        time.sleep(1)
        windowww.append(data)

    if len(windowww) == window_size:
        is_ano, mse = anomaly_readings(windowww)
        if(is_ano):
            resp = generate_anomaly_report(windowww, mse, anomaly_threshold)
            ana={
                "vehicle_id":vehicle_id,
                "report":resp,
                "score":mse,
            }
            try:
                resp = requests.post(
                    "http://localhost:5000/api/v1/anomalies",
                    json=ana,
                    timeout=5
                )
                print("POST status:", resp.status_code)
                print("Response:", resp.text)
            except Exception as e:
                print("Error while sending request:", e)
        windowww = []
    
main()






