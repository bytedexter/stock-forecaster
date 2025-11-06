from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

@dataclass
class ModelAgentConfig:
    sequence_length: int = 60
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs: int = 10
    random_state: int = 42

class StockDataset(Dataset):
    """PyTorch Dataset for time series data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMWithAttention(nn.Module):
    """LSTM model with attention mechanism for stock price prediction"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(LSTMWithAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(32, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)
        
        # Dense layers
        out = self.relu(self.fc1(context_vector))
        out = self.dropout1(out)
        out = self.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.sigmoid(self.fc3(out))
        
        return out.squeeze(), attention_weights.squeeze()

class ModelAgent:
    def __init__(self, cfg: ModelAgentConfig):
        self.cfg = cfg
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_names = None
        
        # Set random seeds for reproducibility
        torch.manual_seed(cfg.random_state)
        np.random.seed(cfg.random_state)
    
    def _create_sequences(self, X: pd.DataFrame, y: pd.Series):
        """Convert tabular data to sequences for LSTM"""
        X_seq = []
        y_seq = []
        
        # Get unique symbols
        if 'symbol' in X.columns:
            X = X.drop(columns=['symbol'])
        
        # Convert to numpy
        X_np = X.values
        y_np = y.values
        
        # Create sequences
        for i in range(len(X_np) - self.cfg.sequence_length):
            X_seq.append(X_np[i:i + self.cfg.sequence_length])
            y_seq.append(y_np[i + self.cfg.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_calibrated(self, X: pd.DataFrame, y: pd.Series):
        """Train LSTM model with attention"""
        print(f"Training on device: {self.device}")
        
        # Store feature names
        self.feature_names = [col for col in X.columns if col != 'symbol']
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X[self.feature_names])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled_df, y)
        print(f"Created {len(X_seq)} sequences of length {self.cfg.sequence_length}")
        
        # Create dataset and dataloader
        dataset = StockDataset(X_seq, y_seq)
        train_loader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        # Initialize model
        input_size = len(self.feature_names)
        self.model = LSTMWithAttention(
            input_size=input_size,
            hidden_size=self.cfg.hidden_size,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)
        
        # Training loop
        self.model.train()
        for epoch in range(self.cfg.epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"Epoch [{epoch+1}/{self.cfg.epochs}], Loss: {avg_loss:.4f}")
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities for sequences"""
        self.model.eval()
        
        # Normalize features
        X_scaled = self.scaler.transform(X[self.feature_names])
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_names, index=X.index)
        
        # For prediction, we need to handle the sequence creation differently
        # We'll create sequences and return predictions for the last items
        all_preds = []
        
        X_np = X_scaled_df.values
        
        # Create sequences
        for i in range(len(X_np) - self.cfg.sequence_length + 1):
            seq = X_np[i:i + self.cfg.sequence_length]
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred, _ = self.model(seq_tensor)
                all_preds.append(pred.cpu().item())
        
        # Pad the beginning with predictions using partial sequences
        for i in range(self.cfg.sequence_length - 1):
            # Use what we have available
            seq_len = i + 1
            seq = X_np[:seq_len]
            # Pad with zeros
            padded_seq = np.zeros((self.cfg.sequence_length, X_np.shape[1]))
            padded_seq[-seq_len:] = seq
            seq_tensor = torch.FloatTensor(padded_seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                pred, _ = self.model(seq_tensor)
                all_preds.insert(0, pred.cpu().item())
        
        return np.array(all_preds)
    
    def get_attention_weights(self, X: pd.DataFrame, index: int = -1) -> np.ndarray:
        """Get attention weights for a specific sample"""
        self.model.eval()
        
        # Normalize features
        X_scaled = self.scaler.transform(X[self.feature_names])
        X_np = X_scaled
        
        # Get the sequence for the specified index
        if index < 0:
            index = len(X_np) + index
        
        start_idx = max(0, index - self.cfg.sequence_length + 1)
        seq = X_np[start_idx:index + 1]
        
        # Pad if necessary
        if len(seq) < self.cfg.sequence_length:
            padded_seq = np.zeros((self.cfg.sequence_length, X_np.shape[1]))
            padded_seq[-len(seq):] = seq
            seq = padded_seq
        
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            _, attention_weights = self.model(seq_tensor)
        
        return attention_weights.cpu().numpy()
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        proba = self.predict_proba(X)
        
        # Align lengths (since we might have padding issues)
        min_len = min(len(proba), len(y))
        proba = proba[:min_len]
        y_actual = y.iloc[:min_len]
        
        preds = (proba >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_actual, preds)),
            "f1_up": float(f1_score(y_actual, preds, zero_division=0))
        }
