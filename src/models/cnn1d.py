"""
1D Convolutional Neural Network (1D-CNN) for Physiological Stress Detection
COEN 6322: AI for Wearable Health Applications, Fall 2026
Author: Open-Source Contributor

Architecture:
Conv1D -> ReLU -> Conv1D -> ReLU -> Global Average Pooling -> Dense -> Binary Output (REST vs STRESS).
Supports CPU with optional GPU execution.
Configurable hyperparameters, early stopping, and seed reproducibility.
"""

import os
import random
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int = 42):
    """Sets fixed seeds across libraries for strict reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PhysiologicalWindowDataset(Dataset):
    """PyTorch Dataset for (Channels, TimeSteps) physiological tensors."""
    def __init__(self, tensors: np.ndarray, labels: np.ndarray):
        self.tensors = torch.tensor(tensors, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


class Small1DCNN(nn.Module):
    """
    Lightweight 1D-CNN architecture strictly matching the proposal specification:
    Conv1D -> ReLU -> Conv1D -> ReLU -> Global Average Pooling -> Dense -> Binary Output.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 2, dropout: float = 0.2):
        super(Small1DCNN, self).__init__()
        self.features = nn.Sequential(
            # First Conv Block
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            
            # Second Conv Block
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            
            # Global Average Pooling over time dimension
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, in_channels, time_steps)
        feat = self.features(x)
        feat = feat.view(feat.size(0), -1)
        out = self.classifier(feat)
        return out


class CNNTrainer:
    """Handles reproducible training, early stopping, and inference for Small1DCNN."""
    
    def __init__(self, in_channels: int = 1, lr: float = 1e-3, batch_size: int = 32,
                 max_epochs: int = 40, patience: int = 7, device: Optional[str] = None):
        self.in_channels = in_channels
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Small1DCNN(in_channels=in_channels, num_classes=2).to(self.device)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """
        Trains model with validation-based early stopping.
        X_train shape: (N_train, in_channels, time_steps)
        """
        train_ds = PhysiologicalWindowDataset(X_train, y_train)
        val_ds = PhysiologicalWindowDataset(X_val, y_val)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Handle class imbalance via class weights
        class_counts = np.bincount(y_train, minlength=2)
        total_samples = len(y_train)
        weights = total_samples / (2.0 * np.maximum(class_counts, 1))
        class_weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        best_val_loss = float("inf")
        best_weights = None
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": [], "val_acc": []}

        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_y)

            train_loss = total_train_loss / len(train_ds)

            # Validation
            self.model.eval()
            total_val_loss = 0.0
            correct = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    total_val_loss += loss.item() * len(batch_y)
                    preds = torch.argmax(outputs, dim=1)
                    correct += (preds == batch_y).sum().item()

            val_loss = total_val_loss / len(val_ds) if len(val_ds) > 0 else 0.0
            val_acc = correct / len(val_ds) if len(val_ds) > 0 else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = self.model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    break

        if best_weights:
            self.model.load_state_dict(best_weights)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions (0 or 1)."""
        self.model.eval()
        ds = PhysiologicalWindowDataset(X, np.zeros(len(X), dtype=np.int64))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
        return np.array(all_preds, dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns softmax class probabilities shape (N, 2)."""
        self.model.eval()
        ds = PhysiologicalWindowDataset(X, np.zeros(len(X), dtype=np.int64))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=False)
        all_probs = []
        softmax = nn.Softmax(dim=1)
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                probs = softmax(outputs).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs) if len(all_probs) > 0 else np.empty((0, 2))
