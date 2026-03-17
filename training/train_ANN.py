"""
Train MLP (Multi-Layer Perceptron) with Optuna Hyperparameter Optimization.

This script trains an MLP neural network for AI code detection using
Optuna for automated hyperparameter tuning.

Features:
- Configurable MLP architecture (2-4 hidden layers)
- Bayesian optimization (TPE) for hyperparameter search
- Median pruning for early stopping of unpromising trials
- Cross-validation support
- Model saving and loading
- Visualization of optimization history

Usage:
    # Run optimization with 50 trials
    python train_mlp.py --features features/features_3000samples_3models.csv --trials 50
    
    # Quick test with 10 trials
    python train_mlp.py --features features/features_*.csv --trials 10 --timeout 600
    
    # Use specific hyperparameters (no optimization)
    python train_mlp.py --features features/features_*.csv --no-optimize
"""

import os
import sys
import argparse
import pickle
import json
import warnings
from pathlib import Path
from datetime import datetime
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             precision_recall_curve, average_precision_score)

# Optuna for hyperparameter optimization
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

warnings.filterwarnings('ignore')

# Configuration
OUTPUT_DIR = Path("dl_models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42
LABEL_SMOOTHING = 0.1  # Default label smoothing value (0.0 = no smoothing)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# MLP Model Definition
class CodeDetectorMLP(nn.Module):
    """
    Multi-Layer Perceptron for AI Code Detection.
    
    Architecture:
    - Input BatchNorm (optional)
    - N hidden layers with configurable sizes
    - Each hidden layer: Linear -> BatchNorm (optional) -> Activation -> Dropout
    - Output layer: Linear -> Softmax (via CrossEntropyLoss)
    
    Args:
        input_size: Number of input features
        hidden_sizes: List of hidden layer sizes, e.g., [256, 128, 64]
        num_classes: Number of output classes (2 for binary)
        dropout: Dropout rate
        activation: Activation function name ('relu', 'leaky_relu', 'gelu', 'silu')
        use_batch_norm: Whether to use batch normalization
    """
    
    ACTIVATIONS = {
        'relu': nn.ReLU,
        'leaky_relu': nn.LeakyReLU,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'elu': nn.ELU,
    }
    
    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int = 2,
                 dropout: float = 0.3, activation: str = 'relu', 
                 use_batch_norm: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm
        
        # Get activation function
        activation_fn = self.ACTIVATIONS.get(activation.lower(), nn.ReLU)
        
        # Input batch normalization
        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)
        else:
            self.input_bn = nn.Identity()
        
        # Build hidden layers
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            # Linear layer
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            # Activation
            layers.append(activation_fn())
            
            # Dropout (less dropout in later layers)
            drop_rate = dropout * (1 - 0.2 * i / len(hidden_sizes))
            layers.append(nn.Dropout(drop_rate))
            
            prev_size = hidden_size
        
        self.hidden = nn.Sequential(*layers)
        
        # Output layer
        self.output = nn.Linear(prev_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_bn(x)
        x = self.hidden(x)
        x = self.output(x)
        return x
    
    def predict_proba(self, x):
        """Return probability predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs
    
    def get_config(self):
        """Return model configuration."""
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'num_classes': self.num_classes,
            'dropout': self.dropout_rate,
            'activation': self.activation_name,
            'use_batch_norm': self.use_batch_norm,
        }

# Training Utilities
class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, 
                 mode: str = 'max', restore_best: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best = restore_best
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif self._is_improvement(score):
            self.best_score = score
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'max':
            return score > self.best_score + self.min_delta
        else:
            return score < self.best_score - self.min_delta
    
    def restore_best_weights(self, model):
        if self.restore_best and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X_batch.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * X_batch.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1 (AI)
            all_labels.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'roc_auc': roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.5,
    }
    
    return metrics, all_preds, all_probs, all_labels

# Data Loading and Preprocessing
def load_and_preprocess_data(feature_paths: list, include_language: bool = True):
    """Load and preprocess feature data."""
    
    # Load all CSV files
    dfs = []
    for path in feature_paths:
        if '*' in path:
            files = glob(path)
            for f in files:
                dfs.append(pd.read_csv(f))
        else:
            dfs.append(pd.read_csv(path))
    
    if not dfs:
        raise ValueError("No feature files found!")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} samples from {len(dfs)} file(s)")
    
    # Get feature columns (exclude metadata)
    exclude_cols = ['sample_id', 'language', 'label', 'source']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Include language_encoded if requested
    if not include_language and 'language_encoded' in feature_cols:
        feature_cols.remove('language_encoded')
    
    print(f"Using {len(feature_cols)} features")
    
    # Extract features and labels
    X = df[feature_cols].values
    y = df['label'].values
    
    # Handle NaN values
    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"Warning: Found {nan_mask.sum()} NaN values, replacing with column median")
        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            X[nan_mask[:, i], i] = col_medians[i]
    
    # Handle infinite values
    X = np.clip(X, -1e10, 1e10)
    
    return X, y, feature_cols


def prepare_data_loaders(X, y, batch_size: int = 32, val_size: float = 0.15, 
                         test_size: float = 0.15, random_state: int = 42):
    """Prepare train, validation, and test data loaders."""
    
    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval, random_state=random_state
    )
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_val_t = torch.FloatTensor(X_val)
    X_test_t = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t = torch.LongTensor(y_val)
    y_test_t = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler

# Optuna Hyperparameter Optimization
def create_objective(X_trainval, y_trainval, input_size, device, 
                     max_epochs=100, patience=10):
    """Create Optuna objective function with k-fold cross-validation."""
    
    def objective(trial):
        # Sample hyperparameters
        # Architecture
        n_layers = trial.suggest_int('n_layers', 2, 5)
        hidden_sizes = []
        
        for i in range(n_layers):
            if i == 0:
                h_size = trial.suggest_int(f'hidden_{i}', 128, 1024)
            elif i == 1:
                h_size = trial.suggest_int(f'hidden_{i}', 64, 512)
            elif i == 2:
                h_size = trial.suggest_int(f'hidden_{i}', 32, 256)
            elif i == 3:
                h_size = trial.suggest_int(f'hidden_{i}', 16, 64)
            else:
                h_size = trial.suggest_int(f'hidden_{i}', 8, 32)
            hidden_sizes.append(h_size)
        
        dropout = trial.suggest_float('dropout', 0.1, 0.5)
        activation = trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'gelu', 'silu'])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # Training
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'AdamW', 'RMSprop'])
        
        # Regularization
        label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.2)
        
        # Cross-validation settings (tunable number of folds)
        n_folds = trial.suggest_int('cv_folds', 3, 6)
        
        # Perform k-fold cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
        fold_scores = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval)):
            X_train, X_val = X_trainval[train_idx], X_trainval[val_idx]
            y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Prepare data loaders
            train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.LongTensor(y_train))
            val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.LongTensor(y_val))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Build model
            model = CodeDetectorMLP(
                input_size=input_size,
                hidden_sizes=hidden_sizes,
                num_classes=2,
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm
            ).to(device)
            
            # Optimizer
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            elif optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
            
            # Loss function with tunable label smoothing
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
            # Learning rate scheduler
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5, patience=5, verbose=False
            )
            
            # Early stopping
            early_stopping = EarlyStopping(patience=patience, mode='max')
            
            best_val_f1 = 0
            
            # Training loop
            for epoch in range(max_epochs):
                train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
                val_f1 = val_metrics['f1']
                
                scheduler.step(val_f1)
                
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                
                early_stopping(val_f1, model)
                if early_stopping.early_stop:
                    break
            
            fold_scores.append(best_val_f1)
            
            # Report intermediate result (average so far) for pruning
            avg_score = np.mean(fold_scores)
            trial.report(avg_score, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Return mean F1 across all folds
        mean_f1 = np.mean(fold_scores)
        return mean_f1
    
    return objective


def run_optimization(X, y, feature_names, n_trials=50, timeout=3600, max_epochs=100, device=DEVICE):
    """Run Optuna hyperparameter optimization with k-fold CV."""
    
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required for optimization. Install with: pip install optuna")
    print("OPTUNA HYPERPARAMETER OPTIMIZATION (with K-Fold CV)")
    
    # Split data: trainval vs test (CV is done internally in objective)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=RANDOM_SEED
    )
    
    print(f"Data: TrainVal={len(X_trainval)} (CV inside), Test={len(X_test)}")
    
    input_size = X_trainval.shape[1]
    
    # Create study
    study = optuna.create_study(
        direction='maximize',
        pruner=MedianPruner(n_warmup_steps=2, n_startup_trials=3),
        sampler=TPESampler(seed=RANDOM_SEED)
    )
    
    # Create objective with internal k-fold CV
    objective = create_objective(
        X_trainval, y_trainval, 
        input_size, device, max_epochs=max_epochs, patience=10
    )
    
    print(f"\nRunning optimization with {n_trials} trials (timeout: {timeout}s, max_epochs: {max_epochs})...")
    print(f"Device: {device}")
    
    # Run optimization
    study.optimize(
        objective, 
        n_trials=n_trials, 
        timeout=timeout,
        show_progress_bar=True
    )
    
    # Results
    print("OPTIMIZATION RESULTS")
    
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best Validation F1: {study.best_value:.4f}")
    print(f"\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Train final model with best hyperparameters
    print("TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    
    best_params = study.best_params
    
    # Build hidden sizes from best params
    n_layers = best_params['n_layers']
    hidden_sizes = [best_params[f'hidden_{i}'] for i in range(n_layers)]
    
    # Standardize data for final training
    scaler = StandardScaler()
    X_trainval_scaled = scaler.fit_transform(X_trainval)
    X_test_scaled = scaler.transform(X_test)
    
    final_model = CodeDetectorMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=2,
        dropout=best_params['dropout'],
        activation=best_params['activation'],
        use_batch_norm=best_params['use_batch_norm']
    ).to(device)
    
    # Training setup
    batch_size = best_params['batch_size']
    trainval_dataset = TensorDataset(torch.FloatTensor(X_trainval_scaled), torch.LongTensor(y_trainval))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.LongTensor(y_test))
    trainval_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if best_params['optimizer'] == 'Adam':
        optimizer = optim.Adam(final_model.parameters(), lr=best_params['lr'], 
                               weight_decay=best_params['weight_decay'])
    elif best_params['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(final_model.parameters(), lr=best_params['lr'], 
                                weight_decay=best_params['weight_decay'])
    else:
        optimizer = optim.RMSprop(final_model.parameters(), lr=best_params['lr'], 
                                  weight_decay=best_params['weight_decay'])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=best_params.get('label_smoothing', LABEL_SMOOTHING))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Track training history for loss curves
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    # Train final model
    print("\nTraining final model...")
    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(final_model, trainval_loader, optimizer, criterion, device)
        
        # Evaluate on test set for monitoring (using test as proxy for validation in final training)
        test_metrics_epoch, _, _, _ = evaluate(final_model, test_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(test_metrics_epoch['loss'])
        history['val_f1'].append(test_metrics_epoch['f1'])
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
        scheduler.step(train_loss)
    
    # Evaluate on test set
    print("FINAL TEST RESULTS")
    
    test_metrics, test_preds, test_probs, test_labels = evaluate(
        final_model, test_loader, criterion, device
    )
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, target_names=['Human', 'AI']))
    
    return final_model, scaler, feature_names, study, test_metrics, best_params, test_loader, history


def train_default_mlp(X, y, feature_names, hidden_sizes=[256, 128, 64], 
                      dropout=0.3, lr=0.001, batch_size=32, max_epochs=100,
                      device=DEVICE):
    """Train MLP with default/specified hyperparameters."""
    print("TRAINING MLP WITH DEFAULT HYPERPARAMETERS")
    
    # Prepare data
    train_loader, val_loader, test_loader, scaler = prepare_data_loaders(
        X, y, batch_size=batch_size
    )
    
    input_size = X.shape[1]
    
    print(f"\nModel Configuration:")
    print(f"  Input size: {input_size}")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Dropout: {dropout}")
    print(f"  Learning rate: {lr}")
    print(f"  Batch size: {batch_size}")
    
    # Build model
    model = CodeDetectorMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        num_classes=2,
        dropout=dropout,
        activation='relu',
        use_batch_norm=True
    ).to(device)
    
    # Training setup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15, mode='max')
    
    print(f"\nTraining on {device}...")
    
    history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    for epoch in range(max_epochs):
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Scheduler
        scheduler.step(val_metrics['f1'])
        
        # Early stopping
        early_stopping(val_metrics['f1'], model)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, Val F1={val_metrics['f1']:.4f}")
        
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Restore best weights
    early_stopping.restore_best_weights(model)
    
    # Evaluate on test set
    print("TEST RESULTS")
    
    test_metrics, test_preds, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device
    )
    
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(test_labels, test_preds)
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    return model, scaler, feature_names, history, test_metrics, test_loader

# Model Saving and Loading
def save_model(model, scaler, feature_names, metrics, hyperparams=None, 
               output_path=None, study=None):
    """Save trained model and metadata."""
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"mlp_model_{timestamp}.pkl"
    else:
        output_path = Path(output_path)
    
    # Prepare model data
    model_data = {
        'model_name': 'MLP',
        'model_state_dict': model.state_dict(),
        'model_config': model.get_config(),
        'scaler': scaler,
        'feature_names': feature_names,
        'metrics': metrics,
        'hyperparameters': hyperparams or model.get_config(),
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save model
    torch.save(model_data, output_path)
    print(f"\nModel saved to: {output_path}")
    
    # Save optimization history if available
    if study is not None:
        history_path = output_path.with_suffix('.json')
        history_data = {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'n_trials': len(study.trials),
        }
        with open(history_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        print(f"Optimization history saved to: {history_path}")
    
    return output_path


def load_model(model_path, device=DEVICE):
    """Load a trained model."""
    
    model_data = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    config = model_data['model_config']
    model = CodeDetectorMLP(
        input_size=config['input_size'],
        hidden_sizes=config['hidden_sizes'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        activation=config['activation'],
        use_batch_norm=config['use_batch_norm']
    ).to(device)
    
    model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    
    return model, model_data['scaler'], model_data['feature_names'], model_data

# Cross-Validation
def cross_validate_mlp(X, y, feature_names, hyperparams=None, n_folds=5, 
                       max_epochs=100, device=DEVICE, verbose=True):
    """
    Perform K-Fold Cross-Validation for more robust model evaluation.
    
    Args:
        X: Feature matrix
        y: Labels
        feature_names: List of feature names
        hyperparams: Dictionary of hyperparameters (if None, uses defaults)
        n_folds: Number of folds for cross-validation
        max_epochs: Maximum training epochs per fold
        device: Device to train on
        verbose: Print progress
    
    Returns:
        cv_results: Dictionary with per-fold and aggregate metrics
    """
    print(f"K-FOLD CROSS-VALIDATION (K={n_folds})")
    
    # Default hyperparameters
    if hyperparams is None:
        hyperparams = {
            'hidden_sizes': [256, 128, 64],
            'dropout': 0.3,
            'activation': 'relu',
            'use_batch_norm': True,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'batch_size': 32,
            'optimizer': 'AdamW'
        }
    
    # Extract hyperparameters
    if 'hidden_sizes' not in hyperparams:
        # Build hidden sizes from individual keys (Optuna format)
        n_layers = hyperparams.get('n_layers', 3)
        hidden_sizes = [hyperparams.get(f'hidden_{i}', 128) for i in range(n_layers)]
    else:
        hidden_sizes = hyperparams['hidden_sizes']
    
    dropout = hyperparams.get('dropout', 0.3)
    activation = hyperparams.get('activation', 'relu')
    use_batch_norm = hyperparams.get('use_batch_norm', True)
    lr = hyperparams.get('lr', 0.001)
    weight_decay = hyperparams.get('weight_decay', 1e-5)
    batch_size = hyperparams.get('batch_size', 32)
    optimizer_name = hyperparams.get('optimizer', 'AdamW')
    
    input_size = X.shape[1]
    
    print(f"\nModel Configuration:")
    print(f"  Hidden sizes: {hidden_sizes}")
    print(f"  Dropout: {dropout}")
    print(f"  Activation: {activation}")
    print(f"  Batch Norm: {use_batch_norm}")
    print(f"  Learning Rate: {lr}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Optimizer: {optimizer_name}")
    
    # Initialize stratified k-fold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    
    # Storage for results
    fold_metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    fold_histories = []
    
    print(f"\nTraining on {device}...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        if verbose:
            print(f"\nFold {fold + 1}/{n_folds}")
        
        # Split data
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Standardize features (fit on train, transform both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train_scaled)
        X_val_t = torch.FloatTensor(X_val_scaled)
        y_train_t = torch.LongTensor(y_train)
        y_val_t = torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        val_dataset = TensorDataset(X_val_t, y_val_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Build model
        model = CodeDetectorMLP(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            num_classes=2,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm
        ).to(device)
        
        # Optimizer
        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        early_stopping = EarlyStopping(patience=15, mode='max')
        
        # Training history for this fold
        fold_history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
        
        # Training loop
        for epoch in range(max_epochs):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
            
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_metrics['loss'])
            fold_history['val_f1'].append(val_metrics['f1'])
            
            scheduler.step(val_metrics['f1'])
            early_stopping(val_metrics['f1'], model)
            
            if early_stopping.early_stop:
                if verbose:
                    print(f"   Early stopping at epoch {epoch + 1}")
                break
        
        # Restore best weights
        early_stopping.restore_best_weights(model)
        fold_histories.append(fold_history)
        
        # Final evaluation on validation fold
        final_metrics, preds, probs, labels = evaluate(model, val_loader, criterion, device)
        
        # Store predictions for aggregate analysis
        all_predictions.extend(preds)
        all_probabilities.extend(probs)
        all_labels.extend(labels)
        
        # Store fold metrics
        for key in fold_metrics:
            fold_metrics[key].append(final_metrics[key])
        
        if verbose:
            print(f"   Epochs: {len(fold_history['train_loss'])}, "
                  f"Acc: {final_metrics['accuracy']:.4f}, "
                  f"F1: {final_metrics['f1']:.4f}, "
                  f"ROC-AUC: {final_metrics['roc_auc']:.4f}")
    
    # Calculate aggregate statistics
    print("CROSS-VALIDATION RESULTS")
    
    cv_results = {
        'n_folds': n_folds,
        'per_fold': fold_metrics,
        'mean': {},
        'std': {},
        'all_predictions': all_predictions,
        'all_probabilities': all_probabilities,
        'all_labels': all_labels,
        'fold_histories': fold_histories,
        'hyperparams': hyperparams
    }
    
    print(f"\n{'Metric':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    
    for metric in fold_metrics:
        values = np.array(fold_metrics[metric])
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        
        cv_results['mean'][metric] = mean_val
        cv_results['std'][metric] = std_val
        
        print(f"{metric.upper():<12} {mean_val:>10.4f} {std_val:>10.4f} {min_val:>10.4f} {max_val:>10.4f}")
    
    # Overall confusion matrix (aggregated)
    print("\nAggregated Confusion Matrix (across all folds):")
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    print(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    
    cv_results['confusion_matrix'] = cm
    
    return cv_results


def plot_cv_results(cv_results, save_dir=None):
    """Plot cross-validation results visualization."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    n_folds = cv_results['n_folds']
    fold_metrics = cv_results['per_fold']
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Metrics per fold (bar chart)
    ax1 = axes[0, 0]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(n_folds)
    width = 0.15
    
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        values = fold_metrics[metric]
        ax1.bar(x + i * width, values, width, label=metric.upper(), color=color, alpha=0.8)
    
    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Score', fontsize=12)
    ax1.set_title('Metrics by Fold', fontsize=14)
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels([f'Fold {i+1}' for i in range(n_folds)])
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_ylim([0.7, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Box plot of metrics
    ax2 = axes[0, 1]
    metric_data = [fold_metrics[m] for m in metrics]
    bp = ax2.boxplot(metric_data, labels=[m.upper() for m in metrics], patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Metrics Distribution Across Folds', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0.7, 1.05])
    
    # Plot 3: Training curves across folds
    ax3 = axes[1, 0]
    fold_histories = cv_results['fold_histories']
    cmap = plt.cm.get_cmap('tab10')
    
    for fold_idx, history in enumerate(fold_histories):
        epochs = range(1, len(history['train_loss']) + 1)
        ax3.plot(epochs, history['val_f1'], '-', color=cmap(fold_idx), 
                 linewidth=1.5, alpha=0.8, label=f'Fold {fold_idx+1}')
    
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation F1', fontsize=12)
    ax3.set_title('Validation F1 Across Training (per Fold)', fontsize=14)
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    means = [cv_results['mean'][m] for m in metrics]
    stds = [cv_results['std'][m] for m in metrics]
    
    bars = ax4.bar(range(len(metrics)), means, color=colors, alpha=0.8, 
                   yerr=stds, capsize=5, error_kw={'elinewidth': 2})
    ax4.set_xticks(range(len(metrics)))
    ax4.set_xticklabels([m.upper() for m in metrics], fontsize=11)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('Mean ± Std Across Folds', fontsize=14)
    ax4.set_ylim([0.7, 1.1])
    
    for bar, mean, std in zip(bars, means, stds):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
                 f'{mean:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'cross_validation_results.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Cross-validation plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()


def save_cv_results(cv_results, output_dir):
    """Save cross-validation results to files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics summary
    summary = {
        'n_folds': cv_results['n_folds'],
        'timestamp': datetime.now().isoformat(),
        'mean_metrics': cv_results['mean'],
        'std_metrics': cv_results['std'],
        'per_fold_metrics': cv_results['per_fold'],
        'hyperparams': cv_results['hyperparams']
    }
    
    json_path = output_dir / 'cv_results.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"CV results saved to: {json_path}")
    
    # Save detailed report
    report_path = output_dir / 'cv_report.txt'
    with open(report_path, 'w') as f:
        f.write("CROSS-VALIDATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Number of Folds: {cv_results['n_folds']}\n\n")
        
        f.write("AGGREGATE METRICS (Mean ± Std)\n")
        for metric in cv_results['mean']:
            mean = cv_results['mean'][metric]
            std = cv_results['std'][metric]
            f.write(f"  {metric.upper():<12}: {mean:.4f} ± {std:.4f}\n")
        
        f.write("\n\nPER-FOLD METRICS\n")
        for fold in range(cv_results['n_folds']):
            f.write(f"\nFold {fold + 1}:\n")
            for metric in cv_results['per_fold']:
                f.write(f"  {metric.upper():<12}: {cv_results['per_fold'][metric][fold]:.4f}\n")
        
        f.write("\n\nCONFUSION MATRIX (Aggregated)\n")
        cm = cv_results['confusion_matrix']
        f.write(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],\n")
        f.write(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]\n")
        
        f.write("\n\nCLASSIFICATION REPORT (Aggregated)\n")
        f.write(classification_report(
            cv_results['all_labels'], 
            cv_results['all_predictions'],
            target_names=['Human', 'AI'],
            digits=4
        ))
    
    print(f"CV report saved to: {report_path}")
    
    return json_path, report_path


# Visualization
def plot_optimization_history(study, save_path=None):
    """Plot Optuna optimization history."""
    
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Optimization history
    ax1 = axes[0]
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials]
    ax1.plot(range(len(values)), values, 'b-', alpha=0.5, label='Trial F1')
    ax1.plot(range(len(values)), pd.Series(values).cummax(), 'r-', linewidth=2, label='Best F1')
    ax1.set_xlabel('Trial')
    ax1.set_ylabel('Validation F1')
    ax1.set_title('Optimization History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Hyperparameter importance
    ax2 = axes[1]
    try:
        importance = optuna.importance.get_param_importances(study)
        params = list(importance.keys())[:10]
        values = [importance[p] for p in params]
        ax2.barh(params, values, color='steelblue')
        ax2.set_xlabel('Importance')
        ax2.set_title('Hyperparameter Importance')
    except Exception as e:
        ax2.text(0.5, 0.5, f'Could not compute importance:\n{e}', 
                 ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Optimization plot saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_validation_loss(history, save_path=None, title="Training and Validation Loss"):
    """Plot training and validation loss curves.
    
    Args:
        history: Dictionary containing 'train_loss' and 'val_loss' lists
        save_path: Optional path to save the plot
        title: Title for the plot
    """
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available for plotting")
        return
    
    train_loss = history.get('train_loss', [])
    val_loss = history.get('val_loss', [])
    
    if not train_loss or not val_loss:
        print("No loss history available to plot")
        return
    
    epochs = range(1, len(train_loss) + 1)
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training and Validation Loss
    ax1 = axes[0]
    ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o', markersize=3, alpha=0.8)
    ax1.plot(epochs, val_loss, 'r-', linewidth=2, label='Validation Loss', marker='s', markersize=3, alpha=0.8)
    
    # Mark the epoch with minimum validation loss
    min_val_loss_idx = np.argmin(val_loss)
    min_val_loss = val_loss[min_val_loss_idx]
    ax1.axvline(x=min_val_loss_idx + 1, color='green', linestyle='--', alpha=0.7, 
                label=f'Best Epoch: {min_val_loss_idx + 1}')
    ax1.scatter([min_val_loss_idx + 1], [min_val_loss], color='green', s=100, zorder=5, 
                edgecolors='black', linewidths=1.5)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([1, len(epochs)])
    
    # Plot 2: Loss Difference (Overfitting indicator)
    ax2 = axes[1]
    loss_diff = np.array(val_loss) - np.array(train_loss)
    colors = ['red' if d > 0 else 'green' for d in loss_diff]
    ax2.bar(epochs, loss_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss - Training Loss', fontsize=12)
    ax2.set_title('Overfitting Indicator\n(Positive = Overfitting)', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_xlim([0.5, len(epochs) + 0.5])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training/validation loss curve saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print summary statistics
    print(f"\n--- Loss Curve Summary ---")
    print(f"Initial Training Loss: {train_loss[0]:.4f}")
    print(f"Final Training Loss: {train_loss[-1]:.4f}")
    print(f"Initial Validation Loss: {val_loss[0]:.4f}")
    print(f"Final Validation Loss: {val_loss[-1]:.4f}")
    print(f"Best Validation Loss: {min_val_loss:.4f} (Epoch {min_val_loss_idx + 1})")
    print(f"Total Epochs: {len(epochs)}")


# Evaluation Visualizations
def plot_confusion_matrix(y_true, y_pred, save_path=None, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    if not PLOTTING_AVAILABLE:
        return None
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Raw counts
    ax1 = axes[0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
                annot_kws={'size': 16})
    ax1.set_xlabel('Predicted Label', fontsize=12)
    ax1.set_ylabel('True Label', fontsize=12)
    ax1.set_title(f'{title}\n(Raw Counts)', fontsize=14)
    
    # Normalized
    ax2 = axes[1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', ax=ax2,
                xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'],
                annot_kws={'size': 16})
    ax2.set_xlabel('Predicted Label', fontsize=12)
    ax2.set_ylabel('True Label', fontsize=12)
    ax2.set_title(f'{title}\n(Normalized)', fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()
    return cm


def plot_roc_curve_eval(y_true, y_probs, save_path=None, title="ROC Curve"):
    """Plot and save ROC curve."""
    if not PLOTTING_AVAILABLE:
        return None, None
    
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='#3498db', lw=2, 
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', 
            label='Random Classifier')
    
    # Find optimal threshold
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], marker='o', s=100, 
               color='#e74c3c', zorder=5,
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.close()
    return roc_auc, optimal_threshold


def plot_precision_recall_curve_eval(y_true, y_probs, save_path=None, title="Precision-Recall Curve"):
    """Plot and save Precision-Recall curve."""
    if not PLOTTING_AVAILABLE:
        return None
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='#27ae60', lw=2,
            label=f'PR Curve (AP = {avg_precision:.4f})')
    
    baseline = np.sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='#95a5a6', lw=2, linestyle='--',
               label=f'Baseline = {baseline:.2f}')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Precision-Recall curve saved to: {save_path}")
    
    plt.close()
    return avg_precision


def plot_metrics_summary(metrics, save_path=None, title="Model Performance Metrics"):
    """Plot and save metrics summary."""
    if not PLOTTING_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    values = [metrics.get(k, 0) for k in metric_keys]
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    
    bars = ax.bar(range(len(metric_names)), values, color=colors)
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Metrics summary saved to: {save_path}")
    
    plt.close()


def run_evaluation(model, test_loader, criterion, device, output_dir, model_name="MLP"):
    """Run full evaluation and generate all visualizations."""
    print("GENERATING EVALUATION VISUALIZATIONS")
    
    # Get all predictions
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.numpy())
    
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_probs = np.array(all_probs)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5,
    }
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix
    cm = plot_confusion_matrix(
        y_true, y_pred,
        save_path=output_dir / 'confusion_matrix.png',
        title=f"{model_name} - Confusion Matrix"
    )
    
    # 2. ROC Curve
    roc_auc, opt_threshold = plot_roc_curve_eval(
        y_true, y_probs,
        save_path=output_dir / 'roc_curve.png',
        title=f"{model_name} - ROC Curve"
    )
    
    # 3. Precision-Recall Curve
    avg_precision = plot_precision_recall_curve_eval(
        y_true, y_probs,
        save_path=output_dir / 'precision_recall_curve.png',
        title=f"{model_name} - Precision-Recall Curve"
    )
    
    # 4. Metrics Summary
    plot_metrics_summary(
        metrics,
        save_path=output_dir / 'metrics_summary.png',
        title=f"{model_name} - Performance Metrics"
    )
    
    # 5. Classification Report (text)
    report = classification_report(y_true, y_pred, target_names=['Human', 'AI'], digits=4)
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"CLASSIFICATION REPORT - {model_name}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix:\n")
        if cm is not None:
            f.write(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],\n")
            f.write(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]\n")
    print(f"Classification report saved to: {report_path}")
    
    # 6. Metrics JSON
    metrics_json = {
        'timestamp': datetime.now().isoformat(),
        'model': model_name,
        'metrics': metrics,
        'optimal_threshold': float(opt_threshold) if opt_threshold else 0.5,
        'test_samples': len(y_true)
    }
    json_path = output_dir / 'metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics JSON saved to: {json_path}")
    
    print(f"\nAll evaluation results saved to: {output_dir}")
    
    return metrics, cm, report


# Main
def main():
    parser = argparse.ArgumentParser(
        description='Train MLP with Optuna Hyperparameter Optimization'
    )
    
    # Data
    parser.add_argument('--features', type=str, required=True,
                        help='Path to feature CSV file(s), supports wildcards')
    
    # Optimization
    parser.add_argument('--trials', type=int, default=50,
                        help='Number of Optuna trials (default: 50)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Optimization timeout in seconds (default: 3600)')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip optimization, use default hyperparameters')
    
    # Default hyperparameters (used with --no-optimize)
    parser.add_argument('--hidden-sizes', type=str, default='256,128,64',
                        help='Hidden layer sizes (default: 256,128,64)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size (default: 32)')

    # Both
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs (default: 100)')
    
    # Cross-validation
    parser.add_argument('--cv', action='store_true',
                        help='Run K-Fold cross-validation for robust evaluation')
    parser.add_argument('--cv-folds', type=int, default=5,
                        help='Number of folds for cross-validation (default: 5)')
    
    # Regularization
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing value (0.0-1.0, default: 0.1). Set to 0 to disable.')
    
    # Other
    parser.add_argument('--output', type=str, default=None,
                        help='Output model path')
    parser.add_argument('--no-language', action='store_true',
                        help='Exclude language_encoded feature')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu, default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device) if args.device else DEVICE
    print(f"Using device: {device}")
    
    # Set label smoothing
    global LABEL_SMOOTHING
    LABEL_SMOOTHING = args.label_smoothing
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    
    # Load data
    print("LOADING DATA")
    
    X, y, feature_names = load_and_preprocess_data(
        [args.features], 
        include_language=not args.no_language
    )
    
    print(f"Features shape: {X.shape}")
    print(f"Labels: {np.bincount(y)} (Human={np.sum(y==0)}, AI={np.sum(y==1)})")
    
    # Train model
    if args.no_optimize:
        # Use default/specified hyperparameters
        hidden_sizes = [int(x) for x in args.hidden_sizes.split(',')]
        model, scaler, feature_names, history, metrics, test_loader = train_default_mlp(
            X, y, feature_names,
            hidden_sizes=hidden_sizes,
            dropout=args.dropout,
            lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            device=device
        )
        study = None
        best_params = {
            'hidden_sizes': hidden_sizes,
            'dropout': args.dropout,
            'lr': args.lr,
            'batch_size': args.batch_size,
        }
    else:
        # Run Optuna optimization
        model, scaler, feature_names, study, metrics, best_params, test_loader, history = run_optimization(
            X, y, feature_names,
            n_trials=args.trials,
            timeout=args.timeout,
            max_epochs=args.epochs,
            device=device
        )
    
    # Save model
    output_path = save_model(
        model, scaler, feature_names, metrics, best_params,
        output_path=args.output, study=study
    )
    
    # Plot optimization history
    if study is not None and PLOTTING_AVAILABLE:
        plot_path = output_path.with_name(output_path.stem + '_optimization.png')
        plot_optimization_history(study, save_path=plot_path)
    
    # Run evaluation and generate visualizations
    if PLOTTING_AVAILABLE:
        eval_dir = Path(output_path).parent
        criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
        run_evaluation(
            model, test_loader, criterion, device, 
            eval_dir, model_name="MLP"
        )
        
        # Plot training/validation loss curves (available for both modes now)
        if history:
            loss_plot_path = eval_dir / 'training_validation_loss.png'
            plot_training_validation_loss(history, save_path=loss_plot_path)
    print("TRAINING COMPLETE")
    print(f"\nFinal Test F1: {metrics['f1']:.4f}")
    print(f"Model saved to: {output_path}")
    if PLOTTING_AVAILABLE:
        print(f"Evaluation results saved to: {eval_dir}")
    
    # Run cross-validation if requested
    if args.cv:
        print("RUNNING CROSS-VALIDATION FOR ROBUST EVALUATION")
        
        # Use best hyperparameters found (or default)
        cv_results = cross_validate_mlp(
            X, y, feature_names,
            hyperparams=best_params,
            n_folds=args.cv_folds,
            max_epochs=args.epochs,
            device=device,
            verbose=True
        )
        
        # Save CV results
        cv_output_dir = Path(output_path).parent / 'cross_validation'
        save_cv_results(cv_results, cv_output_dir)
        
        # Plot CV results
        if PLOTTING_AVAILABLE:
            plot_cv_results(cv_results, save_dir=cv_output_dir)
        print("CROSS-VALIDATION SUMMARY")
        print(f"\nMean F1 Score: {cv_results['mean']['f1']:.4f} ± {cv_results['std']['f1']:.4f}")
        print(f"Mean Accuracy: {cv_results['mean']['accuracy']:.4f} ± {cv_results['std']['accuracy']:.4f}")
        print(f"Mean ROC-AUC:  {cv_results['mean']['roc_auc']:.4f} ± {cv_results['std']['roc_auc']:.4f}")
        print(f"\nCV results saved to: {cv_output_dir}")


if __name__ == "__main__":
    main()
