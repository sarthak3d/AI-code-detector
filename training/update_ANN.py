"""
Update (Incrementally Expand) an Existing MLP Model with New LLM Features
==========================================================================
When a new AI model is released, the workflow is:
  1. Generate AI code with the new model         (generate.py / generate_api.py)
  2. Extract features using the new model         (generate_features.py)
  3. Run this script to expand and fine-tune the existing MLP

This script:
  - Loads the existing trained MLP model (.pkl)
  - Loads the new feature CSV (containing BOTH old and new feature columns)
  - Identifies which features are new vs. already known
  - Expands the input layer to accommodate the new features
  - Transfers all learned weights from the old model
  - Fine-tunes the expanded model on the full dataset
  - Saves the updated model (same .pkl format as train_ANN.py)

Usage:
    # Update model with new features (recommended)
    python update_ANN.py \\
        --model dl_models/mlp_model_20251214_184436.pkl \\
        --features features/features_10000samples_5models.csv

    # Update with controlled fine-tuning
    python update_ANN.py \\
        --model dl_models/mlp_model_20251214_184436.pkl \\
        --features features/features_new.csv \\
        --lr 0.0005 --epochs 60 --freeze-epochs 10

    # Full re-initialization (discard old weights)
    python update_ANN.py \\
        --model dl_models/mlp_model_20251214_184436.pkl \\
        --features features/features_new.csv \\
        --reinitialize
"""

import os
import sys
import copy
import json
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from glob import glob
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("dl_models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


class CodeDetectorMLP(nn.Module):
    """Identical architecture to train_ANN.py for serialization compatibility."""

    ACTIVATIONS = {
        "relu": nn.ReLU,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "elu": nn.ELU,
    }

    def __init__(self, input_size: int, hidden_sizes: list, num_classes: int = 2,
                 dropout: float = 0.3, activation: str = "relu",
                 use_batch_norm: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.activation_name = activation
        self.use_batch_norm = use_batch_norm

        activation_fn = self.ACTIVATIONS.get(activation.lower(), nn.ReLU)

        if use_batch_norm:
            self.input_bn = nn.BatchNorm1d(input_size)
        else:
            self.input_bn = nn.Identity()

        layers = []
        prev_size = input_size
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(activation_fn())
            drop_rate = dropout * (1 - 0.2 * i / len(hidden_sizes))
            layers.append(nn.Dropout(drop_rate))
            prev_size = hidden_size

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_size, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
        return probs

    def get_config(self):
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "num_classes": self.num_classes,
            "dropout": self.dropout_rate,
            "activation": self.activation_name,
            "use_batch_norm": self.use_batch_norm,
        }


class EarlyStopping:
    """Early stopping to halt training when validation metric plateaus."""

    def __init__(self, patience: int = 10, min_delta: float = 0.001,
                 mode: str = "max", restore_best: bool = True):
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
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    def restore_best_weights(self, model):
        if self.restore_best and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


def load_existing_model(model_path: str, device: torch.device):
    """Load an existing model saved by train_ANN.py."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        model_data = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model_data = torch.load(path, map_location=device)

    config = model_data["model_config"]
    model = CodeDetectorMLP(
        input_size=config["input_size"],
        hidden_sizes=config["hidden_sizes"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        activation=config["activation"],
        use_batch_norm=config["use_batch_norm"],
    ).to(device)
    model.load_state_dict(model_data["model_state_dict"])
    model.eval()

    old_feature_names = list(model_data["feature_names"])
    old_scaler = model_data["scaler"]

    return model, config, old_feature_names, old_scaler, model_data


def load_and_preprocess_data(feature_paths: list, include_language: bool = True):
    """Load and preprocess feature CSV(s) -- identical logic to train_ANN.py."""
    dfs = []
    for path in feature_paths:
        if "*" in path:
            for f in glob(path):
                dfs.append(pd.read_csv(f))
        else:
            dfs.append(pd.read_csv(path))

    if not dfs:
        raise ValueError("No feature files found!")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(df)} samples from {len(dfs)} file(s)")

    exclude_cols = ["sample_id", "language", "label", "source"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    if not include_language and "language_encoded" in feature_cols:
        feature_cols.remove("language_encoded")

    print(f"Using {len(feature_cols)} features")

    X = df[feature_cols].values
    y = df["label"].values

    nan_mask = np.isnan(X)
    if nan_mask.any():
        print(f"Warning: Found {nan_mask.sum()} NaN values, replacing with column median")
        col_medians = np.nanmedian(X, axis=0)
        for i in range(X.shape[1]):
            X[nan_mask[:, i], i] = col_medians[i]

    X = np.clip(X, -1e10, 1e10)
    return X, y, feature_cols


def compute_feature_mapping(old_features: list, new_features: list) -> dict:
    """Compute mapping between old and new feature positions.

    Returns a dict with:
      - old_to_new: {old_idx: new_idx} for features that exist in both
      - new_only_indices: list of new indices that have no old counterpart
      - removed: list of old feature names no longer present
    """
    old_set = set(old_features)
    new_set = set(new_features)

    old_to_new = {}
    for old_idx, name in enumerate(old_features):
        if name in new_set:
            new_idx = new_features.index(name)
            old_to_new[old_idx] = new_idx

    new_only_indices = [
        i for i, name in enumerate(new_features) if name not in old_set
    ]
    removed = [name for name in old_features if name not in new_set]

    return {
        "old_to_new": old_to_new,
        "new_only_indices": new_only_indices,
        "removed": removed,
    }


def expand_model(old_model: CodeDetectorMLP, old_features: list,
                 new_features: list, device: torch.device,
                 reinitialize: bool = False) -> CodeDetectorMLP:
    """Create a new model with expanded input_size and transfer old weights.

    The first Linear layer's weight matrix is expanded: existing feature
    columns keep their learned weights, new columns are initialized with
    small random values (He init scaled down by 10x) so they don't
    dominate early in fine-tuning.
    """
    old_config = old_model.get_config()
    old_input_size = old_config["input_size"]
    new_input_size = len(new_features)

    mapping = compute_feature_mapping(old_features, new_features)

    print(f"\n{'=' * 60}")
    print("MODEL EXPANSION PLAN")
    print(f"{'=' * 60}")
    print(f"  Old input size: {old_input_size}")
    print(f"  New input size: {new_input_size}")
    print(f"  Features kept:  {len(mapping['old_to_new'])}")
    print(f"  Features added: {len(mapping['new_only_indices'])}")
    print(f"  Features removed: {len(mapping['removed'])}")

    if mapping["new_only_indices"]:
        new_names = [new_features[i] for i in mapping["new_only_indices"]]
        print(f"\n  New features:")
        for name in new_names[:20]:
            print(f"    + {name}")
        if len(new_names) > 20:
            print(f"    ... and {len(new_names) - 20} more")

    if mapping["removed"]:
        print(f"\n  Removed features:")
        for name in mapping["removed"][:10]:
            print(f"    - {name}")
        if len(mapping["removed"]) > 10:
            print(f"    ... and {len(mapping['removed']) - 10} more")

    new_model = CodeDetectorMLP(
        input_size=new_input_size,
        hidden_sizes=old_config["hidden_sizes"],
        num_classes=old_config["num_classes"],
        dropout=old_config["dropout"],
        activation=old_config["activation"],
        use_batch_norm=old_config["use_batch_norm"],
    ).to(device)

    if reinitialize:
        print("\n  Mode: FULL REINITIALIZATION (discarding old weights)")
        return new_model

    print("\n  Mode: WEIGHT TRANSFER + EXPANSION")

    old_state = old_model.state_dict()
    new_state = new_model.state_dict()

    # Transfer input BatchNorm parameters
    if old_config["use_batch_norm"]:
        old_bn_weight = old_state["input_bn.weight"]
        old_bn_bias = old_state["input_bn.bias"]
        old_bn_mean = old_state["input_bn.running_mean"]
        old_bn_var = old_state["input_bn.running_var"]

        for old_idx, new_idx in mapping["old_to_new"].items():
            new_state["input_bn.weight"][new_idx] = old_bn_weight[old_idx]
            new_state["input_bn.bias"][new_idx] = old_bn_bias[old_idx]
            new_state["input_bn.running_mean"][new_idx] = old_bn_mean[old_idx]
            new_state["input_bn.running_var"][new_idx] = old_bn_var[old_idx]

    # Transfer the first hidden Linear layer (the one that takes the input)
    first_linear_key_w = None
    first_linear_key_b = None
    for key in old_state:
        if key.startswith("hidden.") and key.endswith(".weight"):
            first_linear_key_w = key
            first_linear_key_b = key.replace(".weight", ".bias")
            break

    if first_linear_key_w and first_linear_key_w in old_state:
        old_w = old_state[first_linear_key_w]  # shape: [hidden_0, old_input_size]
        old_b = old_state[first_linear_key_b]

        # Scale new feature weights down so they don't dominate
        scale_factor = 0.1
        nn.init.kaiming_normal_(new_state[first_linear_key_w], mode="fan_out", nonlinearity="relu")
        new_state[first_linear_key_w] *= scale_factor

        # Copy existing feature weights
        for old_idx, new_idx in mapping["old_to_new"].items():
            new_state[first_linear_key_w][:, new_idx] = old_w[:, old_idx]

        new_state[first_linear_key_b] = old_b.clone()

    # Transfer all other layers unchanged (hidden layers after the first, output layer)
    for key in old_state:
        if key in new_state and key not in (
            first_linear_key_w, first_linear_key_b,
            "input_bn.weight", "input_bn.bias",
            "input_bn.running_mean", "input_bn.running_var",
            "input_bn.num_batches_tracked",
        ):
            if old_state[key].shape == new_state[key].shape:
                new_state[key] = old_state[key].clone()

    new_model.load_state_dict(new_state)
    transferred = len(mapping["old_to_new"])
    print(f"  Transferred weights for {transferred}/{old_input_size} input features")
    return new_model


def train_one_epoch(model, train_loader, optimizer, criterion, device):
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
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    unique_labels = np.unique(all_labels)

    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(unique_labels) > 1 else 0.5,
    }

    return metrics, all_preds, all_probs, all_labels


def fine_tune_model(model, X, y, feature_names, device, lr=0.0005,
                    batch_size=32, max_epochs=100, patience=15,
                    freeze_epochs=0, label_smoothing=0.1,
                    val_size=0.15, test_size=0.15):
    """Fine-tune the expanded model on the full dataset.

    Args:
        freeze_epochs: Number of initial epochs where only the new-feature
                       weights in the first layer are trainable (the rest frozen).
                       Helps the new features catch up without destabilizing
                       the already-learned representations.
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED,
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_ratio, stratify=y_trainval,
        random_state=RANDOM_SEED,
    )

    print(f"\nData split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val_s), torch.LongTensor(y_val)),
        batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_test_s), torch.LongTensor(y_test)),
        batch_size=batch_size, shuffle=False,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Phase 1: freeze old weights, only train new feature weights
    if freeze_epochs > 0:
        print(f"\nPhase 1: Warming up new features ({freeze_epochs} epochs, frozen backbone)")

        for name, param in model.named_parameters():
            param.requires_grad = False

        # Unfreeze only the first linear layer (where new feature weights live)
        for name, param in model.named_parameters():
            if name.startswith("hidden.0."):
                param.requires_grad = True
            if name.startswith("input_bn."):
                param.requires_grad = True

        warmup_optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr * 5, weight_decay=1e-5,
        )

        for epoch in range(freeze_epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, warmup_optimizer, criterion, device,
            )
            if (epoch + 1) % 5 == 0 or epoch == 0:
                val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)
                print(
                    f"  Warmup Epoch {epoch + 1:3d}: "
                    f"Train Loss={train_loss:.4f}, Val F1={val_metrics['f1']:.4f}"
                )

        # Unfreeze everything
        for param in model.parameters():
            param.requires_grad = True

    # Phase 2: full fine-tuning
    print(f"\nPhase 2: Full fine-tuning (up to {max_epochs} epochs)")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=False,
    )
    early_stopping = EarlyStopping(patience=patience, mode="max")

    history = {"train_loss": [], "val_loss": [], "val_f1": []}

    for epoch in range(max_epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )
        val_metrics, _, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_f1"].append(val_metrics["f1"])

        scheduler.step(val_metrics["f1"])
        early_stopping(val_metrics["f1"], model)

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch + 1:3d}: Train Loss={train_loss:.4f}, "
                f"Val Loss={val_metrics['loss']:.4f}, Val F1={val_metrics['f1']:.4f}, "
                f"LR={current_lr:.6f}"
            )

        if early_stopping.early_stop:
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    early_stopping.restore_best_weights(model)

    # Evaluate on test set
    print(f"\n{'=' * 60}")
    print("TEST RESULTS (UPDATED MODEL)")
    print(f"{'=' * 60}")

    test_metrics, test_preds, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device,
    )

    print(f"\n  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")

    cm = confusion_matrix(test_labels, test_preds)
    print(f"\n  Confusion Matrix:")
    print(f"    [[TN={cm[0, 0]}, FP={cm[0, 1]}],")
    print(f"     [FN={cm[1, 0]}, TP={cm[1, 1]}]]")

    print(f"\n  Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=["Human", "AI"]))

    # Compute optimal threshold
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    print(f"  Optimal threshold: {optimal_threshold:.4f}")

    return model, scaler, test_metrics, optimal_threshold, history, test_loader


def save_updated_model(model, scaler, feature_names, metrics, old_model_data,
                       optimal_threshold, update_info, output_path=None):
    """Save the updated model in the same format as train_ANN.py."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"mlp_model_{timestamp}.pkl"
    else:
        output_path = Path(output_path)

    model_data = {
        "model_name": "MLP",
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "scaler": scaler,
        "feature_names": feature_names,
        "metrics": metrics,
        "hyperparameters": model.get_config(),
        "timestamp": datetime.now().isoformat(),
        "update_info": update_info,
    }

    torch.save(model_data, output_path)
    print(f"\nUpdated model saved to: {output_path}")

    # Save metrics.json compatible with the inference service
    metrics_json = {
        "timestamp": datetime.now().isoformat(),
        "model_info": {
            "model_path": str(output_path),
            "model_config": model.get_config(),
            "test_samples": update_info.get("test_samples", 0),
            "optimal_threshold": optimal_threshold,
        },
        "metrics": {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
        },
        "update_history": {
            "base_model": update_info.get("base_model_path", ""),
            "features_added": update_info.get("features_added", 0),
            "features_kept": update_info.get("features_kept", 0),
            "old_input_size": update_info.get("old_input_size", 0),
            "new_input_size": update_info.get("new_input_size", 0),
        },
    }

    metrics_path = output_path.with_name("metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Update an existing MLP model with new LLM features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard update (transfer weights + fine-tune)
  python update_ANN.py --model dl_models/mlp_model_20251214_184436.pkl \\
      --features features/features_10000samples_5models.csv

  # With warm-up phase for new features
  python update_ANN.py --model dl_models/mlp_model_20251214_184436.pkl \\
      --features features/features_new.csv --freeze-epochs 10

  # Full reinitialization (discard old weights)
  python update_ANN.py --model dl_models/mlp_model_20251214_184436.pkl \\
      --features features/features_new.csv --reinitialize
""",
    )

    parser.add_argument("--model", type=str, required=True,
                        help="Path to existing .pkl model file")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to new feature CSV file (supports wildcards)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for updated model (default: auto-generated)")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="Fine-tuning learning rate (default: 0.0005)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Maximum fine-tuning epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")
    parser.add_argument("--freeze-epochs", type=int, default=0,
                        help="Epochs to freeze old weights while warming up new features (default: 0)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                        help="Label smoothing (default: 0.1)")
    parser.add_argument("--reinitialize", action="store_true",
                        help="Discard old weights and train from scratch with new input size")
    parser.add_argument("--no-language", action="store_true",
                        help="Exclude language_encoded feature")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu, default: auto)")

    args = parser.parse_args()

    device = torch.device(args.device) if args.device else DEVICE
    print(f"Using device: {device}")

    # Step 1: Load existing model
    print("\n" + "=" * 70)
    print("STEP 1: LOADING EXISTING MODEL")
    print("=" * 70)

    old_model, old_config, old_features, old_scaler, old_model_data = load_existing_model(
        args.model, device,
    )

    print(f"  Model path:    {args.model}")
    print(f"  Input size:    {old_config['input_size']}")
    print(f"  Hidden sizes:  {old_config['hidden_sizes']}")
    print(f"  Features:      {len(old_features)}")
    print(f"  Old metrics:   {old_model_data.get('metrics', 'N/A')}")

    # Step 2: Load new feature data
    print("\n" + "=" * 70)
    print("STEP 2: LOADING NEW FEATURE DATA")
    print("=" * 70)

    X, y, new_features = load_and_preprocess_data(
        [args.features], include_language=not args.no_language,
    )

    print(f"  Features shape: {X.shape}")
    print(f"  Labels: Human={np.sum(y == 0)}, AI={np.sum(y == 1)}")

    # Step 3: Analyze feature changes
    print("\n" + "=" * 70)
    print("STEP 3: ANALYZING FEATURE CHANGES")
    print("=" * 70)

    mapping = compute_feature_mapping(old_features, new_features)

    if not mapping["new_only_indices"] and not mapping["removed"]:
        print("\n  No feature changes detected!")
        print("  The feature set is identical to the existing model.")
        print("  Consider using train_ANN.py instead for full retraining.")
        print("  Proceeding with fine-tuning on the new data anyway...\n")

    # Step 4: Expand model
    print("\n" + "=" * 70)
    print("STEP 4: EXPANDING MODEL")
    print("=" * 70)

    expanded_model = expand_model(
        old_model, old_features, new_features, device,
        reinitialize=args.reinitialize,
    )

    total_params = sum(p.numel() for p in expanded_model.parameters())
    print(f"\n  Total parameters: {total_params:,}")

    # Step 5: Fine-tune
    print("\n" + "=" * 70)
    print("STEP 5: FINE-TUNING EXPANDED MODEL")
    print("=" * 70)

    updated_model, new_scaler, test_metrics, optimal_threshold, history, test_loader = fine_tune_model(
        model=expanded_model,
        X=X,
        y=y,
        feature_names=new_features,
        device=device,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        freeze_epochs=args.freeze_epochs,
        label_smoothing=args.label_smoothing,
    )

    # Step 6: Save updated model
    print("\n" + "=" * 70)
    print("STEP 6: SAVING UPDATED MODEL")
    print("=" * 70)

    update_info = {
        "base_model_path": str(args.model),
        "old_input_size": old_config["input_size"],
        "new_input_size": len(new_features),
        "features_added": len(mapping["new_only_indices"]),
        "features_kept": len(mapping["old_to_new"]),
        "features_removed": len(mapping["removed"]),
        "reinitialize": args.reinitialize,
        "freeze_epochs": args.freeze_epochs,
        "learning_rate": args.lr,
        "test_samples": int(len(y) * 0.15),
        "update_timestamp": datetime.now().isoformat(),
        "new_feature_names": [
            new_features[i] for i in mapping["new_only_indices"]
        ],
    }

    output_path = save_updated_model(
        model=updated_model,
        scaler=new_scaler,
        feature_names=new_features,
        metrics=test_metrics,
        old_model_data=old_model_data,
        optimal_threshold=optimal_threshold,
        update_info=update_info,
        output_path=args.output,
    )

    # Compare old vs new
    old_metrics = old_model_data.get("metrics", {})

    print("\n" + "=" * 70)
    print("UPDATE COMPLETE -- COMPARISON")
    print("=" * 70)

    header = f"{'Metric':<12} {'Old':>10} {'New':>10} {'Delta':>10}"
    print(f"\n  {header}")
    print(f"  {'-' * 44}")

    for metric_key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        old_val = old_metrics.get(metric_key, old_metrics.get(f"{metric_key}_score", None))
        new_val = test_metrics.get(metric_key, 0)

        if old_val is not None:
            delta = new_val - old_val
            sign = "+" if delta >= 0 else ""
            print(f"  {metric_key.upper():<12} {old_val:>10.4f} {new_val:>10.4f} {sign}{delta:>9.4f}")
        else:
            print(f"  {metric_key.upper():<12} {'N/A':>10} {new_val:>10.4f}")

    print(f"\n  Input size: {old_config['input_size']} -> {len(new_features)}")
    print(f"  Model saved to: {output_path}")

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("  1. Copy the updated model to app/model/:")
    print(f"     cp {output_path} app/model/")
    print(f"     cp {output_path.with_name('metrics.json')} app/model/")
    print("  2. Update app/api/service.py FEATURE_MODELS if new scoring models added")
    print("  3. Rebuild Docker image if applicable")
    print("=" * 70)


if __name__ == "__main__":
    main()
