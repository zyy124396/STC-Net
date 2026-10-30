import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix


CLASS_NAMES = ["Normal", "CHF", "CAD"]
NUM_CLASSES = 3
EXPECTED_PARAMETER_COUNT = 569_603
INPUT_CHANNELS = 2
INPUT_LENGTH = 300


def set_seed(seed: int) -> None:
    """Set Python, NumPy and PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")
    return torch.device(device_arg)


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation channel recalibration.

    Manuscript:
      Global average pooling over sequence length;
      FC C -> C/4 (no bias) -> ReLU;
      FC C/4 -> C (no bias) -> Hardsigmoid;
      channel-wise multiplication.
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden_channels = channels // reduction
        if channels % reduction != 0:
            raise ValueError(
                f"SE channels ({channels}) must be divisible by reduction ({reduction})."
            )
        self.fc1 = nn.Linear(channels, hidden_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_channels, channels, bias=False)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, length)
        channel_descriptor = x.mean(dim=-1)  # global average pooling
        channel_weights = self.fc1(channel_descriptor)
        channel_weights = self.relu(channel_weights)
        channel_weights = self.fc2(channel_weights)
        channel_weights = self.hardsigmoid(channel_weights)
        return x * channel_weights.unsqueeze(-1)


class CRRSBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        gn_groups: int = 8,
        se_reduction: int = 4,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )
        self.gn1 = nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
        )
        self.gn2 = nn.GroupNorm(num_groups=gn_groups, num_channels=out_channels)
        self.se = SEBlock1D(out_channels, reduction=se_reduction)

        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1, bias=True
            )
        else:
            self.shortcut = nn.Identity()

        self.relu_out = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.gn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.gn2(out)
        out = self.se(out)

        out = self.relu_out(out + residual)
        out = self.max_pool(out)
        return out


class SpatialRepresentation(nn.Module):
    """Three serial CRRS blocks: 2->64->128->128."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            CRRSBlock(2, 64),
            CRRSBlock(64, 128),
            CRRSBlock(128, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class TemporalPerception(nn.Module):


    def __init__(self):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SR output: (B, 128, 37) -> sequence: (B, 37, 128)
        sequence = x.transpose(1, 2)
        sequence_output, _ = self.bilstm(sequence)

        # Manuscript Eq. (10): select z_T, where T=37.
        # This is output at the final sequence position, not h_n concatenation.
        final_bidirectional_feature = sequence_output[:, -1, :]
        return final_bidirectional_feature


class ClassificationDecision(nn.Module):


    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, NUM_CLASSES)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        # Return logits. Softmax is applied for probability-based evaluation;
        # FocalLoss uses log_softmax internally for numerical stability.
        return x


class STCNet(nn.Module):

    def __init__(self, dropout: float = 0.3):
        super().__init__()
        self.sr = SpatialRepresentation()
        self.tp = TemporalPerception()
        self.cd = ClassificationDecision(dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"STC-Net expects a 3D tensor (B,2,300), but received {tuple(x.shape)}."
            )
        if x.shape[1] != INPUT_CHANNELS or x.shape[2] != INPUT_LENGTH:
            raise ValueError(
                f"STC-Net expects input shape (B,{INPUT_CHANNELS},{INPUT_LENGTH}), "
                f"but received {tuple(x.shape)}."
            )
        x = self.sr(x)
        x = self.tp(x)
        logits = self.cd(x)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities for inference."""
        return torch.softmax(self.forward(x), dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def verify_architecture() -> None:

    model = STCNet(dropout=0.3)
    parameter_count = count_parameters(model)

   
    with torch.no_grad():
        dummy = torch.zeros(2, INPUT_CHANNELS, INPUT_LENGTH)
        sr_output = model.sr(dummy)
        logits = model(dummy)

    expected_sr_shape = (2, 128, 37)
    if tuple(sr_output.shape) != expected_sr_shape:
        raise RuntimeError(
            f"SR output mismatch: expected {expected_sr_shape}, got {tuple(sr_output.shape)}."
        )
    if tuple(logits.shape) != (2, NUM_CLASSES):
        raise RuntimeError(
            f"Classifier output mismatch: expected (2,{NUM_CLASSES}), got {tuple(logits.shape)}."
        )
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Parameter-count audit failed: "
            f"expected {EXPECTED_PARAMETER_COUNT:,} (paper), got {parameter_count:,}. "
            "Please check whether the architecture has been modified."
        )

    print(f"[Architecture check] STC-Net parameters: {parameter_count:,} (matches paper).")
    print("[Architecture check] SR output: (B, 128, 37); logits: (B, 3).")


class WeightedFocalLoss(nn.Module):

    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        if class_weights.numel() != NUM_CLASSES:
            raise ValueError(f"class_weights must have {NUM_CLASSES} elements.")
        self.register_buffer("alpha", class_weights.float())
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probabilities = F.log_softmax(logits, dim=1)
        log_pt = log_probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
        pt = log_pt.exp()
        alpha_t = self.alpha[targets]
        focal_factor = (1.0 - pt).pow(self.gamma)
        loss = -alpha_t * focal_factor * log_pt
        return loss.mean()


def encode_labels(y: np.ndarray) -> np.ndarray:

    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.number):
        encoded = y.astype(np.int64)
    else:
        label_map = {
            "N": 0,
            "NORMAL": 0,
            "NORMAL SINUS RHYTHM": 0,
            "NSR": 0,
            "CHF": 1,
            "CONGESTIVE HEART FAILURE": 1,
            "CAD": 2,
            "CORONARY ARTERY DISEASE": 2,
        }
        encoded_values = []
        for value in y:
            key = str(value).strip().upper()
            if key not in label_map:
                raise ValueError(
                    f"Unrecognized label '{value}'. Edit encode_labels() to map it "
                    "to Normal=0, CHF=1, or CAD=2."
                )
            encoded_values.append(label_map[key])
        encoded = np.asarray(encoded_values, dtype=np.int64)

    if np.any((encoded < 0) | (encoded >= NUM_CLASSES)):
        raise ValueError(
            "Numeric labels must be 0, 1, or 2, corresponding to Normal, CHF, CAD."
        )
    return encoded


def load_npz_dataset(
    data_path: str,
    require_labels: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:

    data_path = str(data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset file not found: {data_path}")

    with np.load(data_path, allow_pickle=False) as data:
        if "X" not in data:
            raise KeyError("The .npz file must contain an array named 'X'.")
        X = np.asarray(data["X"], dtype=np.float32)

        if "patient_id" not in data:
            raise KeyError(
                "The .npz file must contain 'patient_id' (one patient ID per heartbeat)."
            )
        patient_ids = np.asarray(data["patient_id"]).astype(str)

        y = None
        if "y" in data:
            y = encode_labels(np.asarray(data["y"]))
        elif require_labels:
            raise KeyError("The .npz file must contain 'y' for training/CV.")

    # Accept both channel-first and channel-last arrays.
    if X.ndim != 3:
        raise ValueError(f"X must be 3D; received shape {X.shape}.")
    if X.shape[1:] == (INPUT_CHANNELS, INPUT_LENGTH):
        pass
    elif X.shape[1:] == (INPUT_LENGTH, INPUT_CHANNELS):
        X = np.transpose(X, (0, 2, 1))
    else:
        raise ValueError(
            "X must have shape (N,2,300) or (N,300,2); "
            f"received {X.shape}."
        )

    if len(X) != len(patient_ids):
        raise ValueError("X and patient_id have different numbers of samples.")
    if y is not None and len(X) != len(y):
        raise ValueError("X and y have different numbers of samples.")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values.")

    return X.astype(np.float32), y, patient_ids



def build_patient_label_map(
    patient_ids: np.ndarray,
    y: np.ndarray,
) -> Dict[str, int]:
    patient_label_map: Dict[str, int] = {}
    for patient_id, label in zip(patient_ids, y):
        patient_id = str(patient_id)
        label = int(label)
        if patient_id in patient_label_map and patient_label_map[patient_id] != label:
            raise ValueError(
                f"Patient {patient_id} has inconsistent labels across heartbeats."
            )
        patient_label_map[patient_id] = label
    return patient_label_map


def calculate_patient_class_weights(
    train_patient_ids: Sequence[str],
    patient_label_map: Dict[str, int],
) -> torch.Tensor:
    train_labels = np.asarray(
        [patient_label_map[str(patient_id)] for patient_id in train_patient_ids],
        dtype=np.int64,
    )
    counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    if np.any(counts == 0):
        raise ValueError(
            "At least one class is absent from the training patients in this fold; "
            f"patient counts are {counts.tolist()}."
        )
    n_train_patients = len(train_labels)
    weights = n_train_patients / (NUM_CLASSES * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def aggregate_patient_probabilities(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    patient_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    patient_ids = np.asarray(patient_ids).astype(str)
    unique_ids, inverse = np.unique(patient_ids, return_inverse=True)

    probability_sum = np.zeros((len(unique_ids), NUM_CLASSES), dtype=np.float64)
    counts = np.bincount(inverse, minlength=len(unique_ids))
    np.add.at(probability_sum, inverse, probabilities)
    patient_probabilities = probability_sum / counts[:, None]

    if y_true is None:
        return unique_ids, None, patient_probabilities

    patient_y = np.empty(len(unique_ids), dtype=np.int64)
    for index in range(len(unique_ids)):
        labels = np.unique(y_true[inverse == index])
        if len(labels) != 1:
            raise ValueError(
                f"Patient {unique_ids[index]} has inconsistent labels in evaluation data."
            )
        patient_y[index] = int(labels[0])

    return unique_ids, patient_y, patient_probabilities


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total else float("nan")

    sensitivity = []
    specificity = []
    precision = []
    f1_scores = []

    for class_index in range(NUM_CLASSES):
        tp = cm[class_index, class_index]
        fn = cm[class_index, :].sum() - tp
        fp = cm[:, class_index].sum() - tp
        tn = total - tp - fn - fp

        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        pre = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * pre * sen / (pre + sen)) if (pre + sen) > 0 else 0.0

        sensitivity.append(float(sen))
        specificity.append(float(spe))
        precision.append(float(pre))
        f1_scores.append(float(f1))

    return {
        "OA": accuracy,
        "Macro-Sen": float(np.mean(sensitivity)),
        "Macro-Spe": float(np.mean(specificity)),
        "Macro-Pre": float(np.mean(precision)),
        "Macro-F1": float(np.mean(f1_scores)),
    }


def make_prediction_dataframe(
    patient_ids: np.ndarray,
    probabilities: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    y_pred = probabilities.argmax(axis=1)
    output = {
        "patient_id": np.asarray(patient_ids).astype(str),
    }
    if y_true is not None:
        output["y_true"] = y_true.astype(int)
        output["true_class"] = [CLASS_NAMES[int(i)] for i in y_true]
    output["y_pred"] = y_pred.astype(int)
    output["predicted_class"] = [CLASS_NAMES[int(i)] for i in y_pred]
    for class_index, class_name in enumerate(CLASS_NAMES):
        output[f"prob_{class_name}"] = probabilities[:, class_index]
    return pd.DataFrame(output)


def make_tensor_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(np.asarray(X, dtype=np.float32)),
        torch.from_numpy(np.asarray(y, dtype=np.int64)),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def infer_probabilities(
    model: nn.Module,
    X: np.ndarray,
    y: Optional[np.ndarray],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, Optional[float]]:
    if y is None:
        dummy_y = np.zeros(len(X), dtype=np.int64)
        has_labels = False
    else:
        dummy_y = y
        has_labels = True

    loader = make_tensor_loader(
        X, dummy_y, batch_size, shuffle=False,
        num_workers=num_workers, device=device
    )

    model.eval()
    probability_batches = []
    loss_sum = 0.0
    n_samples = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        logits = model(xb)

        if has_labels and criterion is not None:
            loss = criterion(logits, yb)
            loss_sum += float(loss.item()) * len(xb)

        probability_batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        n_samples += len(xb)

    probabilities = np.concatenate(probability_batches, axis=0)
    mean_loss = (loss_sum / n_samples) if (has_labels and criterion is not None) else None
    return probabilities, mean_loss


def evaluate_split(
    model: nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, object]:
    probabilities, mean_loss = infer_probabilities(
        model=model,
        X=X,
        y=y,
        batch_size=batch_size,
        num_workers=num_workers,
        device=device,
        criterion=criterion,
    )

    heartbeat_metrics = calculate_metrics(y, probabilities.argmax(axis=1))
    unique_ids, patient_y, patient_probabilities = aggregate_patient_probabilities(
        y, probabilities, patient_ids
    )
    patient_metrics = calculate_metrics(
        patient_y, patient_probabilities.argmax(axis=1)
    )

    return {
        "loss": float(mean_loss),
        "heartbeat_metrics": heartbeat_metrics,
        "patient_metrics": patient_metrics,
        "probabilities": probabilities,
        "patient_ids": unique_ids,
        "patient_y": patient_y,
        "patient_probabilities": patient_probabilities,
    }


def train_one_fold(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    train_patient_ids: Sequence[str],
    patient_label_map: Dict[str, int],
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    fold: int,
    fold_dir: Path,
) -> Tuple[STCNet, Dict[str, object], int]:
    fold_dir.mkdir(parents=True, exist_ok=True)

    class_weights = calculate_patient_class_weights(
        train_patient_ids, patient_label_map
    ).to(device)

    model = STCNet(dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    criterion = WeightedFocalLoss(
        class_weights=class_weights,
        gamma=args.gamma,
    ).to(device)

    train_loader = make_tensor_loader(
        X[train_indices],
        y[train_indices],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        device=device,
    )

    history = []
    best_epoch = -1
    best_score = -float("inf")
    if args.selection_metric == "val_loss":
        best_score = float("inf")

    checkpoint_path = fold_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(xb)
            train_n += len(xb)

        train_loss = train_loss_sum / max(train_n, 1)

        validation = evaluate_split(
            model=model,
            X=X[val_indices],
            y=y[val_indices],
            patient_ids=patient_ids[val_indices],
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            criterion=criterion,
        )

        val_patient_metrics = validation["patient_metrics"]
        val_heartbeat_metrics = validation["heartbeat_metrics"]
        val_loss = float(validation["loss"])

        if args.selection_metric == "patient_macro_f1":
            current_score = float(val_patient_metrics["Macro-F1"])
            is_better = current_score > best_score
        elif args.selection_metric == "patient_accuracy":
            current_score = float(val_patient_metrics["OA"])
            is_better = current_score > best_score
        else:  # val_loss
            current_score = val_loss
            is_better = current_score < best_score

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_patient_OA": val_patient_metrics["OA"],
            "val_patient_Macro_F1": val_patient_metrics["Macro-F1"],
            "val_heartbeat_OA": val_heartbeat_metrics["OA"],
            "val_heartbeat_Macro_F1": val_heartbeat_metrics["Macro-F1"],
        })

        if is_better:
            best_score = current_score
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dropout": args.dropout,
                    "class_names": CLASS_NAMES,
                    "input_shape": [INPUT_CHANNELS, INPUT_LENGTH],
                    "class_weights": class_weights.detach().cpu(),
                    "gamma": args.gamma,
                    "seed": seed,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "selection_metric": args.selection_metric,
                    "selection_score": best_score,
                    "parameter_count": count_parameters(model),
                },
                checkpoint_path,
            )

        print(
            f"[Seed {seed} | Fold {fold}] "
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
            f"val_patient_Macro-F1={val_patient_metrics['Macro-F1']:.4f}"
        )

    pd.DataFrame(history).to_csv(fold_dir / "training_history.csv", index=False)

    if not checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was saved.")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint, best_epoch


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_label_map = build_patient_label_map(patient_ids, y)
    unique_patient_ids = np.asarray(list(patient_label_map.keys()), dtype=str)
    patient_labels = np.asarray(
        [patient_label_map[pid] for pid in unique_patient_ids],
        dtype=np.int64,
    )

    class_patient_counts = np.bincount(patient_labels, minlength=NUM_CLASSES)
    if np.any(class_patient_counts < args.folds):
        raise ValueError(
            "Each class must contain at least --folds patients for stratified CV. "
            f"Patient counts (Normal, CHF, CAD): {class_patient_counts.tolist()}."
        )

    print(f"Total heartbeats: {len(X)}")
    print(f"Total patients: {len(unique_patient_ids)}")
    print(f"Patient class counts (Normal, CHF, CAD): {class_patient_counts.tolist()}")
    print(f"Device: {device}")

    all_seed_metric_rows = []
    all_fold_metric_rows = []

    for seed in args.seeds:
        set_seed(seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # One OOF prediction per heartbeat for this seed.
        oof_probabilities = np.full(
            (len(X), NUM_CLASSES), np.nan, dtype=np.float32
        )
        oof_fold_ids = np.full(len(X), -1, dtype=np.int64)

        splitter = StratifiedKFold(
            n_splits=args.folds,
            shuffle=True,
            random_state=seed,
        )

        for fold, (development_patient_indices, test_patient_indices) in enumerate(
            splitter.split(np.zeros(len(unique_patient_ids)), patient_labels),
            start=1,
        ):
            development_patient_ids = unique_patient_ids[development_patient_indices]
            test_patient_ids = unique_patient_ids[test_patient_indices]
            development_labels = patient_labels[development_patient_indices]

            # Manuscript: split 32 development patients into 24 train + 8 validation.
            train_patient_ids, val_patient_ids = train_test_split(
                development_patient_ids,
                test_size=args.val_ratio,
                random_state=seed * 100 + fold,
                stratify=development_labels,
            )

            # Explicit patient-level index assignment: no patient leakage.
            train_indices = np.flatnonzero(np.isin(patient_ids, train_patient_ids))
            val_indices = np.flatnonzero(np.isin(patient_ids, val_patient_ids))
            test_indices = np.flatnonzero(np.isin(patient_ids, test_patient_ids))

            train_set = set(train_patient_ids.tolist())
            val_set = set(val_patient_ids.tolist())
            test_set = set(test_patient_ids.tolist())
            if train_set & val_set or train_set & test_set or val_set & test_set:
                raise RuntimeError("Patient overlap detected among train/val/test sets.")

            print(
                f"\n[Seed {seed} | Fold {fold}] "
                f"patients: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}; "
                f"heartbeats: train={len(train_indices)}, val={len(val_indices)}, "
                f"test={len(test_indices)}"
            )

            fold_dir = seed_dir / f"fold_{fold}"
            model, checkpoint, best_epoch = train_one_fold(
                X=X,
                y=y,
                patient_ids=patient_ids,
                train_indices=train_indices,
                val_indices=val_indices,
                train_patient_ids=train_patient_ids,
                patient_label_map=patient_label_map,
                args=args,
                device=device,
                seed=seed,
                fold=fold,
                fold_dir=fold_dir,
            )

            # The test subset is evaluated only after selecting the best checkpoint.
            test_criterion = WeightedFocalLoss(
                class_weights=checkpoint["class_weights"].to(device),
                gamma=float(checkpoint["gamma"]),
            ).to(device)
            test_result = evaluate_split(
                model=model,
                X=X[test_indices],
                y=y[test_indices],
                patient_ids=patient_ids[test_indices],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                criterion=test_criterion,
            )

            oof_probabilities[test_indices] = test_result["probabilities"]
            oof_fold_ids[test_indices] = fold

            fold_row = {
                "seed": seed,
                "fold": fold,
                "best_epoch": best_epoch,
                "train_patients": len(train_set),
                "val_patients": len(val_set),
                "test_patients": len(test_set),
                "train_heartbeats": len(train_indices),
                "val_heartbeats": len(val_indices),
                "test_heartbeats": len(test_indices),
                "class_weights": json.dumps(
                    checkpoint["class_weights"].cpu().numpy().tolist()
                ),
            }
            for key, value in test_result["heartbeat_metrics"].items():
                fold_row[f"heartbeat_{key}"] = value
            for key, value in test_result["patient_metrics"].items():
                fold_row[f"patient_{key}"] = value
            all_fold_metric_rows.append(fold_row)

            # Save fold-specific test predictions for traceability.
            fold_beat_df = make_prediction_dataframe(
                patient_ids=patient_ids[test_indices],
                probabilities=test_result["probabilities"],
                y_true=y[test_indices],
            )
            fold_beat_df.insert(0, "heartbeat_index", test_indices)
            fold_beat_df.insert(1, "fold", fold)
            fold_beat_df.to_csv(fold_dir / "test_heartbeat_predictions.csv", index=False)

            fold_patient_df = make_prediction_dataframe(
                patient_ids=test_result["patient_ids"],
                probabilities=test_result["patient_probabilities"],
                y_true=test_result["patient_y"],
            )
            fold_patient_df.insert(0, "fold", fold)
            fold_patient_df.to_csv(fold_dir / "test_patient_predictions.csv", index=False)

        if np.isnan(oof_probabilities).any() or np.any(oof_fold_ids < 0):
            raise RuntimeError(
                f"Seed {seed}: OOF predictions are incomplete. Check CV partitioning."
            )

        # Aggregate OOF heartbeat probabilities to patient-level probabilities.
        heartbeat_predictions = oof_probabilities.argmax(axis=1)
        heartbeat_metrics = calculate_metrics(y, heartbeat_predictions)

        oof_patient_ids, oof_patient_y, oof_patient_probabilities = (
            aggregate_patient_probabilities(y, oof_probabilities, patient_ids)
        )
        patient_predictions = oof_patient_probabilities.argmax(axis=1)
        patient_metrics = calculate_metrics(oof_patient_y, patient_predictions)

        for level, metrics in [
            ("heartbeat", heartbeat_metrics),
            ("patient", patient_metrics),
        ]:
            row = {"seed": seed, "level": level}
            row.update(metrics)
            all_seed_metric_rows.append(row)

        heartbeat_df = make_prediction_dataframe(
            patient_ids=patient_ids,
            probabilities=oof_probabilities,
            y_true=y,
        )
        heartbeat_df.insert(0, "heartbeat_index", np.arange(len(X)))
        heartbeat_df.insert(1, "fold", oof_fold_ids)
        heartbeat_df.to_csv(
            seed_dir / f"seed_{seed}_oof_heartbeat_predictions.csv",
            index=False,
        )

        patient_df = make_prediction_dataframe(
            patient_ids=oof_patient_ids,
            probabilities=oof_patient_probabilities,
            y_true=oof_patient_y,
        )
        patient_df.to_csv(
            seed_dir / f"seed_{seed}_oof_patient_predictions.csv",
            index=False,
        )
        pd.DataFrame(
            confusion_matrix(y, heartbeat_predictions, labels=np.arange(NUM_CLASSES)),
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(seed_dir / f"seed_{seed}_heartbeat_confusion_matrix.csv")
        pd.DataFrame(
            confusion_matrix(
                oof_patient_y, patient_predictions, labels=np.arange(NUM_CLASSES)
            ),
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_csv(seed_dir / f"seed_{seed}_patient_confusion_matrix.csv")

        print(f"\n[Seed {seed}] OOF heartbeat metrics: {heartbeat_metrics}")
        print(f"[Seed {seed}] OOF patient metrics:   {patient_metrics}")

    fold_metrics_df = pd.DataFrame(all_fold_metric_rows)
    fold_metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)

    seed_metrics_df = pd.DataFrame(all_seed_metric_rows)
    seed_metrics_df.to_csv(output_dir / "metrics_by_seed.csv", index=False)

    metric_columns = ["OA", "Macro-Sen", "Macro-Spe", "Macro-Pre", "Macro-F1"]
    summary_rows = []
    for level, group in seed_metrics_df.groupby("level"):
        for metric in metric_columns:
            summary_rows.append({
                "level": level,
                "metric": metric,
                "mean": group[metric].mean(),
                "sd": (
                    group[metric].std(ddof=1)
                    if len(group[metric]) > 1
                    else 0.0
                ),
                "mean_plus_minus_sd": (
                    f"{group[metric].mean() * 100:.2f} ± "
                    f"{(group[metric].std(ddof=1) if len(group[metric]) > 1 else 0.0) * 100:.2f}%"
                ),
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "metrics_mean_std.csv", index=False)

    print("\n========== Mean ± SD across random seeds ==========")
    print(summary_df[["level", "metric", "mean_plus_minus_sd"]].to_string(index=False))
    print(f"\nAll results saved to: {output_dir.resolve()}")


def run_prediction(
    X: np.ndarray,
    y: Optional[np.ndarray],
    patient_ids: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required when --mode predict.")

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = STCNet(dropout=float(checkpoint.get("dropout", args.dropout))).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    probabilities, _ = infer_probabilities(
        model=model,
        X=X,
        y=y,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        criterion=None,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heartbeat_df = make_prediction_dataframe(
        patient_ids=patient_ids,
        probabilities=probabilities,
        y_true=y,
    )
    heartbeat_df.insert(0, "heartbeat_index", np.arange(len(X)))
    heartbeat_df.to_csv(output_dir / "heartbeat_predictions.csv", index=False)

    patient_ids_agg, patient_y, patient_probabilities = aggregate_patient_probabilities(
        y, probabilities, patient_ids
    )
    patient_df = make_prediction_dataframe(
        patient_ids=patient_ids_agg,
        probabilities=patient_probabilities,
        y_true=patient_y,
    )
    patient_df.to_csv(output_dir / "patient_predictions.csv", index=False)

    if y is not None:
        heartbeat_metrics = calculate_metrics(y, probabilities.argmax(axis=1))
        patient_metrics = calculate_metrics(
            patient_y, patient_probabilities.argmax(axis=1)
        )
        with open(output_dir / "test_metrics.json", "w", encoding="utf-8") as file:
            json.dump(
                {
                    "heartbeat_level": heartbeat_metrics,
                    "patient_level": patient_metrics,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        print("Heartbeat-level metrics:", heartbeat_metrics)
        print("Patient-level metrics:", patient_metrics)

    print(f"Predictions saved to: {output_dir.resolve()}")

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible PyTorch implementation of STC-Net."
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["cv", "predict"],
        default="cv",
        help="cv: repeated patient-level stratified CV; predict: external inference.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--gamma", type=float, default=2.0)

    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.25)
    parser.add_argument(
        "--selection_metric",
        type=str,
        choices=["patient_macro_f1", "patient_accuracy", "val_loss"],
        default="patient_macro_f1",
        help=(
            "Validation checkpoint-selection metric. The manuscript does not "
            "specify this detail; default is patient-level Macro-F1."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument("--num_workers", type=int, default=0)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    verify_architecture()

    X, y, patient_ids = load_npz_dataset(
        args.data,
        require_labels=(args.mode == "cv"),
    )

    if args.mode == "cv":
        if y is None:
            raise ValueError("Training/CV mode requires ground-truth labels.")
        run_cross_validation(
            X=X,
            y=y,
            patient_ids=patient_ids,
            args=args,
            device=device,
        )
    else:
        run_prediction(
            X=X,
            y=y,
            patient_ids=patient_ids,
            args=args,
            device=device,
        )


if __name__ == "__main__":
    main()
