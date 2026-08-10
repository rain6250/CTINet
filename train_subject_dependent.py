"""
CTINet: Subject-Dependent Training Pipeline

Paper authors:
    Xinyu Zhang, Keum-Shik Hong, Guanghao Huang,
    Peng Sun, and Haiqiang Yang

Code implementation:
    Xinyu Zhang

Corresponding author:
    Haiqiang Yang

This script implements subject-dependent trial-level 10-fold
cross-validation. For each participant, trials are partitioned into
10 folds using a fixed random seed of 929 before sliding-window sample
construction. In each iteration, nine folds are used for gradient-based
training, while the remaining fold is held out from parameter updates
for model monitoring, learning-rate scheduling, early stopping, and
checkpoint selection.

Usage:
    python train_subject_dependent.py
    python train_subject_dependent.py --task MI
    python train_subject_dependent.py --task MA
    python train_subject_dependent.py --task WG
"""

import argparse
import gc
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from ctinet import CTINet


# ========================= configuration =========================
SEED = 929
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 15
MAX_EPOCHS = 300
BATCH_SIZE = 32

K_WINDOWS = 4
PROJ_DIM = 128
CMIT_LAYERS = 2
DROPOUT = 0.5

CHM_LAMBDA = 1.0
FNIRS_SCALE = 1e3
WINDOWS_PER_TRIAL = 10

OUTPUT_ROOT = "./results"

TASK_CONFIG = {
    "WG": {
        "subject_path": "./data/WG",
        "format": "interpolated",
    },
    "MA": {
        "subject_path": "./data/MA",
        "format": "prebuilt",
    },
    "MI": {
        "subject_path": "./data/MI",
        "format": "prebuilt",
    },
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_save_model(model, path):
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    temp_path = path + ".tmp"
    torch.save(state, temp_path)
    os.replace(temp_path, path)


def safe_load_model(path):
    return torch.load(path, map_location="cpu")


def build_wg_model_input(
    eeg_interp,
    hbo_interp,
    hbr_interp,
    labels,
    win_len_sec=3,
    time_offset_sec=3,
    eeg_segments=10,
    fnirs_segments=22,
    lag=11,
):
    """Construct the delayed-window EEG-fNIRS inputs for the WG task."""
    n_epochs = eeg_interp.shape[0]
    eeg_win_len = int(win_len_sec * 200)
    fnirs_win_len = int(win_len_sec * 10)

    eeg_interp = eeg_interp.astype(np.float32)
    hbo_interp = hbo_interp.astype(np.float32)
    hbr_interp = hbr_interp.astype(np.float32)

    eeg_windows = np.zeros(
        (n_epochs, eeg_segments, 16, 16, eeg_win_len), dtype=np.float32
    )
    hbo_windows = np.zeros(
        (n_epochs, fnirs_segments, 16, 16, fnirs_win_len), dtype=np.float32
    )
    hbr_windows = np.zeros_like(hbo_windows)

    off_eeg = int(time_offset_sec * 200)
    off_fnirs = int(time_offset_sec * 10)
    for epoch in range(n_epochs):
        for i in range(eeg_segments):
            start = off_eeg + i * 200
            end = start + eeg_win_len
            if end <= eeg_interp.shape[3]:
                eeg_windows[epoch, i] = eeg_interp[epoch, :, :, start:end]
        for i in range(fnirs_segments):
            start = off_fnirs + i * 10
            end = start + fnirs_win_len
            if end <= hbo_interp.shape[3]:
                hbo_windows[epoch, i] = hbo_interp[epoch, :, :, start:end]
                hbr_windows[epoch, i] = hbr_interp[epoch, :, :, start:end]

    primary = 10
    fnirs_lagged = np.zeros(
        (n_epochs, primary, lag, 16, 16, fnirs_win_len, 2), dtype=np.float32
    )
    for epoch in range(n_epochs):
        for p in range(primary):
            fnirs_lagged[epoch, p, :, :, :, :, 0] = hbo_windows[epoch, p:p + lag]
            fnirs_lagged[epoch, p, :, :, :, :, 1] = hbr_windows[epoch, p:p + lag]

    eeg_out = np.expand_dims(
        eeg_windows.reshape(n_epochs * eeg_segments, 16, 16, eeg_win_len), -1
    )
    fnirs_out = fnirs_lagged.reshape(
        n_epochs * primary, lag, 2, 16, 16, fnirs_win_len
    )
    labels_out = np.repeat(np.asarray(labels, dtype=np.int64), primary)
    return eeg_out, fnirs_out, labels_out


def trial_to_sample_indices(trial_indices):
    indices = []
    for trial in np.asarray(trial_indices, dtype=int):
        start = trial * WINDOWS_PER_TRIAL
        indices.extend(range(start, start + WINDOWS_PER_TRIAL))
    return np.asarray(indices, dtype=int)


class EEGfNIRSDataset(Dataset):
    def __init__(self, eeg, fnirs, labels):
        self.eeg = eeg
        self.fnirs = fnirs
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "eeg_input": torch.as_tensor(self.eeg[index], dtype=torch.float32),
            "fnirs_input": torch.as_tensor(self.fnirs[index], dtype=torch.float32),
        }, {
            "class_output": torch.tensor(self.labels[index], dtype=torch.long),
        }


def make_loader(eeg, fnirs, labels, shuffle):
    return DataLoader(
        EEGfNIRSDataset(eeg, fnirs, labels),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


class Trainer:
    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.criterion = nn.CrossEntropyLoss().to(DEVICE)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=PATIENCE // 2, factor=0.5
        )

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        for inputs, targets in loader:
            eeg = inputs["eeg_input"].to(DEVICE, non_blocking=True)
            fnirs = inputs["fnirs_input"].to(DEVICE, non_blocking=True)
            labels = targets["class_output"].to(DEVICE, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            output = self.model(eeg, fnirs, labels=labels)
            loss = self.criterion(output["class_output"], labels)
            loss = loss + CHM_LAMBDA * output["chm_loss"]

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(1, len(loader))

    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0.0
        predictions, truths = [], []
        with torch.no_grad():
            for inputs, targets in loader:
                eeg = inputs["eeg_input"].to(DEVICE, non_blocking=True)
                fnirs = inputs["fnirs_input"].to(DEVICE, non_blocking=True)
                labels = targets["class_output"].to(DEVICE, non_blocking=True)
                output = self.model(eeg, fnirs, labels=labels)
                total_loss += self.criterion(output["class_output"], labels).item()
                predictions.extend(output["class_output"].argmax(1).cpu().numpy())
                truths.extend(labels.cpu().numpy())

        predictions = np.asarray(predictions)
        truths = np.asarray(truths)
        kappa = cohen_kappa_score(truths, predictions)
        return {
            "loss": total_loss / max(1, len(loader)),
            "class_acc": float(np.mean(predictions == truths)),
            "class_f1": float(f1_score(truths, predictions, average="macro", zero_division=0)),
            "class_kappa": 0.0 if np.isnan(kappa) else float(kappa),
        }


def save_results(all_results, path, task):
    rows = []
    subject_means = []
    for subject, folds in all_results.items():
        for fold_name, metrics in folds.items():
            rows.append({"Subject": subject, "Fold": fold_name, **metrics})
        values = list(folds.values())
        mean_row = {
            "Subject": subject,
            "Fold": "Average",
            "class_acc": np.mean([x["class_acc"] for x in values]),
            "class_f1": np.mean([x["class_f1"] for x in values]),
            "class_kappa": np.mean([x["class_kappa"] for x in values]),
        }
        rows.append(mean_row)
        subject_means.append(mean_row)

    df = pd.DataFrame(rows)
    summary = pd.DataFrame([{
        "Task": task,
        "Subject mean accuracy": np.mean([x["class_acc"] for x in subject_means]),
        "Subject std accuracy": np.std([x["class_acc"] for x in subject_means]),
        "Subject mean F1": np.mean([x["class_f1"] for x in subject_means]),
        "Subject mean Kappa": np.mean([x["class_kappa"] for x in subject_means]),
    }])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="PerSubject")
        summary.to_excel(writer, index=False, sheet_name="Summary")


def run_fold(task, subject, fold, train_data, eval_data, output_dirs):
    eeg_train, fnirs_train, labels_train = train_data
    eeg_eval, fnirs_eval, labels_eval = eval_data
    fnirs_train = fnirs_train.astype(np.float32) * FNIRS_SCALE
    fnirs_eval = fnirs_eval.astype(np.float32) * FNIRS_SCALE

    train_loader = make_loader(eeg_train, fnirs_train, labels_train, True)
    eval_loader = make_loader(eeg_eval, fnirs_eval, labels_eval, False)

    model = CTINet(
        num_classes=2,
        windows=K_WINDOWS,
        latent_dim=PROJ_DIM,
        cmit_layers=CMIT_LAYERS,
        dropout=DROPOUT,
    )
    trainer = Trainer(model)
    print(f"[{task}][{subject}][fold {fold + 1}] params={count_parameters(model):,}")

    subject_name = subject.replace(".npz", "")
    best_path = os.path.join(
        output_dirs["best"], f"best_model_{subject_name}_fold{fold + 1}.pt"
    )
    best_acc = -1.0
    patience_counter = 0
    train_losses, eval_losses, eval_accs = [], [], []
    fold_start = time.time()

    for epoch in range(MAX_EPOCHS):
        epoch_start = time.time()
        train_loss = trainer.train_epoch(train_loader)
        eval_result = trainer.evaluate(eval_loader)
        trainer.scheduler.step(eval_result["loss"])

        train_losses.append(train_loss)
        eval_losses.append(eval_result["loss"])
        eval_accs.append(eval_result["class_acc"])
        print(
            f"[{task}][{subject}][fold {fold + 1}] epoch {epoch + 1:03d} | "
            f"train={train_loss:.4f} | eval_loss={eval_result['loss']:.4f} | "
            f"eval_acc={eval_result['class_acc']:.4f} | "
            f"time={time.time() - epoch_start:.1f}s"
        )

        if eval_result["class_acc"] > best_acc:
            best_acc = eval_result["class_acc"]
            safe_save_model(model, best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"[{task}][{subject}][fold {fold + 1}] early stopping")
                break

    model.load_state_dict(safe_load_model(best_path))
    final_result = trainer.evaluate(eval_loader)
    print(
        f"[{task}][{subject}][fold {fold + 1}] finished | "
        f"best_acc={best_acc:.4f} | final_acc={final_result['class_acc']:.4f} | "
        f"f1={final_result['class_f1']:.4f} | kappa={final_result['class_kappa']:.4f}"
    )

    curve_path = os.path.join(
        output_dirs["vis"], f"training_{subject_name}_fold{fold + 1}.png"
    )
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(eval_losses, label="Held-out loss")
    plt.plot(eval_accs, label="Held-out accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_path, dpi=150)
    plt.close()

    result = {
        "class_acc": final_result["class_acc"],
        "class_f1": final_result["class_f1"],
        "class_kappa": final_result["class_kappa"],
        "loss": final_result["loss"],
        "best_acc": best_acc,
        "time": time.time() - fold_start,
    }
    del model, trainer, train_loader, eval_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_task(task, debug=False):
    set_seed()
    config = TASK_CONFIG[task]
    root = os.path.join(OUTPUT_ROOT, f"CTINet_{task}")
    output_dirs = {
        "best": os.path.join(root, "best_parameter"),
        "results": os.path.join(root, "results"),
        "vis": os.path.join(root, "vis"),
    }
    for path in output_dirs.values():
        os.makedirs(path, exist_ok=True)

    subjects = sorted(
        f for f in os.listdir(config["subject_path"]) if f.endswith(".npz")
    )
    all_results = {}
    print(f"\n{'=' * 72}\nTask: {task}\nModel: CTINet\n"
          f"Split: subject-dependent trial-level 10-fold cross-validation\n"
          f"Output: {root}\n{'=' * 72}")

    for subject in subjects:
        if debug and subject != subjects[0]:
            break
        path = os.path.join(config["subject_path"], subject)
        folds = {}

        if config["format"] == "interpolated":
            with np.load(path, allow_pickle=True) as data:
                eeg = data["eeg"]
                hbo = data["hbo"]
                hbr = data["hbr"]
                labels = data["label"]
                eeg_fs = data.get("eeg_fs", np.array(200)).item()
                fnirs_fs = data.get("fnirs_fs", np.array(10)).item()
            del eeg_fs, fnirs_fs
            n_trials = eeg.shape[0]
            kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
            for fold, (train_idx, eval_idx) in enumerate(kf.split(np.arange(n_trials))):
                if debug and fold > 0:
                    break
                train_labels = np.argmax(labels[:, train_idx], axis=0)
                eval_labels = np.argmax(labels[:, eval_idx], axis=0)
                train_data = build_wg_model_input(
                    eeg[train_idx], hbo[train_idx], hbr[train_idx], train_labels
                )
                eval_data = build_wg_model_input(
                    eeg[eval_idx], hbo[eval_idx], hbr[eval_idx], eval_labels
                )
                folds[f"fold_{fold + 1}"] = run_fold(
                    task, subject, fold, train_data, eval_data, output_dirs
                )
        else:
            with np.load(path, allow_pickle=True) as data:
                eeg_all = data["eeg"]
                fnirs_all = np.transpose(data["fnirs"], (0, 1, 5, 2, 3, 4))
                labels_all = np.argmax(data["label"], axis=1)

            n_trials = labels_all.shape[0] // WINDOWS_PER_TRIAL
            trial_labels = labels_all.reshape(n_trials, WINDOWS_PER_TRIAL)[:, 0]
            kf = KFold(n_splits=10, shuffle=True, random_state=SEED)
            for fold, (train_trials, eval_trials) in enumerate(kf.split(np.arange(n_trials))):
                if debug and fold > 0:
                    break
                train_idx = trial_to_sample_indices(train_trials)
                eval_idx = trial_to_sample_indices(eval_trials)
                print(
                    f"[{task}][{subject}][fold {fold + 1}] "
                    f"train_trials={len(train_trials)} eval_trials={len(eval_trials)} "
                    f"labels={np.bincount(trial_labels[train_trials], minlength=2).tolist()}"
                )
                train_data = (
                    eeg_all[train_idx], fnirs_all[train_idx], labels_all[train_idx]
                )
                eval_data = (
                    eeg_all[eval_idx], fnirs_all[eval_idx], labels_all[eval_idx]
                )
                folds[f"fold_{fold + 1}"] = run_fold(
                    task, subject, fold, train_data, eval_data, output_dirs
                )

        all_results[subject] = folds
        subject_acc = np.mean([x["class_acc"] for x in folds.values()])
        print(f"[{task}][{subject}] 10-fold mean acc={subject_acc:.4f}")

        # Save immediately after each subject so completed results are not
        # lost if the long-running experiment is interrupted later.
        intermediate_path = os.path.join(
            output_dirs["results"], "intermediate_results.xlsx"
        )
        save_results(all_results, intermediate_path, task)
        print(f"[{task}] intermediate results saved to: {intermediate_path}")

    result_path = os.path.join(output_dirs["results"], "final_results.xlsx")
    save_results(all_results, result_path, task)
    print(f"[{task}] results saved to: {result_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["WG", "MA", "MI", "ALL"], default="ALL")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    tasks = ("WG", "MA", "MI") if args.task == "ALL" else (args.task,)
    for task in tasks:
        run_task(task, debug=args.debug)


if __name__ == "__main__":
    main()
