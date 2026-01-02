##########################################################
## EXPERIENCE : Comparison with CNN 1D
##########################################################

# ======================================================
# IMPORTS
# ======================================================

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import random
import pandas as pd

from braindecode.models import EEGInceptionMI

from dataloader import dataloader


# ======================================================
# CONFIGURATION
# ======================================================

SUBJECTS = range(1, 10)        # Subjects 1 to 9
N_EPOCHS = 50
BATCH_SIZE = 64
LR = 1e-3
N_RUNS = 3

# MODEL_TO_RUN = "CNN-1D"
# MODEL_TO_RUN = "EEG-Inception"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "exp_cnn_eeg-inception_logs"
os.makedirs(SAVE_DIR, exist_ok=True)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# ======================================================
# MODEL SELECTION
# ======================================================

print("\nSelect model to run:")
print("1/ CNN 1D")
print("2/ EEG-Inception")

choice = input("Enter 1 or 2: ").strip()

if choice == "1":
    MODEL_TO_RUN = "CNN-1D"
elif choice == "2":
    MODEL_TO_RUN = "EEG-Inception"
else:
    raise ValueError("Invalid choice. Please enter 1 or 2.")


# CSV_PATH = os.path.join(SAVE_DIR, "results_cnn_eeg-inception_models.csv")
CSV_PATH = os.path.join(SAVE_DIR, f"results_{MODEL_TO_RUN}.csv")

def get_model(model_name, X_train_all):
    """ Choose the model """
    if model_name == "CNN-1D":
        return SimpleCNN1D(
            n_chans=22,
            n_classes=4
        )

    elif model_name == "EEG-Inception":
        return EEGInceptionMI(
            n_chans=22,
            n_outputs=4,
            n_times=X_train_all.shape[2],
            sfreq=250,
            n_filters=16
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")



# ======================================================
# DATA LOADING
# ======================================================

def get_aggregated_dataset():
    """ Load and concatenates data from all subjects """
    print("Loading and merging data from all subjects...")
    X_train_list, y_train_list = [], []
    # On n'a besoin que du train global ici

    for subject_id in tqdm(SUBJECTS, desc="Loading Data"):
        try:
            xt, yt, _, _ = dataloader(subject_id)
            if xt.shape[2] < xt.shape[1]:
                xt = xt.transpose(0, 2, 1)
            X_train_list.append(xt)
            y_train_list.append(yt)
        except Exception as e:
            print(f"Warning: Sujet {subject_id} ignoré ({e})")

    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    print(f"Dataset Global Train: {X_train_all.shape}")
    return X_train_all, y_train_all


# ======================================================
# MODELS CNN 1D
# ======================================================

class SimpleCNN1D(nn.Module):
    """ Implement a simple CNN 1D """
    def __init__(self, n_chans=22, n_classes=4):
        super().__init__()

        self.conv1 = nn.Conv1d(n_chans, 32, kernel_size=25, padding=12)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=25, padding=12)

        self.pool = nn.MaxPool1d(2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.global_pool(x)
        x = x.squeeze(-1)
        return self.fc(x)


# ======================================================
# TRAIN / EVAL
# ======================================================

def train_model(model,X_train,y_train,augmenter=None,epochs=N_EPOCHS):
    """ Train a NN on the training data """
    model.to(DEVICE)
    model.train()

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    # batch_size = 8 if isinstance(model, EEGInceptionMI) else 64

    loader = DataLoader(
        dataset,
        # batch_size=batch_size,
        batch_size = BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = []

    for epoch in tqdm(range(epochs), desc="Training", unit="epoch"):
        epoch_loss = 0.0

        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            # if augmenter is not None:
            #     Xb = augmenter(Xb)


            optimizer.zero_grad()

            out = model(Xb)

            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / len(loader))

    return model, losses


def evaluate_per_subject(model, subject_list, device):
    """ Evaluates the overall model on each subject individually """
    model.eval()
    subject_accuracies = {}
    print("\n--- Évaluation détaillée par sujet ---")
    with torch.no_grad():
        for subject_id in subject_list:
            try:
                _, _, X_test, y_test = dataloader(subject_id)
                if X_test.shape[2] < X_test.shape[1]:
                    X_test = X_test.transpose(0, 2, 1)

                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

                out = model(X_tensor)

                _, predicted = torch.max(out, 1)

                correct = (predicted == y_tensor).sum().item()
                total = y_tensor.size(0)
                acc = 100 * correct / total
                subject_accuracies[subject_id] = acc
            except Exception as e:
                print(f"Erreur eval sujet {subject_id}: {e}")
                subject_accuracies[subject_id] = np.nan
    return subject_accuracies

def plot_loss_curve(losses, model_name, save_dir):
    """ Trace and save the loss curve for a model """
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(losses) + 1)

    plt.plot(epochs, losses, label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss (CrossEntropy)")
    plt.title(f"Training Loss – {model_name}")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"loss_{model_name}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Loss curve saved to: {save_path}")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    print("=== EXPERIMENT ===")
    print(f"Model: {MODEL_TO_RUN}")
    print(f"Device: {DEVICE}")

    # 1. Load data once
    X_train_all, y_train_all = get_aggregated_dataset()

    # 2. Determine number of runs
    # if MODEL_TO_RUN == "CNN-1D":
    #     runs_to_do = N_RUNS
    # else:
    #     runs_to_do = 1
    
    runs_to_do = N_RUNS

    all_run_results = []
    all_run_subject_accs = []

    # 3. Multi-run loop
    for run_id in range(1, runs_to_do + 1):

        print(f"\n--- {MODEL_TO_RUN} | Run {run_id}/{runs_to_do} ---")

        set_seed(run_id)

        model = get_model(MODEL_TO_RUN, X_train_all)

        start_time = time.time()

        trained_model, losses = train_model(
            model,
            X_train_all,
            y_train_all, 
            epochs=N_EPOCHS
        )

        train_time = time.time() - start_time

        accs = evaluate_per_subject(trained_model, SUBJECTS, DEVICE)
        all_run_subject_accs.append(accs)
        
        mean_acc = np.mean(list(accs.values()))

        print(f"Mean accuracy: {mean_acc:.2f}% | Time: {train_time:.1f}s")

        all_run_results.append({
            "Run": run_id,
            "MeanAccuracy": mean_acc,
            "TrainingTimeSeconds": train_time
        })

        # plot_loss_curve(losses, f"{MODEL_TO_RUN}_run{run_id}", SAVE_DIR)

    # 4. Aggregate results
    mean_acc = np.mean([r["MeanAccuracy"] for r in all_run_results])
    std_acc = np.std([r["MeanAccuracy"] for r in all_run_results])

    mean_time = np.mean([r["TrainingTimeSeconds"] for r in all_run_results])
    std_time = np.std([r["TrainingTimeSeconds"] for r in all_run_results])

    print("\n=== FINAL RESULTS ===")
    print(f"Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"Training time: {mean_time:.1f}s ± {std_time:.1f}s")

    # 5. Save CSV
    rows = []
    for run_idx, accs in enumerate(all_run_subject_accs, start=1):
        row = {"Run": run_idx}

        for subj in SUBJECTS:
            row[f"Subj{subj}"] = accs[subj]

        row["Mean_Run"] = np.mean(list(accs.values()))
        rows.append(row)

    mean_subject_row = {"Run": "Mean_Subjects"}

    for subj in SUBJECTS:
        mean_subject_row[f"Subj{subj}"] = np.mean(
            [run_accs[subj] for run_accs in all_run_subject_accs]
        )

    mean_subject_row["Mean_Run"] = np.mean(
        [row["Mean_Run"] for row in rows]
    )

    rows.append(mean_subject_row)
    # Std per subject across runs
    std_subject_row = {"Run": "Std_Subjects"}

    for subj in SUBJECTS:
        std_subject_row[f"Subj{subj}"] = np.std(
            [run_accs[subj] for run_accs in all_run_subject_accs]
        )

    std_subject_row["Mean_Run"] = np.std(
        [row["Mean_Run"] for row in rows if isinstance(row["Run"], int)]
    )

    rows.append(std_subject_row)


    for row in rows:
        if isinstance(row["Run"], int):
            row["Mean_Run_PM"] = f"{row['Mean_Run']:.2f}"
        elif row["Run"] == "Mean_Subjects":
            row["Mean_Run_PM"] = f"{row['Mean_Run']:.2f} +/- {std_subject_row['Mean_Run']:.2f}"
        else:
            row["Mean_Run_PM"] = ""


    df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, f"exp_{MODEL_TO_RUN}.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nFinal CSV saved to: {csv_path}")

