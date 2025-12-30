import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from braindecode.models import EEGInceptionMI

# Vos modules
from dataloader import dataloader
from data_augment import GaussianNoise, HundredHzNoise

# ==========================================
# CONFIGURATION
# ==========================================
SUBJECTS = range(1, 10)
N_EPOCHS = 100
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "exp_global_logs"

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Fonctions existantes ---

def get_aggregated_dataset():
    """Charge et concatène les données de TOUS les sujets."""
    print("Chargement et fusion des données de tous les sujets...")
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

def train_model(X_train, y_train, augmenter=None, desc="Training"):
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = EEGInceptionMI(
        n_chans=22,
        n_outputs=4,
        n_times=X_train.shape[2], 
        sfreq=250,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    losses = []
    
    model.train()
    progress_bar = tqdm(range(N_EPOCHS), desc=desc, unit="epoch")
    
    for epoch in progress_bar:
        epoch_loss = 0
        batch_count = 0
        
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            if augmenter is not None:
                Xb = augmenter(Xb)

            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}"})
    
    return model, losses

def evaluate_per_subject(model, subject_list, device):
    """Évalue le modèle global sur chaque sujet individuellement."""
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

def plot_losses(losses_dict, save_path):
    """Trace et sauvegarde les courbes de loss."""
    plt.figure(figsize=(10, 6))
    
    styles = {'Baseline': '-', 'FreqAug': '--', 'GausAug': '-.'}
    colors = {'Baseline': 'black', 'FreqAug': 'blue', 'GausAug': 'red'}
    
    for name, loss_list in losses_dict.items():
        epochs = range(1, len(loss_list) + 1)
        plt.plot(epochs, loss_list, label=name, 
                 linestyle=styles.get(name, '-'), color=colors.get(name))
        
    plt.title("Évolution de la Loss d'Entraînement Global")
    plt.xlabel("Époques")
    plt.ylabel("Loss (CrossEntropy)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    print(f"Sauvegarde du graphique dans : {save_path}")
    plt.savefig(save_path, dpi=300)
    plt.close() # Important pour libérer la mémoire

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"=== Expérience Globale (Train Global -> Test par Sujet) ===")
    
    # 1. Chargement Train Global
    X_train_all, y_train_all = get_aggregated_dataset()
    
    # 2. Calibration des bruits
    print("\nCalibration des bruits...")
    X_tensor_all = torch.tensor(X_train_all, dtype=torch.float32)
    
    freq_augm = HundredHzNoise(X_tensor_all, p=0.9)
    scale_val = freq_augm.relative_scale
    print(f"Scale global : {scale_val}")
    gaus_augm = GaussianNoise(relative_scale=scale_val, p=0.9)
    
    freq_augm = freq_augm.to(DEVICE)
    gaus_augm = gaus_augm.to(DEVICE)

    # 3. Boucle d'Expérience
    conditions = {
        "Baseline": None,
        "FreqAug": freq_augm,
        "GausAug": gaus_augm
    }
    
    detailed_results = []
    all_losses_history = {} # Nouveau : Pour stocker les courbes

    for name, augmenter in conditions.items():
        print(f"\n=== Condition : {name} ===")
        
        # A. Entraînement Global
        model, losses = train_model(X_train_all, y_train_all, augmenter, desc=f"Train {name}")
        
        # Stockage de l'historique de loss pour ce modèle
        all_losses_history[name] = losses
        
        # B. Évaluation PAR SUJET
        accuracies = evaluate_per_subject(model, SUBJECTS, DEVICE)
        
        avg_acc = np.mean(list(accuracies.values()))
        print(f" >> Moyenne Globale pour {name} : {avg_acc:.2f}%")
        
        for subj, acc in accuracies.items():
            detailed_results.append({
                "Method": name,
                "Subject": subj,
                "Accuracy": acc
            })
            
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"Global_{name}.pth"))

    # 4. Tracé des Losses (NOUVEAU)
    plot_path = os.path.join(SAVE_DIR, "training_losses_comparison.png")
    plot_losses(all_losses_history, plot_path)

    # 5. Sauvegarde des données brutes de loss (pour refaire des plots plus tard si besoin)
    with open(os.path.join(SAVE_DIR, "losses_history.pkl"), "wb") as f:
        pickle.dump(all_losses_history, f)

    # 6. Tableau Final
    df_detailed = pd.DataFrame(detailed_results)
    pivot_table = df_detailed.pivot(index="Subject", columns="Method", values="Accuracy")
    pivot_table["Mean"] = pivot_table.mean(axis=1)
    pivot_table.loc['Total Mean'] = pivot_table.mean()

    print("\n=== TABLEAU RÉCAPITULATIF (Accuracy %) ===")
    print(pivot_table.round(2))
    
    pivot_table.to_csv(os.path.join(SAVE_DIR, "detailed_global_results.csv"))