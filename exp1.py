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
N_EPOCHS = 11
BATCH_SIZE = 64
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "exp_global_logs"
EVAL_INTERVAL = 10 # Évaluer toutes les 10 époques

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Fonctions ---

def get_aggregated_dataset():
    """Charge et concatène les données de TOUS les sujets."""
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for subject_id in tqdm(SUBJECTS, desc="Loading Data"):
        try:
            xt, yt, xe, ye = dataloader(subject_id)
            # Transposition (Time, Chan) -> (Chan, Time) pour Braindecode
            if xt.shape[2] < xt.shape[1]: 
                xt = xt.transpose(0, 2, 1)
                xe = xe.transpose(0, 2, 1)
            X_train_list.append(xt)
            y_train_list.append(yt)
            X_test_list.append(xe)
            y_test_list.append(ye)
        except Exception as e:
            print(f"Warning: Sujet {subject_id} ignoré ({e})")

    X_train_all = np.concatenate(X_train_list, axis=0)
    y_train_all = np.concatenate(y_train_list, axis=0)
    X_test_all = np.concatenate(X_test_list, axis=0)
    y_test_all = np.concatenate(y_test_list, axis=0)
    
    print(f"Dataset Global Train: {X_train_all.shape}")
    print(f"Dataset Global Test : {X_test_all.shape}")
    return X_train_all, y_train_all, X_test_all, y_test_all

def train_model(X_train, y_train, X_test, y_test, augmenter=None, desc="Training"):
    # 1. CHARGEMENT GPU (Train ET Test)
    print("  -> Transfert Train/Test sur GPU...")
    X_train_gpu = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_gpu = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_gpu = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_gpu = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    dataset = TensorDataset(X_train_gpu, y_train_gpu)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

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
    progress_bar = tqdm(range(1, N_EPOCHS + 1), desc=desc, unit="epoch")
    
    for epoch in progress_bar:
        epoch_loss = 0
        batch_count = 0
        
        # --- Boucle d'entraînement ---
        for Xb, yb in loader:            
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
        
        # --- Évaluation Périodique ---
        log_msg = {"Loss": f"{avg_loss:.4f}"}
        
        if epoch % EVAL_INTERVAL == 0:
            model.eval() # Mode éval (fige Dropout/BatchNorm)
            with torch.no_grad():
                out_test = model(X_test_gpu)
                _, pred = torch.max(out_test, 1)
                acc = (pred == y_test_gpu).sum().item() / y_test_gpu.size(0) * 100
            print(f"Current Test Acc: {acc:.2f}% at epoch {epoch}")
            model.train() # Retour en mode train
            log_msg["Test Acc"] = f"{acc:.2f}%"
            
        progress_bar.set_postfix(log_msg)
    
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
                
                # Ici on charge petit à petit pour éviter de saturer si on a peu de RAM
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
    plt.figure(figsize=(10, 6))
    styles = {'Baseline': '-', 'FreqAug': '--', 'GausAug': '-.'}
    colors = {'Baseline': 'black', 'FreqAug': 'blue', 'GausAug': 'red'}
    for name, loss_list in losses_dict.items():
        epochs = range(1, len(loss_list) + 1)
        plt.plot(epochs, loss_list, label=name, 
                 linestyle=styles.get(name, '-'), color=colors.get(name))
    plt.title("Évolution de la Loss")
    plt.xlabel("Époques")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"=== Expérience Globale (Train Global -> Test par Sujet) ===")
    
    # 1. Chargement Train ET Test Globaux
    X_train_all, y_train_all, X_test_all, y_test_all = get_aggregated_dataset()
    
    # 2. Calibration des bruits
    print("\nCalibration des bruits...")
    # Pour calibration, on utilise une version CPU ou temporaire pour ne pas saturer GPU
    X_tensor_all = torch.tensor(X_train_all, dtype=torch.float32)
    
    freq_augm = HundredHzNoise(X_tensor_all, p=0.9)
    scale_val = freq_augm.relative_scale
    if isinstance(scale_val, torch.Tensor): scale_val = scale_val.item()
        
    print(f"Scale global : {scale_val:.5f}")
    
    # Init Augmenters et Envoi GPU
    gaus_augm = GaussianNoise(relative_scale=scale_val, p=0.9).to(DEVICE)
    freq_augm = freq_augm.to(DEVICE)

    # 3. Boucle d'Expérience
    conditions = {
        "Baseline": None,
        "FreqAug": freq_augm,
        "GausAug": gaus_augm
    }
    
    detailed_results = []
    all_losses_history = {}

    for name, augmenter in conditions.items():
        print(f"\n=== Condition : {name} ===")
        
        # A. Entraînement Global (Avec monitoring Test)
        # On passe X_test_all et y_test_all pour l'eval interne
        model, losses = train_model(
            X_train_all, y_train_all, 
            X_test_all, y_test_all, 
            augmenter, desc=f"Train {name}"
        )
        
        all_losses_history[name] = losses
        
        # B. Évaluation Finale Détaillée PAR SUJET
        accuracies = evaluate_per_subject(model, SUBJECTS, DEVICE)
        
        avg_acc = np.mean(list(accuracies.values()))
        print(f" >> Moyenne Globale Finale pour {name} : {avg_acc:.2f}%")
        
        for subj, acc in accuracies.items():
            detailed_results.append({
                "Method": name,
                "Subject": subj,
                "Accuracy": acc
            })
            
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"Global_{name}.pth"))

    # 4. Tracé et Sauvegarde
    plot_path = os.path.join(SAVE_DIR, "training_losses_comparison.png")
    plot_losses(all_losses_history, plot_path)

    with open(os.path.join(SAVE_DIR, "losses_history.pkl"), "wb") as f:
        pickle.dump(all_losses_history, f)

    df_detailed = pd.DataFrame(detailed_results)
    pivot_table = df_detailed.pivot(index="Subject", columns="Method", values="Accuracy")
    pivot_table["Mean"] = pivot_table.mean(axis=1)
    pivot_table.loc['Total Mean'] = pivot_table.mean()

    print("\n=== TABLEAU RÉCAPITULATIF (Accuracy %) ===")
    print(pivot_table.round(2))
    
    pivot_table.to_csv(os.path.join(SAVE_DIR, "detailed_global_results.csv"))