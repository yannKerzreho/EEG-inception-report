import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from braindecode.models import EEGInceptionMI
import random

# Vos modules
from dataloader import dataloader
from data_augment import GaussianNoise, HundredHzNoise

# ==========================================
# CONFIGURATION
# ==========================================
SUBJECTS = range(1, 10)
N_EPOCHS = 200        # Comme demandé
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "exp3_logs"

# Paramètres fixes
FIXED_N_FILTERS = 16  # On fixe le modèle à 16 filtres
EVAL_INTERVAL = 5     # Évaluer toutes les X époques
N_EVAL_SAMPLES = 200  # Taille du subset de validation (200 est safe pour la VRAM)

# 5 Graines pour la robustesse statistique
SEEDS = [1,2,3] 

os.makedirs(SAVE_DIR, exist_ok=True)

# --- Fonctions Utilitaires ---

def set_seed(seed):
    """Fixe la graine pour la reproductibilité."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_aggregated_dataset():
    """Charge et concatène les données de TOUS les sujets."""
    print("Chargement et fusion des données de tous les sujets...")
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for subject_id in tqdm(SUBJECTS, desc="Loading Data"):
        try:
            xt, yt, xe, ye = dataloader(subject_id)
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
    
    return X_train_all, y_train_all, X_test_all, y_test_all

def train_model(X_train, y_train, X_test, y_test, augmenter=None, desc="Training"):
    # Dataset
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Modèle FIXÉ à n_filters=16
    model = EEGInceptionMI(
        n_chans=22,
        n_outputs=4,
        n_times=X_train.shape[2], 
        sfreq=250,
        n_filters=FIXED_N_FILTERS # <--- C'est ici qu'on applique votre contrainte
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    losses = []
    val_accuracies = [] 
    
    model.train()
    progress_bar = tqdm(range(1, N_EPOCHS + 1), desc=desc, unit="ep", leave=False)
    
    for epoch in progress_bar:
        epoch_loss = 0
        batch_count = 0
        
        for Xb, yb in loader:
            # Application de l'augmentation si présente
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
        
        # Eval subset périodique
        if epoch % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                idxs = torch.randperm(X_test.size(0))[:N_EVAL_SAMPLES]
                X_sub = X_test[idxs]
                y_sub = y_test[idxs]
                
                out_test = model(X_sub)
                _, pred = torch.max(out_test, 1)
                acc = (pred == y_sub).sum().item() / N_EVAL_SAMPLES * 100
                val_accuracies.append(acc)
            model.train()
            
    return model, losses, val_accuracies

def evaluate_per_subject(model, subject_list, device):
    """Évalue le modèle global sur chaque sujet individuellement."""
    model.eval()
    subject_accuracies = {}
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
            except:
                subject_accuracies[subject_id] = np.nan
    return subject_accuracies

def plot_history_with_shaded(history_dict, title, ylabel, save_path, x_axis_step=1):
    """
    Trace Moyenne +/- MinMax pour chaque méthode.
    """
    plt.figure(figsize=(12, 7))
    
    # Couleurs par méthode d'augmentation
    colors = {'Baseline': 'black', 'FreqAug': 'blue', 'GausAug': 'red'}
    
    for name, runs_list in history_dict.items():
        data = np.array(runs_list)
        
        mean_curve = np.mean(data, axis=0)
        min_curve = np.min(data, axis=0)
        max_curve = np.max(data, axis=0)
        
        epochs = np.arange(1, len(mean_curve) + 1) * x_axis_step
        color = colors.get(name, 'gray')
        
        plt.plot(epochs, mean_curve, label=f"{name} (Mean)", color=color, linewidth=2)
        plt.fill_between(epochs, min_curve, max_curve, color=color, alpha=0.15)
        
    plt.title(title)
    plt.xlabel("Époques")
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"=== Expérience Augmentation (n={FIXED_N_FILTERS}) Multi-Seed ===")
    
    # 1. Chargement Unique
    X_train, y_train, X_test, y_test = get_aggregated_dataset()
    
    # Envoi GPU massif
    X_train_all = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_all = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
    X_test_all = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_test_all = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

    del X_train, y_train, X_test, y_test

    # 2. Calibration Unique des Bruits (avant les boucles)
    # On calcule le SD sur le dataset global une fois pour toutes
    print("\nCalibration des bruits...")
    temp_aug = HundredHzNoise(X_train_all, p=1.0) # p=1 juste pour forcer le calcul init
    noise_sd_ref = temp_aug.noise_sd # On récupère le vecteur SD
    print(f"Noise SD ref (3 premiers) : {noise_sd_ref.flatten()[:3]}")

    # Structures de stockage globales
    METHODS = ["Baseline", "FreqAug", "GausAug"]
    global_losses = {k: [] for k in METHODS}
    global_val_accs = {k: [] for k in METHODS}
    
    final_results_flat = []

    # 3. Boucle Méthodes -> Seeds
    for method_name in METHODS:
        print(f"\n>>> Méthode : {method_name}")
        
        for seed in SEEDS:
            set_seed(seed)
            
            # A. Instanciation de l'Augmenter (propre à chaque seed pour le random)
            augmenter = None
            if method_name == "FreqAug":
                augmenter = HundredHzNoise(X_train_all, p=0.9).to(DEVICE)
            elif method_name == "GausAug":
                # On utilise le SD de référence calculé plus haut
                augmenter = GaussianNoise(sd=noise_sd_ref, p=0.9).to(DEVICE)
            
            # B. Entraînement
            desc = f"{method_name}|Seed={seed}"
            model, losses, val_accs = train_model(
                X_train_all, y_train_all, 
                X_test_all, y_test_all, 
                augmenter=augmenter, 
                desc=desc
            )
            
            # Stockage Historique
            global_losses[method_name].append(losses)
            global_val_accs[method_name].append(val_accs)
            
            # C. Évaluation Finale par Sujet
            accuracies = evaluate_per_subject(model, SUBJECTS, DEVICE)
            
            avg_acc = np.mean([v for v in accuracies.values() if not np.isnan(v)])
            print(f"   Run Seed {seed} -> Avg Acc: {avg_acc:.2f}%")
            
            for subj, acc in accuracies.items():
                final_results_flat.append({
                    "Method": method_name,
                    "Seed": seed,
                    "Subject": subj,
                    "Accuracy": acc
                })
            
            # Sauvegarde dernier modèle
            if seed == SEEDS[-1]:
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"Model_{method_name}_n{FIXED_N_FILTERS}_last.pth"))

    # ==========================================
    # 4. RÉSULTATS & VISUALISATION
    # ==========================================
    
    print("\nGénération des graphiques...")
    plot_history_with_shaded(
        global_losses, f"Loss (n_filters={FIXED_N_FILTERS})", "Loss",
        os.path.join(SAVE_DIR, "plot_losses_shaded.png"), x_axis_step=1
    )
    
    plot_history_with_shaded(
        global_val_accs, f"Validation Accuracy (n_filters={FIXED_N_FILTERS})", "Accuracy (%)",
        os.path.join(SAVE_DIR, "plot_acc_shaded.png"), x_axis_step=EVAL_INTERVAL
    )

    # Tableau Récapitulatif
    df = pd.DataFrame(final_results_flat)
    df.to_csv(os.path.join(SAVE_DIR, "raw_results_all_seeds.csv"), index=False)
    
    # Agrégation Globale
    df_global = df.groupby(["Method", "Seed"])["Accuracy"].mean().reset_index()
    summary_global = df_global.groupby("Method")["Accuracy"].agg(
        ['mean', 'std', 'min', 'max']
    ).reset_index()
    
    print("\n=== PERFORMANCES GLOBALES (Moyenne des 5 Seeds) ===")
    print(summary_global.round(2))
    summary_global.to_csv(os.path.join(SAVE_DIR, "summary_global.csv"))