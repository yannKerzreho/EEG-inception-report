import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import random
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
SEEDS = [1, 2, 3] 
N_EPOCHS = 200
BATCH_SIZE = 128
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "exp2_logs_multiseed_shaded"
EVAL_INTERVAL = 5
N_EVAL_SAMPLES = 500

os.makedirs(SAVE_DIR, exist_ok=True)

CONDITIONS = {
    "4": 4,
    "8": 8,
    "16": 16,
    "32": 32,
}

def set_seed(seed):
    """Fixe la graine pour tous les modules aléatoires."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_aggregated_dataset():
    """Charge et concatène les données de TOUS les sujets."""
    print("Chargement et fusion des données de tous les sujets...")
    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    
    for subject_id in tqdm(SUBJECTS, desc="Loading Data"):
        try:
            xt, yt, xe, ye = dataloader(subject_id)
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

def train_model(X_train, y_train, X_test, y_test, n_filters, augmenter=None, desc="Training"):

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = EEGInceptionMI(
        n_chans=22,
        n_outputs=4,
        n_times=X_train.shape[2], 
        sfreq=250,
        n_filters=n_filters
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    losses = []
    val_accuracies = [] 
    
    model.train()
    progress_bar = tqdm(range(1, N_EPOCHS + 1), desc=desc, unit="ep", leave=False)
    current_acc_str = "?"
    
    for epoch in progress_bar:
        epoch_loss = 0
        batch_count = 0
        
        # --- TRAIN LOOP ---
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
        
        # --- EVAL LOOP ---
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
                current_acc_str = f"{acc:.2f}%"
            model.train()
            
        progress_bar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": current_acc_str})
    
    return model, losses, val_accuracies

def evaluate_per_subject(model, subject_list, device):
    """Évalue le modèle global sur chaque sujet individuellement."""
    model.eval()
    subject_accuracies = {}
    with torch.no_grad():
        for subject_id in subject_list:
            try:
                _, _, X_test, y_test = dataloader(subject_id)
                X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
                
                out = model(X_tensor)
                _, predicted = torch.max(out, 1)
                
                correct = (predicted == y_tensor).sum().item()
                total = y_tensor.size(0)
                acc = 100 * correct / total
                subject_accuracies[subject_id] = acc
            except Exception as e:
                subject_accuracies[subject_id] = np.nan
    return subject_accuracies

def plot_history_shaded(history_dict, title, ylabel, save_path, x_axis_step=1):
    """
    Trace Mean avec une zone ombragée (Min-Max) sur les différentes graines.
    """
    plt.figure(figsize=(10, 6))
    
    colors = {'4': 'black', '8': 'blue', '16': 'red', '32': 'green'}
    
    # 1. Regrouper les données par condition (ignorer le _sX)
    # structure: grouped_data['4'] = [ [liste_loss_seed1], [liste_loss_seed2], ... ]
    grouped_data = {}
    
    for key, values in history_dict.items():
        cond_name = key.split('_s')[0]
        if cond_name not in grouped_data:
            grouped_data[cond_name] = []
        grouped_data[cond_name].append(values)
        
    # 2. Calculer les stats et tracer
    for name, lists_of_values in grouped_data.items():
        # Convertir en tableau numpy (n_seeds, n_epochs)
        # Attention : suppose que toutes les seeds ont le même nombre d'époques
        arr = np.array(lists_of_values)
        
        mean_curve = np.mean(arr, axis=0)
        min_curve = np.min(arr, axis=0)
        max_curve = np.max(arr, axis=0)
        
        epochs = np.arange(1, len(mean_curve) + 1) * x_axis_step
        col = colors.get(name, 'gray')
        
        # Tracer la moyenne
        plt.plot(epochs, mean_curve, label=f"{name} filters (Mean)", color=col, linewidth=2)
        
        # Tracer la zone d'ombre (Min - Max)
        plt.fill_between(epochs, min_curve, max_curve, color=col, alpha=0.2, label=f"{name} Range (Min-Max)")

    plt.title(title)
    plt.xlabel("Époques")
    plt.ylabel(ylabel)
    # Déduplication des labels dans la légende si besoin, sinon affichage standard
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"=== Expérience Multi-Seed {SEEDS} (Shaded Graphs & Full Stats) ===")
    
    # 1. Chargement Data Global
    X_train_np, y_train_np, X_test_np, y_test_np = get_aggregated_dataset()
    X_train_all = torch.tensor(X_train_np, dtype=torch.float32).to(DEVICE)
    y_train_all = torch.tensor(y_train_np, dtype=torch.long).to(DEVICE)
    X_test_all = torch.tensor(X_test_np, dtype=torch.float32).to(DEVICE)
    y_test_all = torch.tensor(y_test_np, dtype=torch.long).to(DEVICE)
    del X_train_np, y_train_np, X_test_np, y_test_np
    
    detailed_results = []
    all_losses_history = {}
    all_testacc_history = {}

    # 2. Boucle Seed / Conditions
    for seed in SEEDS:
        print(f"\n>>> DÉBUT SEED {seed} <<<")
        set_seed(seed)
        
        for name, n_filters in CONDITIONS.items():        
            # A. Train
            model, losses, val_accs = train_model(
                X_train_all, y_train_all, 
                X_test_all, y_test_all, 
                n_filters, None, 
                desc=f"S{seed}|Filter={name}"
            )
            
            key_id = f"{name}_s{seed}"
            all_losses_history[key_id] = losses
            all_testacc_history[key_id] = val_accs
            
            # B. Eval per subject
            accuracies = evaluate_per_subject(model, SUBJECTS, DEVICE)
            
            for subj, acc in accuracies.items():
                detailed_results.append({
                    "Seed": seed,
                    "n_filters": name,
                    "Subject": subj,
                    "Accuracy": acc
                })
            
            # Save Model (optionnel, prend de la place)
            # torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"Global_{name}_s{seed}.pth"))

    # 3. Sauvegarde CSV Historique Brut
    pd.DataFrame(all_losses_history).to_csv(os.path.join(SAVE_DIR, "history_losses_raw.csv"))
    pd.DataFrame(all_testacc_history).to_csv(os.path.join(SAVE_DIR, "history_val_accuracy_raw.csv"))

    # 4. Tracé des Courbes (Shaded: Mean + Min/Max)
    print("Génération des graphiques...")
    plot_history_shaded(all_losses_history, "Loss (Mean & Min-Max)", "Loss", 
                        os.path.join(SAVE_DIR, "plot_losses_shaded.png"), x_axis_step=1)
                 
    plot_history_shaded(all_testacc_history, "Val Accuracy (Mean & Min-Max)", "Accuracy (%)", 
                        os.path.join(SAVE_DIR, "plot_val_accuracy_shaded.png"), x_axis_step=EVAL_INTERVAL)

    # 5. Tableau Final Détaillé (Mean, SD, Max, Min)
    df_detailed = pd.DataFrame(detailed_results)
    df_detailed.to_csv(os.path.join(SAVE_DIR, "detailed_results_all_seeds.csv"), index=False)

    print("Calcul des statistiques finales...")
    
    # Agrégation par (Subject, n_filters)
    grouped = df_detailed.groupby(['Subject', 'n_filters'])['Accuracy'].agg(['mean', 'std', 'max', 'min'])
    
    # Fonction de formatage pour l'affichage : "Mean ± SD (Min - Max)"
    def format_stats(row):
        return f"{row['mean']:.2f} ±{row['std']:.2f} ({row['min']:.2f}-{row['max']:.2f})"
    
    grouped['stats_str'] = grouped.apply(format_stats, axis=1)
    
    # Pivot pour affichage
    pivot_table = grouped.reset_index().pivot(index="Subject", columns="n_filters", values="stats_str")
    
    # Ajout de la ligne GLOBAL AVERAGE (Moyenne sur tous les sujets pour chaque filtre)
    # On refait le calcul sur la base brute globale
    total_grouped = df_detailed.groupby(['n_filters'])['Accuracy'].agg(['mean', 'std', 'max', 'min'])
    total_row = total_grouped.apply(format_stats, axis=1)
    
    pivot_table.loc['GLOBAL AVG'] = total_row

    print("\n=== TABLEAU RÉCAPITULATIF : Mean ± SD (Min - Max) ===")
    print(pivot_table)
    
    pivot_table.to_csv(os.path.join(SAVE_DIR, "summary_table_stats.csv"))