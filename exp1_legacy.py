import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import pickle
from tqdm import tqdm
from braindecode.models import EEGInceptionMI

# Imports de vos modules (assurez-vous qu'ils sont accessibles)
from dataloader import dataloader
from data_augment import GaussianNoise, HundredHzNoise # Vos classes définies précédemment

# ==========================================
# CONFIGURATION
# ==========================================
SUBJECTS = range(1, 10) # De 1 à 9
N_EPOCHS = 100
BATCH_SIZE = 16 # Petit batch size pour la régularisation
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "exp1_logs"

os.makedirs(SAVE_DIR, exist_ok=True)

def train_one_model(X_train, y_train, augmenter=None, desc="Train"):
    """
    Entraine un modèle frais sur les données fournies.
    """
    # 1. Préparation des Dataloaders
    # Conversion en Tensor
    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32), 
        torch.tensor(y_train, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Init du Modèle (Frais à chaque appel !)
    model = EEGInceptionMI(
        n_chans=22,
        n_outputs=4,
        n_times=X_train.shape[2], # 1001 ou 313 selon votre découpage
        sfreq=250,
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    losses = []

    # 3. Boucle d'entrainement
    model.train()
    for epoch in range(N_EPOCHS):
        epoch_loss = 0

        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

            # --- AUGMENTATION ---
            if augmenter is not None:
                # L'augmenter gère lui-même le mode .training et la proba
                Xb = augmenter(Xb)
            # --------------------

            optimizer.zero_grad()
            out = model(Xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        losses.append(epoch_loss / len(loader))
    
    return model, losses

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle sur le jeu de test.
    """        
    dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32), 
        torch.tensor(y_test, dtype=torch.long)
    )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)
        out = model(X_test)
        _, predicted = torch.max(out.data, 1)
        correct += (predicted == y_test).sum().item()
            
    return 100 * correct / len(y_test)

if __name__ == "__main__":
    results_summary = [] # Pour le tableau final
    full_logs = {}       # Pour sauvegarder les objets lourds

    print(f"Lancement de l'expérience sur {DEVICE}...")

    for subject_id in tqdm(SUBJECTS, desc="Sujets"):
        print(f"\nTraitement Sujet {subject_id}")
        
        # 1. Chargement des données
        try:
            X_train, y_train, X_test, y_test = dataloader(subject_id)
        except Exception as e:
            print(f"Erreur chargement sujet {subject_id}: {e}")
            continue

        # 2. Préparation des Augmenteurs
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        freq_augm = HundredHzNoise(X_train_tensor, p=0.9).to(DEVICE)
        scale = freq_augm.relative_scale.numpy()
        gaus_augm = GaussianNoise(relative_scale=scale, p=0.9).to(DEVICE)

        print(f"  > Scale Bruit calculé : {scale}")

        # 3. Lancement des 3 conditions
        conditions = {
            "Baseline": None,
            "FreqAug": freq_augm,
            "GausAug": gaus_augm
        }

        for name, augmenter in conditions.items():
            # Entrainement
            model, losses = train_one_model(X_train, y_train, augmenter, desc=name)
            
            # Évaluation
            acc = evaluate_model(model, X_test, y_test)
            
            # Sauvegarde Logique
            res_entry = {
                "Subject": subject_id,
                "Method": name,
                "Accuracy": acc,
                "Final_Loss": losses[-1],
                "Noise_Scale": scale if name != "Baseline" else 0
            }
            results_summary.append(res_entry)
            
            # Sauvegarde Physique (Modèle + Logs détaillés)
            # On sauvegarde le state_dict pour prendre moins de place
            save_name = f"S{subject_id}_{name}"
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{save_name}.pth"))
            full_logs[save_name] = {
                "losses": losses,
                "accuracy": acc
            }
            
            print(f"  [{name}] Acc: {acc:.2f}% | Loss: {losses[-1]:.4f}")

    # ==========================================
    # RÉSULTATS ET SAUVEGARDE
    # ==========================================

    # 1. Création du DataFrame
    df_results = pd.DataFrame(results_summary)

    # 2. Sauvegarde du tableau CSV et des logs complets
    df_results.to_csv(os.path.join(SAVE_DIR, "results_summary.csv"), index=False)
    with open(os.path.join(SAVE_DIR, "full_logs.pkl"), "wb") as f:
        pickle.dump(full_logs, f)

    # 3. Affichage du Tableau Récapitulatif
    print("\n=== RÉSULTATS FINAUX ===")
    # Pivot pour avoir Sujets en lignes et Méthodes en colonnes (plus lisible)
    pivot_table = df_results.pivot(index="Subject", columns="Method", values="Accuracy")
    pivot_table["Mean"] = pivot_table.mean(axis=1) # Moyenne par sujet
    print(pivot_table.round(2))

    print("\n=== MOYENNES GLOBALES ===")
    print(df_results.groupby("Method")["Accuracy"].mean().round(2))