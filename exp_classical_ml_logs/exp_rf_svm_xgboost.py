##########################################################
## EXPERIENCE 2 : Comparison with classical ML models
##########################################################


# ======================================================
# IMPORTS
# ======================================================

import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from scipy.stats import skew, kurtosis

from dataloader import dataloader


# ======================================================
# CONFIGURATION
# ======================================================

SUBJECTS = range(1, 10)   # sujets 1 à 9
N_ESTIMATORS = 300
RANDOM_STATE = 42

# ======================================================
# FEATURES EXTRACTION 
# ======================================================

def extract_features_epoch(epoch):
    """ Extrait les features statistiques de base d'un essai EEG """
    feats = []
    for ch in epoch:
        feats.extend([
            np.mean(ch),
            np.var(ch),
            np.sqrt(np.mean(ch**2)),
            skew(ch),
            kurtosis(ch)
        ])
    return np.array(feats)

def extract_features_dataset(X):
    """ Applique l'extraction de features à tous les essais """
    return np.vstack([extract_features_epoch(x) for x in X])


# ======================================================
# DATA LOADING
# ======================================================

def get_aggregated_train_features():
    """ Regroupe les training features de tous les patients """
    X_list, y_list = [], []

    for subject_id in tqdm(SUBJECTS, desc="Loading train data"):
        X_train, y_train, _, _ = dataloader(subject_id)

        if X_train.shape[2] < X_train.shape[1]:
            X_train = X_train.transpose(0,2,1)

        X_feat = extract_features_dataset(X_train)

        X_list.append(X_feat)
        y_list.append(y_train)

    X_all = np.vstack(X_list)
    y_all = np.concatenate(y_list)

    print("Train features shape:", X_all.shape)
    return X_all, y_all

def evaluate_per_subject(model):
    """ Evalue le modèle sur chaque sujet séparément """
    accs = {}

    for subject_id in SUBJECTS:
        _, _, X_test, y_test = dataloader(subject_id)

        if X_test.shape[2] < X_test.shape[1]:
            X_test = X_test.transpose(0,2,1)

        X_feat = extract_features_dataset(X_test)
        y_pred = model.predict(X_feat)

        acc = np.mean(y_pred == y_test)*100
        accs[subject_id] = acc

    return accs


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    X_train_all, y_train_all = get_aggregated_train_features()

    models = {
        "Random Forest": Pipeline([("scaler", StandardScaler()),
         ("rf", RandomForestClassifier(n_estimators=N_ESTIMATORS,random_state=RANDOM_STATE,n_jobs=-1))
        ]),

        "SVM": Pipeline([("scaler", StandardScaler()),
         ("svm", SVC(kernel="rbf",C=1.0,gamma="scale",random_state=RANDOM_STATE))
        ]),

        "XGBoost": Pipeline([("scaler", StandardScaler()),
         ("xgb", XGBClassifier(
                objective="multi:softmax",
                num_class=4,
                n_estimators=N_ESTIMATORS,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                eval_metric="mlogloss"))
        ])
    }

    results_rows = []

    for name, model in models.items():
        print(f"\n=== {name.upper()} ===")
        model.fit(X_train_all, y_train_all)
        accs = evaluate_per_subject(model)

        for s, acc in accs.items():
            print(f"Subject {s}: {acc:.2f}%")
            results_rows.append({
                "Model": name,
                "Subject": s,
                "Accuracy": acc
            })

        mean_acc = np.mean(list(accs.values()))
        print(f"Mean accuracy: {mean_acc:.2f}%")

        results_rows.append({
            "Model": name,
            "Subject": "Mean",
            "Accuracy": mean_acc
        })

df_results = pd.DataFrame(results_rows)
df_results.to_csv("results_exp3_rf_svm_xgboost.csv", index=False)

print("\nResults saved to results_classical_ml.csv")
