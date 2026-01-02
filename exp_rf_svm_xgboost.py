##########################################################
## EXPERIENCE : Comparison with classical ML models
##########################################################


# ======================================================
# IMPORTS
# ======================================================

import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import os

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

SUBJECTS = range(1, 10)   # subjects 1 to 9
N_ESTIMATORS = 300
RANDOM_STATE = 42

N_RUNS = 3
SAVE_DIR = "exp_classical_ml_logs"
os.makedirs(SAVE_DIR, exist_ok=True)


# ======================================================
# FEATURES EXTRACTION 
# ======================================================

def extract_features_epoch(epoch):
    """ Extract statistical features from an EEG test """
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
    """ Apply feature extraction to all tests """
    return np.vstack([extract_features_epoch(x) for x in X])


# ======================================================
# DATA LOADING
# ======================================================

def get_aggregated_train_features():
    """ Group the training features of all patients """
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



# ======================================================
# EVALUATION
# ======================================================

def evaluate_per_subject(model):
    """ Evaluate the model on each subject separately """
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
    
    print("\n=== EXPERIMENT : Classical ML models (all models) ===")

    X_train_all, y_train_all = get_aggregated_train_features()

    for model_name in ["Random Forest", "SVM", "XGBoost"]:

        print(f"\n==============================")
        print(f" Running model: {model_name}")
        print(f"==============================")

        all_run_subject_accs = []

        # Multi-run loop
        for run_id in range(1, N_RUNS + 1):

            print(f"\n--- {model_name} | Run {run_id}/{N_RUNS} ---")

            np.random.seed(run_id)

            if model_name == "Random Forest":
                model = Pipeline([("scaler", StandardScaler()),
                    ("rf", RandomForestClassifier(
                        n_estimators=N_ESTIMATORS,
                        random_state=run_id,
                        n_jobs=-1
                    ))
                ])
            elif model_name == "SVM":
                model = Pipeline([("scaler", StandardScaler()),
                    ("svm", SVC(
                        kernel="rbf",
                        C=1.0,
                        gamma="scale",
                        random_state=run_id
                    ))
                ])
            elif model_name == "XGBoost":
                model = Pipeline([("scaler", StandardScaler()),
                    ("xgb", XGBClassifier(
                        objective="multi:softmax",
                        num_class=4,
                        n_estimators=N_ESTIMATORS,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=run_id,
                        n_jobs=-1,
                        eval_metric="mlogloss"
                    ))
                ])

            # Train
            start_time = time.time()
            model.fit(X_train_all, y_train_all)
            train_time = time.time() - start_time

            # Evaluate per subject
            accs = evaluate_per_subject(model)
            all_run_subject_accs.append(accs)

            mean_acc = np.mean(list(accs.values()))

            print(f"Mean accuracy: {mean_acc:.2f}% | Training time: {train_time:.1f}s")

        # Build final CSV table for this model
        rows = []

        # One row per run
        for run_idx, accs in enumerate(all_run_subject_accs, start=1):
            row = {"Run": run_idx}
            for subj in SUBJECTS:
                row[f"Subj{subj}"] = accs[subj]
            row["Mean_Run"] = np.mean(list(accs.values()))
            rows.append(row)

        # Mean / Std per subject across runs
        mean_subject_row = {"Run": "Mean_Subjects"}
        std_subject_row = {"Run": "Std_Subjects"}

        for subj in SUBJECTS:
            subj_vals = [run_accs[subj] for run_accs in all_run_subject_accs]
            mean_subject_row[f"Subj{subj}"] = np.mean(subj_vals)
            std_subject_row[f"Subj{subj}"] = np.std(subj_vals)

        mean_subject_row["Mean_Run"] = np.mean([r["Mean_Run"] for r in rows])
        std_subject_row["Mean_Run"] = np.std([r["Mean_Run"] for r in rows])

        rows.extend([mean_subject_row, std_subject_row])

        # Save CSV
        df = pd.DataFrame(rows)

        model_name_safe = model_name.replace(" ", "_")
        csv_path = os.path.join(SAVE_DIR,f"exp_{model_name_safe}.csv")

        df.to_csv(csv_path, index=False)

        print(f"\nCSV saved to: {csv_path}")

