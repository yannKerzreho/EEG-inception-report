import numpy as np
import scipy.io # pour lire les .mat
import os

def dataloader(subject_id):
    """
    Charge Train (avec labels internes) et Test (avec labels externes .mat).
    Suppose que vos vrais labels sont dans : 'true_labels/A0iE.mat' (ou A0iT.mat selon votre nommage)
    """
    # --- 1. Définition des chemins ---
    # Données brutes (signaux)
    path_train_npz = f'bcidatasetIV2a/A0{subject_id}T.npz'
    path_test_npz  = f'bcidatasetIV2a/A0{subject_id}E.npz'
    path_true_labels = f'true_labels/A0{subject_id}E.mat' 

    # --- 2. Chargement du TRAIN (Facile, tout est dans le .npz) ---
    X_train, y_train = _load_data_internal(path_train_npz)
    X_train = X_train.transpose(0, 2, 1)
    # --- 3. Chargement du TEST (Complexe, besoin du .mat) ---
    if not os.path.exists(path_true_labels):
        raise FileNotFoundError(f"Fichier de labels manquant : {path_true_labels}")
    
    # On charge les signaux du test ET les vrais labels
    X_test, y_test = _load_data_external(path_test_npz, path_true_labels)
    X_test = X_test.transpose(0, 2, 1)
    
    print(f"X_train : {X_train.shape}") # Devrait être (n_run, 313, 22)
    print(f"y_train : {y_train.shape}") # Devrait être (n_run,)
    print(f"X_test  : {X_test.shape}")  # Devrait être (n_run, 313, 22)
    print(f"y_test  : {y_test.shape}")  # Devrait être (n_run,)

    return X_train, y_train, X_test, y_test

def _load_data_internal(npz_path):
    """Charge un fichier où les labels sont déjà connus"""
    data = np.load(npz_path)
    etyp = data['etyp'].flatten()
    epos = data['epos'].flatten()
    edur = data['edur'].flatten()
    signals = data['s']

    indices = np.where(etyp == 768)[0]

    epos = epos[indices]
    edur = edur[indices]
    etyp = etyp[indices+np.ones_like(indices)]
    
    return _create_arrays(signals, epos, etyp, edur, source='internal')

def _load_data_external(npz_path, mat_path):
    """Charge un fichier masqué (Test - 783) et applique les labels du .mat"""
    data = np.load(npz_path)
    etyp = data['etyp'].flatten()
    epos = data['epos'].flatten()
    edur = data['edur'].flatten()
    signals = data['s']
    
    indices = np.where(etyp == 768)[0]

    edur = edur[indices]
    epos = epos[indices]
    etyp = etyp[indices+np.ones_like(indices)]
    
    mat_data = scipy.io.loadmat(mat_path)
    true_labels = mat_data['classlabel'].flatten() 

    if len(epos) != len(true_labels):
        raise ValueError(f"Désynchronisation ! {len(epos)} essais trouvés dans le .npz mais {len(true_labels)} labels dans le .mat")

    return _create_arrays(signals, epos, true_labels, edur, source='external')

def _create_arrays(signals, positions, labels, length, source):
    """Fonction commune pour découper les signaux"""
    if (length[0] != length).any():
        raise ValueError("Tous les durées d'essais ne sont pas égales. Découpage non supporté.")
    n_trials = len(positions)
    X = np.zeros((n_trials, length[0], 22))
    y = np.zeros(n_trials)
    
    for i in range(n_trials):
        start = positions[i]
        # Découpage strict (length points, 22 canaux)
        X[i] = signals[start:start+length[0], 0:22]
        
        # GESTION DES LABELS
        if source == 'internal':
            y[i] = labels[i] - 769
        elif source == 'external':
            y[i] = labels[i] - 1
            
    return X, y