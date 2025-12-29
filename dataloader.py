import numpy as np
import scipy.io  # Indispensable pour lire les .mat
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
    
    # Vrais labels (ajustez le nom de fichier selon ce que vous avez, ex: A01E.mat)
    # Le user a dit "AOiX.mat", donc je suppose A01X.mat, A02X.mat...
    path_true_labels = f'true_labels/A0{subject_id}E.mat' 
    # NOTE : Si vos fichiers s'appellent vraiment A01X.mat, changez 'E' par 'X' ci-dessus.

    # --- 2. Chargement du TRAIN (Facile, tout est dans le .npz) ---
    X_train, y_train = _load_data_internal(path_train_npz)

    # --- 3. Chargement du TEST (Complexe, besoin du .mat) ---
    if not os.path.exists(path_true_labels):
        raise FileNotFoundError(f"Fichier de labels manquant : {path_true_labels}")
    
    # On charge les signaux du test ET les vrais labels
    X_test, y_test = _load_data_external(path_test_npz, path_true_labels)

    print(f"X_train : {X_train.shape}") # Devrait être (n_run, 313, 22)
    print(f"y_train : {y_train.shape}") # Devrait être (n_run,)
    print(f"X_test  : {X_test.shape}")  # Devrait être (n_run, 313, 22)
    print(f"y_test  : {y_test.shape}")  # Devrait être (n_run,)

    return X_train, y_train, X_test, y_test

def _load_data_internal(npz_path):
    """Charge un fichier où les labels sont déjà connus (Train - 769 à 772)"""
    data = np.load(npz_path)
    etyp = data['etyp'].T[0]
    epos = data['epos'].T[0]
    signals = data['s']

    # On ne garde que les essais moteurs (769-772)
    valid_codes = [769, 770, 771, 772]
    mask = np.isin(etyp, valid_codes)
    
    epos = epos[mask]
    etyp = etyp[mask]
    
    return _create_arrays(signals, epos, etyp, source='internal')

def _load_data_external(npz_path, mat_path):
    """Charge un fichier masqué (Test - 783) et applique les labels du .mat"""
    data = np.load(npz_path)
    etyp = data['etyp'].T[0]
    epos = data['epos'].T[0]
    signals = data['s']
    
    # On cherche uniquement les marqueurs "Inconnu/Test" (783)
    mask = (etyp == 783)
    epos = epos[mask]
    # On n'utilise pas etyp ici car il vaut 783 partout
    
    # B. Charger le MAT pour avoir les classes (QUOI se passe)
    mat_data = scipy.io.loadmat(mat_path)
    
    # La clé s'appelle souvent 'classlabel' dans les fichiers BCI IV 2a
    # .flatten() permet de passer d'un vecteur colonne (Nx1) à un tableau plat (N,)
    true_labels = mat_data['classlabel'].flatten() 

    # C. Vérification de sécurité (CRITIQUE)
    if len(epos) != len(true_labels):
        raise ValueError(f"Désynchronisation ! {len(epos)} essais trouvés dans le .npz mais {len(true_labels)} labels dans le .mat")

    return _create_arrays(signals, epos, true_labels, source='external')

def _create_arrays(signals, positions, labels, source):
    """Fonction commune pour découper les signaux"""
    n_trials = len(positions)
    X = np.zeros((n_trials, 313, 22))
    y = np.zeros(n_trials)
    
    for i in range(n_trials):
        start = positions[i]
        # Découpage strict (313 points, 22 canaux)
        X[i] = signals[start:start+313, 0:22]
        
        # GESTION DES LABELS
        if source == 'internal':
            # 769..772 -> 0..3
            y[i] = labels[i] - 769
        elif source == 'external':
            # MATLAB est indexé à 1 (classes 1,2,3,4) -> Python (0,1,2,3)
            y[i] = labels[i] - 1 
            
    return X, y