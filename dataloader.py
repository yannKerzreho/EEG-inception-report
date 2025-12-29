import mne
import numpy as np
from braindecode.datasets import create_from_mne_raw, BaseConcatDataset
from braindecode.preprocessing import (
    preprocess, Preprocessor, exponential_moving_standardize
)

class BCIDatasetBuilder:
    """
    Classe pour charger, nettoyer et préparer des datasets EEG (format GDF)
    pour l'apprentissage profond avec Braindecode.
    """
    
    def __init__(self, file_path, subject_id=1):
        self.file_path = file_path
        self.subject_id = subject_id
        self.raw = None
        self.dataset = None
        
        # Mapping basé sur la doc https://www.bbci.de/competition/iv/desc_2a.pdf
        self.event_mapping = {
            'Left Hand': 769,
            'Right Hand': 770,
            'Foot': 771,
            'Tongue': 772
        }

    def load_and_preprocess(self):
        """
        Charge le fichier GDF, applique un filtre passe-bande et une normalisation.
        """
        print(f"Chargement de {self.file_path}...")
        
        self.raw = mne.io.read_raw_gdf(self.file_path, preload=True, verbose=False)
        
        print("Application du prétraitement...")
        
        # Définition de la chaîne de prétraitement
        preprocessors = [
            # Conversion Volts -> MicroVolts
            Preprocessor(lambda data: data * 1e6),  
            # Sélection des canaux EEG seulement
            Preprocessor('pick_types', eeg=True, meg=False, stim=False), 
        ]
        
        # Application sur les données brutes
        preprocess(self.raw, preprocessors)
        return self

    def create_windows(self, window_size_samples=1000, trial_start_offset_samples=0):
        """
        Découpe le signal continu en fenêtres (époques) basées sur les événements.
        Retourne un WindowsDataset prêt pour PyTorch.
        """
        if self.raw is None:
            raise ValueError("Veuillez d'abord appeler load_and_preprocess()")

        print("Extraction des événements et découpage en fenêtres...")
        
        # Extraction des événements spécifiques (769, 770, 771, 772)
        # On ignore les 'Rejected trial' (1023) ou 'Eye movements' (1072) pour l'entrainement
        events, _ = mne.events_from_annotations(self.raw, event_id=self.event_mapping)
        
        # Mapping inverse pour que Braindecode comprenne (Label 769 -> Classe 0, etc.)
        mapping = {v: k for k, v in self.event_mapping.items()}
        
        # Création du dataset fenêtré
        # trial_start_offset_samples : permet de décaler le début de la fenêtre après le "Bip" (cue)
        self.dataset = create_from_mne_raw(
            [self.raw],
            trial_start_offset_samples=trial_start_offset_samples,
            trial_stop_offset_samples=0,
            window_size_samples=window_size_samples,
            window_stride_samples=window_size_samples, # Pas de chevauchement par défaut
            drop_last_window=False,
            mapping=mapping
        )
        
        print(f"Succès ! Dataset créé avec {len(self.dataset)} exemples.")
        return self.dataset