import torch
import torch.nn as nn
import numpy as np

class DataAugmentation(nn.Module):
    """
    Classe abstraite pour l'augmentation de données.
    Hérite de nn.Module pour s'intégrer aux pipelines PyTorch.
    """
    def __init__(self, p=0.8):
        super().__init__()
        self.p = p  # Probabilité d'appliquer l'augmentation

    def forward(self, x):
        # x shape: (Batch, Channels, Time) ou (Batch, Time, Channels)
        # On n'applique l'augm que si on est en mode training et avec proba p
        if self.training and torch.rand(1).item() < self.p:
            return self.augment(x)
        return x

    def augment(self, x):
        raise NotImplementedError("La méthode augment doit être implémentée.")

class GaussianNoise(DataAugmentation):
    """
    Ajoute un bruit gaussien dont l'écart-type est proportionnel 
    à l'écart-type du signal pour chaque canal individuel.
    """
    def __init__(self, relative_scale=0.1, p=0.8):
        """
        Args:
            relative_scale (float): Le facteur de proportionnalité. 
                                    0.1 signifie que le bruit aura 10% de l'amplitude du signal.
            p (float): Probabilité d'application.
        """
        super().__init__(p=p)

        self.relative_scale = relative_scale

    def augment(self, x):
        """
        x shape attendue : (Batch, Channels, Time)
        """
        std_signal = x.std(dim=-1, keepdim=True)
        noise = torch.randn_like(x)
        scaled_noise = noise * std_signal * self.relative_scale
        
        return x + scaled_noise

class HundredHzNoise(DataAugmentation):
    """
    Stocke le bruit > 100Hz extrait du Training Set et l'ajoute aux données.
    """
    def __init__(self, X_train, fs=250, cutoff=100, amplitude=1, p=0.8):
        """
        Args:
            X_train (np.array ou torch.Tensor): Les données d'entrainement complètes (N, C, T) ou (N, T, C).
            fs (int): Fréquence d'échantillonnage (ex: 250Hz).
            cutoff (int): Fréquence de coupure (ex: 100Hz).
            amplitude (float): Facteur multiplicateur du bruit ajouté.
        """
        super().__init__(p=p)

        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32)

        self.amplitude = amplitude

        self.signal_sd = X_train.std(dim=-1).mean(dim=0)

        self.noise_bank = self._extract_high_freq_noise(X_train, fs, cutoff)
        self.noise_sd = self.noise_bank.std(dim=-1).mean(dim=0)
        self.relative_scale = (self.noise_sd / self.signal_sd) * self.amplitude

    def _extract_high_freq_noise(self, X, fs, cutoff):
        # Conversion en numpy si ce n'est pas déjà le cas
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        
        # X shape attendue pour fft: (N, Channels, Time) idéalement
        # Si (N, Time, Channels), on transpose pour faciliter la FFT sur l'axe temporel
        transposed = False
        if X.shape[2] < X.shape[1]: # Heuristique: Time est souvent la plus grande dim
             X = X.transpose(0, 2, 1)
             transposed = True
        
        n_samples = X.shape[0]
        n_times = X.shape[2]
        
        # Calcul de la FFT
        freqs = np.fft.rfftfreq(n_times, 1/fs)
        fft_vals = np.fft.rfft(X, axis=2)
        
        # Création du masque (On met à 0 tout ce qui est INFÉRIEUR au cutoff)
        # On ne garde que > 100Hz
        mask = freqs > cutoff
        
        # Application du masque (broadcasting sur les canaux et batchs)
        fft_filtered = fft_vals * mask[None, None, :]
        
        # Reconstruction du signal temporel (Inverse FFT)
        noise_signals = np.fft.irfft(fft_filtered, axis=2, n=n_times)
        
        # On remet dans le format d'origine si besoin
        if transposed:
            noise_signals = noise_signals.transpose(0, 2, 1)
            
        # Conversion en Tensor PyTorch pour stockage (float32 pour économiser RAM)
        return torch.tensor(noise_signals, dtype=torch.float32)

    def augment(self, x):
        """
        x: Batch de données (Batch_Size, Channels, Time)
        """
        batch_size = x.shape[0]
        
        # 1. Choisir des indices aléatoires dans la banque de bruit
        indices = torch.randint(0, len(self.noise_bank), (batch_size,))
        
        # 2. Récupérer le bruit
        noise_sample = self.noise_bank[indices].to(x.device)
        
        # 3. Ajouter le bruit (mise à l'échelle par l'amplitude)
        return x + (noise_sample * self.amplitude)