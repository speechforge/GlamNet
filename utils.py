import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import yaml

def load_audio(path: str, target_sr: int = 24000) -> torch.Tensor:
    """
    Loads an audio file and resamples to target_sr if necessary.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
        
    wav, sr = torchaudio.load(path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        
    # Convert to mono if necessary
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    return wav

def save_audio(path: str, wav: torch.Tensor, sr: int = 24000):
    """
    Saves a tensor to a wav file.
    """
    # Ensure standard normalization
    wav = wav / max(1.0, wav.abs().max().item())
    torchaudio.save(path, wav.cpu(), sr)

def load_config(config_path: str) -> dict:
    """
    Loads model configuration from a YAML file.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def plot_spectrogram(spec: torch.Tensor, title: str = "Spectrogram", save_path: str = None):
    """
    Utility to plot a log-mel spectrogram for debugging.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(spec.squeeze().cpu().numpy(), origin='lower', aspect='auto', cmap='magma')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def compute_mel_spectrogram(wav: torch.Tensor, n_fft: int = 1024, hop_length: int = 256) -> torch.Tensor:
    """
    Computes standard mel spectrogram representation.
    """
    # Core DSP logic deferred to main release
    raise NotImplementedError("DSP core logic pending release.")
