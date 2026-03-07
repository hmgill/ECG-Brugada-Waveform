"""ECG-specific data augmentation functions."""

import numpy as np
from scipy.interpolate import CubicSpline

from .models import AugmentationConfig


class ECGAugmentation:
    """Apply ECG-specific augmentations."""
    
    def __init__(self, config: AugmentationConfig):
        self.config = config
    
    def __call__(self, signal: np.ndarray) -> np.ndarray:
        """Apply random augmentations to ECG signal."""
        cfg = self.config
        
        # Amplitude scaling
        if np.random.random() < cfg.amplitude_scale_prob:
            signal = signal * np.random.uniform(*cfg.amplitude_scale_range)
        
        # Gaussian noise
        if np.random.random() < cfg.noise_prob:
            noise = np.random.normal(0, cfg.noise_std, signal.shape)
            signal = signal + noise * signal.std()
        
        # Baseline wander
        if np.random.random() < cfg.baseline_wander_prob:
            signal = self._add_baseline_wander(signal)
        
        # Time warping
        if np.random.random() < cfg.time_warp_prob:
            signal = self._time_warp(signal)
        
        # Per-lead scaling
        if np.random.random() < cfg.lead_scale_prob:
            for lead_idx in range(signal.shape[1]):
                signal[:, lead_idx] *= np.random.uniform(*cfg.lead_scale_range)
        
        return signal
    
    def _add_baseline_wander(self, signal: np.ndarray) -> np.ndarray:
        """Add low-frequency baseline wander."""
        n_samples = signal.shape[0]
        freq = np.random.uniform(*self.config.baseline_wander_frequency)
        amplitude = self.config.baseline_wander_amplitude * (signal.max() - signal.min())
        
        t = np.arange(n_samples)
        wander = amplitude * np.sin(2 * np.pi * freq * t / 100)
        return signal + wander[:, np.newaxis]
    
    def _time_warp(self, signal: np.ndarray) -> np.ndarray:
        """Apply subtle time warping."""
        cfg = self.config
        n_samples, n_leads = signal.shape
        
        # Create warping function
        orig_steps = np.linspace(0, n_samples - 1, cfg.time_warp_knots)
        warps = np.random.normal(0, cfg.time_warp_sigma, cfg.time_warp_knots)
        warp_steps = np.sort(orig_steps + warps * n_samples)
        warp_steps[0], warp_steps[-1] = 0, n_samples - 1
        
        warper = CubicSpline(orig_steps, warp_steps)
        warped_indices = np.clip(warper(np.arange(n_samples)), 0, n_samples - 1)
        
        # Apply to all leads
        warped = np.zeros_like(signal)
        for lead_idx in range(n_leads):
            warped[:, lead_idx] = np.interp(
                warped_indices, np.arange(n_samples), signal[:, lead_idx]
            )
        
        return warped
