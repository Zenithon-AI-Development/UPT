import torch
import torch.nn as nn
from pathlib import Path

class Normalizer(nn.Module):
    def __init__(self, size, max_accumulations=10**5, std_epsilon=1e-2, name='Normalizer', device='cuda'):
        super(Normalizer, self).__init__()
        self.name=name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(std_epsilon, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_count = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._num_accumulations = torch.tensor(0, dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._acc_sum_squared = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._mean_fixed = torch.zeros((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._std_fixed = torch.ones((1, size), dtype=torch.float32, requires_grad=False, device=device)
        self._is_fixed = False

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        batched_data = batched_data.to(self._acc_sum.device)
        if accumulate and not self._is_fixed:
            # Stop accumulating after max_accumulations to prevent numerical issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
            else:
                # Fix statistics after max_accumulations
                self._fix_statistics()
        
        if self._is_fixed:
            mean = self._mean_fixed
            std = self._std_fixed
        else:
            mean = self._mean()
            std = self._std_with_epsilon()
        
        # Check for NaN/inf values
        if torch.isnan(mean).any() or torch.isnan(std).any() or torch.isinf(mean).any() or torch.isinf(std).any():
            print(f"Warning: NaN/inf detected in normalizer {self.name}, using fixed stats")
            mean = self._mean_fixed
            std = self._std_fixed
        
        return (batched_data - mean) / std

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        if self._is_fixed:
            return normalized_batch_data * self._std_fixed + self._mean_fixed
        else:
            return normalized_batch_data * self._std_with_epsilon() + self._mean()
    
    def _fix_statistics(self):
        """Fix statistics to prevent further accumulation and numerical issues."""
        if not self._is_fixed:
            self._mean_fixed = self._mean().clone()
            self._std_fixed = self._std_with_epsilon().clone()
            self._is_fixed = True
            print(f"Normalizer {self.name} statistics fixed after {self._num_accumulations} accumulations")

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        batched_data = batched_data.to(self._acc_sum.device)  # Move batched_data to the device where self._acc_sum is located
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batched_data**2, axis=0, keepdims=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(self._acc_count, torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device))
        mean_val = self._mean()
        variance = self._acc_sum_squared / safe_count - mean_val**2
        # Ensure variance is non-negative (handle numerical precision issues)
        variance = torch.clamp(variance, min=0.0)
        std = torch.sqrt(variance + self._std_epsilon**2)  # Add epsilon before sqrt for numerical stability
        # For channels with near-zero variance, use minimum std of 1.0 to prevent numerical issues
        # This is critical for constant channels (like pressure=1.0 everywhere)
        min_std = torch.tensor(1.0, dtype=torch.float32, device=std.device)
        return torch.maximum(std, min_std)

    def get_variable(self):
        
        dict = {'_max_accumulations':self._max_accumulations,
        '_std_epsilon':self._std_epsilon,
        '_acc_count': self._acc_count,
        '_num_accumulations':self._num_accumulations,
        '_acc_sum': self._acc_sum,
        '_acc_sum_squared':self._acc_sum_squared,
        'name':self.name
        }

        return dict


class FixedNormalizer(nn.Module):
    """
    Normalizer with pre-computed, fixed statistics.
    More stable than online accumulation, especially for channels with low variance.
    """
    def __init__(self, mean, std, name='FixedNormalizer', device='cuda'):
        super(FixedNormalizer, self).__init__()
        self.name = name
        
        # Convert to tensors and move to device
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean, dtype=torch.float32)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std, dtype=torch.float32)
        
        # Ensure shape is (1, size) for broadcasting
        if mean.dim() == 1:
            mean = mean.unsqueeze(0)
        if std.dim() == 1:
            std = std.unsqueeze(0)
        
        self.register_buffer('_mean', mean.to(device))
        self.register_buffer('_std', std.to(device))
    
    def forward(self, batched_data, accumulate=True):
        """Normalize using fixed statistics. accumulate parameter ignored."""
        batched_data = batched_data.to(self._mean.device)
        return (batched_data - self._mean) / self._std
    
    def inverse(self, normalized_batch_data):
        """Inverse transformation."""
        return normalized_batch_data * self._std + self._mean
    
    @classmethod
    def from_stats_file(cls, stats_path, stat_type='input', name=None, device='cuda'):
        """
        Load normalizer from pre-computed statistics file.
        
        Args:
            stats_path: Path to .pt file with statistics
            stat_type: 'input', 'label', or 'residual' to select which stats to use
            name: Optional name for the normalizer
            device: Device to place tensors on
        """
        stats = torch.load(stats_path, map_location='cpu', weights_only=False)
        
        if stat_type == 'input':
            mean = stats['input_mean']
            std = stats['input_std']
        elif stat_type == 'label':
            mean = stats['label_mean']
            std = stats['label_std']
        elif stat_type == 'residual':
            # For residuals: mean should be 0 (no shift), only use std for scaling
            mean = stats.get('residual_mean', torch.zeros_like(stats['residual_std']))
            std = stats['residual_std']
        else:
            raise ValueError(f"stat_type must be 'input', 'label', or 'residual', got {stat_type}")
        
        if name is None:
            name = f"Fixed{stat_type.capitalize()}Normalizer"
        
        return cls(mean, std, name=name, device=device)