"""
Legacy codebase - organized for backward compatibility.

This module contains the original UPT codebase reorganized for clarity.
All imports are preserved for backward compatibility.
"""

# Re-export main modules for backward compatibility
from legacy import models
from legacy import trainers
from legacy import datasets
from legacy import collators
from legacy import conditioners

__all__ = ['models', 'trainers', 'datasets', 'collators', 'conditioners']

