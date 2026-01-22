"""
Common utilities and datasets for VTON training

This package provides shared dataset implementations and utilities
that can be used across different VTON training methods.
"""

from .dataset import (
    S3VTONDataset,
    S3VTONDatasetEasy,
    S3VTONDatasetMedium,
    S3VTONDatasetHard,
    get_vton_dataset
)

__all__ = [
    'S3VTONDataset',
    'S3VTONDatasetEasy',
    'S3VTONDatasetMedium',
    'S3VTONDatasetHard',
    'get_vton_dataset',
]

__version__ = '1.0.0'
