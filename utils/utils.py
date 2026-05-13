"""
Utility functions for IAGNet
"""

import os
import yaml


def ensure_dir(path):
    """Ensure directory exists"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def read_yaml(path):
    """Read YAML configuration file"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def write_yaml(path, data):
    """Write YAML configuration file"""
    with open(path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_time(seconds):
    """Format seconds to HH:MM:SS"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
