"""ddpm_derm: HAM10000 DDPM data-augmentation portfolio project.

Layers are split by dependency weight so the data/metric logic can be
tested without a deep-learning stack installed:

- config, manifests, metrics : pure Python + pandas/numpy (no torch)
- dataset, model, train_classifier : require torch / torchvision (Colab)
"""

__version__ = "0.1.0"
