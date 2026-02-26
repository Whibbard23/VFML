# event_training

Minimal event_training scripts for ROI crop-based event classification.

## Purpose
This folder contains a small, self-contained training pipeline to:
- load ROI crop images referenced by CSVs,
- run a smoke test to validate data loading and model shapes,
- run a short or full training run and save checkpoints.

It is intentionally minimal so you can adapt it to your full training stack.

## Files
- `data_loader.py` — PyTorch Dataset and DataLoader factory.
- `utils.py` — simple CNN model, metrics, and checkpoint helpers.
- `train.py` — training loop with smoke mode.
- `config.yaml` — example configuration.
- `experiments/` — directory where checkpoints and logs are saved.

## Requirements
Install Python packages
```bash
pip install torch torchvision pillow pyyaml
