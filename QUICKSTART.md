# SWAG Quick Reference Guide

## Quick Start Commands

### 1. Environment Setup
```bash
conda env create -f env.yaml
conda activate avt
```

### 2. Basic Training
```bash
# SWAG-SP* on Cholec80 (30-min horizon)
python train_net.py dataset_name=cholec80 model_name=supra eval_horizons=[30]

# SWAG-SP* on AutoLaparo21
python train_net.py dataset_name=autolaparo21 model_name=supra eval_horizons=[30]
```

### 3. View Results
```bash
tensorboard --logdir OUTPUTS/expts/
```

## Model Variants

| Model | Command | Description |
|-------|---------|-------------|
| SWAG-SP* | `model_name=supra conditional_probs_embeddings=true` | Single-pass with prior knowledge (best) |
| SWAG-SP | `model_name=supra conditional_probs_embeddings=false` | Single-pass without priors |
| SWAG-AR | `model_name=lstm decoder_type=autoregressive` | Autoregressive decoder |
| R2C | `model_name=supra decoder_anticipation=regression` | Regression-to-Classification |

## Key Configuration Parameters

### Dataset Settings
```yaml
dataset_name: cholec80          # or autolaparo21
split_idx: 1                    # 1-4 for k-fold CV
train_start: 1                  # Starting video index
train_end: 10                   # Ending video index
test_start: 15                  # Test split start
test_end: 21                    # Test split end
```

### Model Settings
```yaml
model_name: supra               # Model architecture
conditional_probs_embeddings: true  # Enable prior knowledge
pooling_dim: 32                 # Feature pooling dimension
```

### Training Settings
```yaml
num_epochs: 40                  # Training epochs
eval_horizons: [30]            # Anticipation horizons (minutes)
batch_size: 4                   # Batch size
learning_rate: 0.0001          # Learning rate
```

### Evaluation Settings
```yaml
test_only: false               # Set true for evaluation only
finetune_ckpt: best            # Checkpoint to load (best/latest)
main_metric: acc_curr_future   # Primary metric for model selection
```

## Common Hydra Overrides

### Single Override
```bash
python train_net.py dataset_name=cholec80
```

### Multiple Overrides
```bash
python train_net.py dataset_name=cholec80 num_epochs=50 batch_size=8
```

### List Parameters
```bash
python train_net.py eval_horizons=[15,30,60]
```

### Nested Parameters
```bash
python train_net.py model.hidden_dim=512 opt.lr=0.0001
```

## Directory Structure Quick Reference

| Directory | Purpose |
|-----------|---------|
| `conf/` | Configuration files (Hydra) |
| `models/` | Model architectures |
| `datasets/` | Dataset loaders |
| `func/` | Training/evaluation code |
| `loss_fn/` | Loss functions |
| `common/` | Utilities and helpers |
| `scripts/` | Execution scripts |
| `expts/` | Experiment config files |
| `experiments/` | Results tracking |
| `OUTPUTS/` | Training outputs (auto-generated) |
| `R2A2/` | Baseline implementation |

## File Naming Conventions

### Experiment Config Files (`expts/`)
Format: `{model}_{dataset}_{description}.txt`

Examples:
- `skit_al21_base_g12_nemb_rt18_r2c.txt`
- `lstm_c80_ct144_at60_ls442_eosw_loc24.txt`

### Output Directories (`OUTPUTS/expts/`)
Format: `{config_file_name}/local/`

## Evaluation Metrics

### Classification Metrics
- **F1**: Frame-wise F1 score
- **SegF1**: Segment-wise F1 score (IoU ≥ 0.5)
- **Accuracy**: Frame-wise accuracy

### Regression Metrics
- **wMAE**: Weighted Mean Absolute Error
- **inMAE**: In-phase Mean Absolute Error
- **outMAE**: Out-of-phase Mean Absolute Error

## Experiment Management

### Launch Multiple Experiments
```bash
# Create config file: expts/my_sweep.txt
python scripts/launch.py -c expts/my_sweep.txt -l
```

### Monitor Training
```bash
# TensorBoard
tensorboard --logdir OUTPUTS/expts/{experiment_name}/local/logs/

# Check outputs
ls OUTPUTS/expts/{experiment_name}/local/
```

### Resume Training
```bash
python train_net.py \
    dataset_name=cholec80 \
    finetune_ckpt=latest  # or path to specific checkpoint
```

## Tips and Tricks

### Speed Up Training
- Reduce `num_epochs` for quick tests
- Use smaller `eval_horizons` (e.g., `[15]` instead of `[30]`)
- Increase `batch_size` if GPU memory allows
- Use `save_video_labels_to_npy=true` to cache labels

### Debug Mode
```bash
# Run single epoch to test pipeline
python train_net.py num_epochs=1 test_only=false

# Use debug flag with launcher
python scripts/launch.py -c expts/test.txt -l -g
```

### Save Disk Space
- Outputs are stored in `OUTPUTS/` (gitignored)
- Delete old experiment outputs: `rm -rf OUTPUTS/expts/old_experiment/`
- Keep only best checkpoints, remove intermediate ones

## Getting Help

1. Check configuration: `python train_net.py --cfg job`
2. View available config groups: `python train_net.py --help`
3. See Hydra docs: https://hydra.cc/
4. Refer to paper for methodology details
5. Check `CLEANUP_RECOMMENDATIONS.md` for codebase maintenance

## Citation

```bibtex
@article{boels2025swag,
  title={SWAG: long-term surgical workflow prediction with generative-based anticipation},
  author={Boels, Maxence and Liu, Yang and Dasgupta, Prokar and Granados, Alejandro and Ourselin, Sebastien},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2025},
  publisher={Springer}
}
```
