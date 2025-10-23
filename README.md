# SWAG: Surgical Workflow Anticipative Generation

[![Paper](https://img.shields.io/badge/Paper-Springer-blue)](https://link.springer.com/article/10.1007/s11548-025-03452-8)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://maxboels.com/research/swag)

**Long-term Surgical Workflow Prediction with Generative-Based Anticipation**

*Maxence Boels¹ · Yang Liu¹ · Prokar Dasgupta¹ · Alejandro Granados¹ · Sebastien Ourselin¹*

¹Surgical and Interventional Engineering, School of Biomedical Engineering and Imaging Sciences, King's College London

---

## 📋 Abstract

SWAG is a unified encoder-decoder framework for surgical phase recognition and long-term anticipation that addresses a critical gap in intraoperative decision support. While existing approaches excel at recognizing current surgical phases, they provide limited foresight into future procedural steps. SWAG combines phase recognition and anticipation using a generative approach, predicting sequences of future surgical phases at minute intervals over horizons up to 60 minutes.

## 📚 Documentation

- **[README.md](README.md)** - This file: comprehensive project overview
- **[QUICKSTART.md](QUICKSTART.md)** - Quick reference guide for common tasks and commands
- **[CLEANUP_RECOMMENDATIONS.md](CLEANUP_RECOMMENDATIONS.md)** - Codebase maintenance and cleanup guide
- **[REPOSITORY_CLEANUP_SUMMARY.md](REPOSITORY_CLEANUP_SUMMARY.md)** - Recent repository organization changes

For contribution guidelines and community standards, see the [docs/](docs/) directory.

## 🎯 Key Features

- **Unified Recognition and Anticipation**: Jointly addresses surgical phase recognition and long-term workflow prediction
- **Dual Generative Approaches**: Implements both single-pass (SP) and autoregressive (AR) decoding methods
- **Prior Knowledge Embedding**: Novel embedding approach using class transition probabilities (SP*)
- **Regression-to-Classification (R2C)**: Framework for converting remaining time predictions to discrete phase sequences
- **Long-horizon Predictions**: Extends anticipation from typical 5-minute limits to 20-60 minute horizons
- **Multi-dataset Validation**: Evaluated on Cholec80 and AutoLaparo21 datasets

## 🏗️ Architecture Overview

```
Input Video Frames → Vision Encoder (ViT) → 
    ↓
Windowed Self-Attention (WSA) →
    ↓
Compression & Pooling (CP) →
    ├─→ Recognition Head → Current Phase
    └─→ Decoder (SP/AR) → Future Phases (N × 60s intervals)
```

**Key Components:**
- **Vision Encoder**: Fine-tuned ViT with AVT approach for spatial-temporal features
- **WSA**: Sliding window self-attention (W=20, L=1440 frames)
- **Compression**: Global key-pooling (SP) and interval-pooling (AR)
- **Decoders**: 
  - SP: Single-pass transformer decoder with parallel prediction
  - AR: GPT-2-based autoregressive generation
  - SP*: Enhanced with prior knowledge embeddings

## 📁 Project Structure

```
SWAG-surgical-workflow-anticipative-generation/
├── train_net.py                # Main training entry point
├── env.yaml                    # Conda environment specification
├── README.md                   # This file
├── LICENSE                     # MIT License
│
├── conf/                       # Hydra configuration files
│   ├── config.yaml            # Main config with all parameters
│   ├── data/                  # Dataset-specific configs
│   ├── model/                 # Model architecture configs
│   ├── opt/                   # Optimizer configs
│   └── train_eval_op/         # Training operation configs
│
├── src/                        # Source code (all Python modules)
│   ├── models/                # Model architectures
│   │   ├── supra.py          # SWAG-SP/SP* implementation
│   │   ├── lstm.py           # LSTM-based AR model
│   │   ├── transformers.py   # Transformer decoder variants
│   │   └── base_model.py     # Base model class
│   │
│   ├── datasets/              # Dataset loaders
│   │   ├── cholec80/         # Cholec80 dataset utilities
│   │   ├── autolaparo21/     # AutoLaparo21 dataset utilities
│   │   └── base_video_dataset.py # Base video dataset class
│   │
│   ├── func/                  # Training and evaluation functions
│   │   ├── train.py          # Main training loop
│   │   └── train_eval_ops.py # Training operations
│   │
│   ├── loss_fn/               # Loss function implementations
│   │   ├── multidim_xentropy.py   # Multi-dimensional cross-entropy
│   │   ├── remaining_time_loss.py # Remaining time regression loss
│   │   ├── mse.py            # Mean squared error
│   │   └── mae.py            # Mean absolute error
│   │
│   └── common/                # Common utilities
│       ├── utils.py          # General utilities
│       ├── transforms.py     # Data transformations
│       ├── sampler.py        # Data samplers
│       └── scheduler.py      # Learning rate schedulers
│
├── scripts/                    # Execution scripts
│   ├── launch.py              # Experiment launcher
│   ├── run_experiments.sh     # Batch experiment runner
│   └── runai.sh               # Cluster deployment script
│
├── experiments/                # Experiment tracking
│   ├── configs/               # Experiment configuration files (formerly expts/)
│   ├── top_runs*.json         # Best experiment results
│   └── run_*.txt              # Experiment logs
│
├── docs/                       # Documentation
│   ├── README files and guides
│   ├── CODE_OF_CONDUCT.md     # Code of conduct
│   ├── CONTRIBUTING.md        # Contribution guidelines
│   └── assets/                # Media files (GIFs, images)
│
├── baselines/                  # Baseline implementations
│   ├── R2A2/                  # R2A2 baseline
│   └── Informer2020/          # Informer baseline
│
└── OUTPUTS/                    # Training outputs (gitignored)
    └── expts/                 # Experiment outputs
        └── {experiment_name}/ # Individual experiment results
            ├── checkpoints/   # Model checkpoints
            ├── logs/          # TensorBoard logs
            └── plots/         # Evaluation plots
```

## 📊 Results

### Phase Anticipation Performance

| Method | Cholec80 F1 (%) | AutoLaparo21 F1 (%) | Cholec80 SegF1 (%) | AutoLaparo21 SegF1 (%) |
|--------|-----------------|---------------------|---------------------|------------------------|
| SP* | **32.1** | **41.3** | 29.8 | **34.8** |
| R2C | 36.1 | 32.9 | **32.5** | 29.2 |
| AR | 27.8 | 29.3 | 25.0 | 23.3 |

### Remaining Time Regression (Cholec80)

| Horizon | wMAE (min) | inMAE (min) | outMAE (min) |
|---------|------------|-------------|--------------|
| 2-min | **0.32** | **0.54** | **0.09** |
| 3-min | **0.48** | **0.77** | **0.17** |
| 5-min | 0.80 | 1.26 | 0.34 |

*Outperforms IIA-Net and Bayesian baselines without requiring additional instrument annotations*

## 🔧 Installation

### Prerequisites
- Python 3.7+
- CUDA 11.0+ (for GPU support)
- Conda (recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/maxboels/SWAG-surgical-workflow-anticipative-generation.git
cd SWAG-surgical-workflow-anticipative-generation

# Create environment from yaml file
conda env create -f env.yaml
conda activate avt

# The environment includes all necessary dependencies:
# - PyTorch with CUDA support
# - Hydra for configuration management
# - timm for vision transformers
# - faiss-cpu for efficient similarity search
# - and other required packages
```

### Dataset Preparation

Download and prepare the datasets:

1. **Cholec80**: Download from [CAMMA](http://camma.u-strasbg.fr/datasets)
2. **AutoLaparo21**: Download from [AutoLaparo](https://autolaparo.github.io/)

Extract videos and annotations to:
```
datasets/
├── cholec80/
│   ├── videos/
│   └── annotations/
└── autolaparo21/
    ├── videos/
    └── annotations/
```

## 📦 Datasets

The model is evaluated on two publicly available datasets:

- **[Cholec80](http://camma.u-strasbg.fr/datasets)**: 80 cholecystectomy videos with 7 surgical phases
  - Split: 32 train / 8 val / 40 test (4-fold cross-validation available)
  - Average duration: 38 minutes
  - Sampled at 1 fps

- **[AutoLaparo21](https://autolaparo.github.io/)**: 21 laparoscopic hysterectomy videos
  - Split: 10 train / 4 val / 7 test
  - Average duration: 66 minutes
  - Sampled at 1 fps

Both datasets use 7 surgical phases + end-of-surgery (EOS) class for anticipation.

**Dataset Organization:**
```
datasets/
├── cholec80/
│   ├── videos/          # Video files or extracted frames
│   ├── labels/          # Phase annotations
│   └── splits/          # Train/val/test splits
└── autolaparo21/
    ├── videos/
    ├── labels/
    └── splits/
```

## 🚀 Usage

### Training

The project uses [Hydra](https://hydra.cc/) for configuration management. All configurations are in `conf/config.yaml`.

#### Train SWAG-SP* (Single-Pass with Prior Knowledge)
```bash
# Train on Cholec80
python train_net.py \
    dataset_name=cholec80 \
    model_name=supra \
    conditional_probs_embeddings=true \
    eval_horizons=[30] \
    num_epochs=40

# Train on AutoLaparo21
python train_net.py \
    dataset_name=autolaparo21 \
    model_name=supra \
    conditional_probs_embeddings=true \
    eval_horizons=[30] \
    num_epochs=40
```

#### Train SWAG-AR (Autoregressive)
```bash
python train_net.py \
    dataset_name=cholec80 \
    model_name=lstm \
    decoder_type=autoregressive \
    eval_horizons=[30]
```

#### Train R2C (Regression-to-Classification)
```bash
python train_net.py \
    dataset_name=cholec80 \
    model_name=supra \
    decoder_anticipation=regression \
    probs_to_regression_method=first_occurrence
```

### Batch Experiments with Launch Script

For running multiple experiments or hyperparameter sweeps:

```bash
# Create an experiment config file in experiments/configs/
# e.g., experiments/configs/my_experiment.txt with Hydra overrides

# Run locally
python scripts/launch.py -c experiments/configs/my_experiment.txt -l

# Run on cluster (SLURM)
python scripts/launch.py -c experiments/configs/my_experiment.txt -p gpu_partition

# Debug mode (single GPU)
python scripts/launch.py -c experiments/configs/my_experiment.txt -l -g
```

### Evaluation

Evaluation metrics are computed during training and logged to TensorBoard:

```bash
# View training progress
tensorboard --logdir OUTPUTS/expts/YOUR_EXPERIMENT/local/logs/
```

#### Evaluate Saved Checkpoints
```bash
# Test mode uses the best checkpoint
python train_net.py \
    dataset_name=cholec80 \
    model_name=supra \
    test_only=true \
    finetune_ckpt=best
```

### Configuration

Key configuration parameters in `conf/config.yaml`:

- `dataset_name`: cholec80 or autolaparo21
- `model_name`: supra (SP/SP*), lstm (AR), naive1, naive2
- `eval_horizons`: List of anticipation horizons in minutes (e.g., [30])
- `conditional_probs_embeddings`: Enable prior knowledge (SP*)
- `num_epochs`: Training epochs
- `split_idx`: For k-fold cross-validation (1-4)

See `conf/config.yaml` for all available options.

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@article{boels2025swag,
  title={SWAG: long-term surgical workflow prediction with generative-based anticipation},
  author={Boels, Maxence and Liu, Yang and Dasgupta, Prokar and Granados, Alejandro and Ourselin, Sebastien},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2025},
  publisher={Springer},
  doi={10.1007/s11548-025-03452-8}
}
```

## 🔬 Related Work

This work builds upon and compares with several state-of-the-art methods:

- **[Trans-SVNet](https://arxiv.org/abs/2103.09712)**: Transformer-based surgical workflow analysis
- **[SKiT](https://arxiv.org/abs/2207.09903)**: Fast key information video transformer for surgical phase recognition
- **[LoViT](https://arxiv.org/abs/2110.08085)**: Long Video Transformer for surgical phase recognition
- **[IIA-Net](https://link.springer.com/chapter/10.1007/978-3-031-16449-1_22)**: Instrument interaction anticipation network
- **Action Anticipation**: Builds on concepts from video action anticipation literature

Our R2A2 baseline implementation is included in the `R2A2/` directory.

## 🐛 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the correct conda environment
conda activate avt

# Verify Python can find the modules
export PYTHONPATH="${PYTHONPATH}:/path/to/SWAG-surgical-workflow-anticipative-generation"
```

**CUDA Out of Memory**
- Reduce batch size in config: `batch_size=2`
- Reduce sequence length: Modify window size in config
- Use gradient accumulation steps

**Dataset Not Found**
- Check dataset paths in `conf/data/` configs
- Ensure datasets are properly extracted
- Verify file permissions

**Hydra Configuration Errors**
- Check syntax in override files (`.txt` in `expts/`)
- Ensure all referenced config groups exist in `conf/`
- Use `--cfg job` to print resolved config

### Performance Tips

- Use multiple workers for data loading: `num_workers=4`
- Enable mixed precision training for faster training
- Use distributed training for multi-GPU setups
- Cache preprocessed features to speed up subsequent runs

## 📚 Additional Resources

- [Project Page](https://maxboels.com/research/swag) - Visualizations and supplementary materials
- [Paper](https://link.springer.com/article/10.1007/s11548-025-03452-8) - Full technical details
- See `CLEANUP_RECOMMENDATIONS.md` for codebase maintenance guidelines

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Datasets: Cholec80 (Strasbourg University) and AutoLaparo21
- Vision Transformer implementation based on timm library
- Transformer architectures adapted from PyTorch

## 📧 Contact

For questions or collaboration inquiries:
- Maxence Boels: maxence.boels@kcl.ac.uk
- Project Page: https://maxboels.com/research/swag

## 🔮 Future Directions

- Uncertainty estimation for stochastic workflow modeling
- Multi-trajectory anticipation frameworks
- Scaling to larger surgical datasets with fine-grained annotations
- Integration with robotic surgical systems for proactive assistance