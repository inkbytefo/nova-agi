# NovaNet Training Guide on Cloud TPU VM (v3-8)

This guide outlines the steps to train NovaNet on the Turkish C4 dataset using a TPU v3-8 environment.

## 1. Prerequisites

*   **Google Cloud Project** with TPU quota.
*   **TPU VM** (v3-8) created and running.
*   **SSH Access** to the TPU VM.

## 2. Environment Setup

Connect to your TPU VM:
```bash
gcloud compute tpus tpu-vm ssh <your-tpu-name> --zone <your-zone>
```

Clone the repository:
```bash
git clone https://github.com/inkbytefo/nova-agi.git
cd nova-agi
```

Install Dependencies (JAX for TPU):
```bash
# Update pip
pip install --upgrade pip

# Install JAX for TPU
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install Project Dependencies
pip install -r requirements.txt
```

**Note:** Ensure `requirements.txt` includes `hydra-core`, `transformers`, `datasets`, `optax`, `flax`, `wandb`.

## 3. Configuration

The TPU configuration is located at `configs/tpu_v3_8.yaml`.
Key settings:
*   **Batch Size:** 1024 (Global) - Optimized for 8 cores (128/core).
*   **Model:** 8 Layers, 512 Hidden Dim.
*   **Dataset:** `c4_tr` (Streaming).

To modify, edit `configs/tpu_v3_8.yaml` using `nano` or `vim`.

## 4. Running Training

Start the training using the TPU config:

```bash
# Login to WandB (Optional, if monitoring is enabled)
wandb login

# Run Training
python scripts/train.py --config-name tpu_v3_8
```

### Monitoring
*   **Console:** Displays Loss and generated text every 10 epochs.
*   **WandB:** If enabled, tracks loss curves, system metrics (TPU utilization), and generated text samples.

## 5. Text Generation (Inference)

To generate text using a trained checkpoint:

```bash
python scripts/generate.py --config-name tpu_v3_8
```
This will load the latest checkpoint from `checkpoints/` and start an interactive prompt loop.

## 6. Troubleshooting

*   **OOM (Out of Memory):** Reduce `batch_size` or `hidden_dim` in `configs/tpu_v3_8.yaml`.
*   **Slow Startup:** JAX compilation can take a few minutes for the first step. This is normal.
*   **NaN Loss:** Try reducing `lr` (Learning Rate).
