## Developer: inkbytefo
## Modified: 2025-12-12

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

The TPU configuration is located at `configs/tpu_v3_8_hdct.yaml`.
Key settings:
*   **Batch Size:** 512 (Global) - Optimized for 8 cores (64/core), HDCT model için ayarlandı.
*   **Model:** 12 Layers, 768 Hidden Dim, HDCT (Hypergraph Discrete Cosine Transform).
*   **Dataset:** `turkish_v2` (Streaming) with `dataset.mode: curriculum`.
*   **Sequence Length:** `max_seq_len: 2048`.
*   **Checkpoint:** `gs://nova-agi-hdct-checkpoints/hdct-v1` (GCS bucket).

To modify, edit `configs/tpu_v3_8_hdct.yaml` using `nano` or `vim`.

## 4. Running Training

Start the training using the TPU config:

```bash
# Login to WandB (Optional, if monitoring is enabled)
wandb login

# Run Training (Curriculum Mode)
python scripts/train.py --config-name tpu_v3_8_hdct dataset.mode=curriculum

nohup python scripts/train.py --config-name tpu_v3_8_hdct dataset.mode=curriculum use_wandb=true > train.log 2>&1 &

# Verify TPU devices recognized (optional)
python - <<'PY'
import jax
print('Devices:', jax.devices())
print('Local count:', jax.local_device_count())
PY
```

### Monitoring
*   **Console:** Displays Loss and generated text every 10 epochs.
*   **WandB:** If enabled, tracks loss curves, system metrics (TPU utilization), and generated text samples.

## 5. Text Generation (Inference)

To generate text using a trained checkpoint:

```bash
python scripts/generate.py --config-name tpu_v3_8_hdct
```
This will load the latest checkpoint from `checkpoints/` and start an interactive prompt loop.

## 6. Troubleshooting

*   **OOM (Out of Memory):** Reduce `batch_size` or `hidden_dim` in `configs/tpu_v3_8_hdct.yaml`.
*   **Slow Startup:** JAX compilation can take a few minutes for the first step. This is normal.
*   **NaN Loss:** Try reducing `lr` (Learning Rate).
*   **No TPU devices:** Ensure `pip install "jax[tpu]"` completed and you are running on a TPU VM.
*   **HF Streaming Errors:** Datasets can occasionally rate-limit. Retry or ensure VM has outbound internet.

## 7. Curriculum Phases

NovaNet uses a phased curriculum with dynamic dataset mixing:
*   **Phase 0 (0–15k steps):** Turkish Corpus 70%, Python Code 30%
*   **Phase 1 (15k–30k steps):** Turkish Corpus 40%, Python Code 30%, Turkish Instructions 30%
*   **Phase 2 (30k–45k steps):** Turkish CoT 40% (fallback: Turkish-Alpaca), Complex Code 40%, Turkish Corpus 20%

Phase changes are detected inside the training loop and logged to console/WandB.
