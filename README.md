# Nova: Thermodynamic Hypergraph AGI (Turkish C4 & TPU Edition)

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![JAX](https://img.shields.io/badge/JAX-TPU%20Optimized-green.svg)
![Status](https://img.shields.io/badge/Status-Research-orange.svg)

**Nova** is an experimental AGI research framework that models cognition as a thermodynamic process on a dynamic hypergraph topology. It integrates **Friston’s Free Energy Principle** with **Geometric Deep Learning** to create self-organizing neural structures.

This repository is currently focused on training **NovaNet** on the **Turkish C4 (Colossal Clean Crawled Corpus)** dataset using **Google Cloud TPUs**.

## 🚀 Key Features

*   **Dynamic Hypergraph Topology**: Nodes represent tokens, edges represent sequential, local (window), and global (sentence) context.
*   **Thermodynamic Loss**: Minimizes variational free energy (Accuracy + Energy Efficiency).
*   **Streaming Data Pipeline**: Efficiently streams terabytes of text data (C4 Turkish) without loading into RAM.
*   **Autoregressive Generation**: Supports text generation via hypergraph reconstruction.
*   **TPU Optimization**: Fully compatible with JAX `pmap` for multi-core TPU v3-8 training.

---

## 🛠️ Installation

### Prerequisites
*   Python 3.10+
*   JAX (with TPU or CUDA support)

### Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/inkbytefo/nova-agi.git
    cd nova-agi
    ```

2.  **Install Dependencies:**
    ```bash
    # Standard installation
    pip install -r requirements.txt

    # For TPU usage, ensure you have the correct JAX version:
    # pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
    ```

---

## 🏃 Usage

### 1. Training (Turkish C4)

To train the model on the Turkish C4 dataset using the default configuration (or TPU config):

```bash
# Local Training (CPU/GPU)
python scripts/train.py

# TPU Training (Optimized for v3-8)
python scripts/train.py --config-name tpu_v3_8
```

See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed TPU setup instructions.

### 2. Text Generation (Inference)

You can generate text interactively using a trained checkpoint:

```bash
# Interactive Mode
python scripts/generate.py

# Specify a config or checkpoint directory
python scripts/generate.py --config-name tpu_v3_8
```

### 3. Configuration

Nova uses [Hydra](https://hydra.cc/) for configuration management.
*   `configs/config.yaml`: Default local settings.
*   `configs/tpu_v3_8.yaml`: High-performance TPU settings.

Override parameters via CLI:
```bash
python scripts/train.py training.batch_size=32 model.hidden_dim=256
```

---

## 🧠 Architecture

NovaNet processes text not as a flat sequence, but as a **Hypergraph**:

1.  **Tokenization**: Text -> Token IDs (BERT Turkish).
2.  **Graph Construction**:
    *   **Nodes**: Tokens.
    *   **Edges**:
        *   *Sequential*: $t_i \leftrightarrow t_{i+1}$
        *   *Context*: $t_i \leftrightarrow t_{i+1} \leftrightarrow t_{i+2}$
        *   *Global*: All tokens in a sentence connected by one hyperedge.
3.  **Message Passing**: Hypergraph Convolution (Gather -> Update -> Scatter).
4.  **Prediction**: Predicts next token minimizing Cross-Entropy (Information Surprise) + Energy Cost.

---

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License. See `LICENSE` for details.
