---
language: en
license: mit
tags:
- graph-neural-networks
- hypergraphs
- agi-research
- jax
- flax
dataset:
- zinc
- synthetic
---

# Model Card for NovaNet v0.6

## Model Details

- **Name:** NovaNet
- **Version:** v0.6
- **Type:** Dynamic Hypergraph Neural Network
- **Framework:** JAX / Flax
- **Developer:** Nova Research Team
- **License:** MIT

## Intended Use

### Primary Use Case
This model is designed for **experimental research** in:
- Graph Representation Learning (GRL) on higher-order structures (hypergraphs).
- Architectures inspired by the Free Energy Principle (FEP).
- Dynamic topology optimization in neural networks.

### Out-of-Scope Use
- Production deployment for critical decision-making.
- General-purpose language modeling or computer vision tasks (without adaptation).

## Training Data

The model supports training on:
1. **Synthetic Hypergraphs**: Generated procedurally to test topological adaptation capabilities.
2. **ZINC Dataset**: A subset of the ZINC database of commercially available chemical compounds, used for molecular property prediction benchmarks.

## Model Architecture

NovaNet utilizes a **Dynamic Topology** mechanism where the graph structure itself is a learnable component.
- **Input**: Graph/Hypergraph structures (Nodes, Edges/Hyperedges).
- **Core**: Message passing layers with attention mechanisms.
- **Loss**: A composite of Task Loss (MSE/CrossEntropy) and **Thermodynamic Loss** (energy efficiency/complexity penalty).

## Ethical Considerations

This is **research code** intended to explore AGI architectures.
- **Reliability**: The dynamic nature of the topology may lead to non-deterministic behaviors.
- **Bias**: If trained on molecular data (ZINC), biases in the dataset regarding chemical properties will be reflected.
- **Energy**: While designed to optimize "thermodynamic" efficiency, large-scale training of dynamic graphs can be computationally intensive.

## Citation

Please cite this repository if you use Nova in your research (see `CITATION.cff`).
