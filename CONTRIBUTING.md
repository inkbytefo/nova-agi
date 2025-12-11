# Contributing to Nova

Thank you for your interest in contributing to **Nova**! We are building an experimental AGI framework, and we value clarity, rigor, and innovation.

## Code of Conduct

*   Be respectful and constructive.
*   Focus on technical merit and scientific accuracy.

## How to Contribute

1.  **Fork** the repository.
2.  **Create a Branch** for your feature or fix (`git checkout -b feature/amazing-feature`).
3.  **Commit** your changes with clear messages.
4.  **Push** to your fork.
5.  **Open a Pull Request** (PR).

## Development Guidelines

### 1. Style Guide
*   **Python**: Follow [PEP 8](https://peps.python.org/pep-0008/).
*   **JAX/Flax**: Follow functional programming principles. Avoid side effects in model code.
*   **Type Hints**: Use `typing` and `jaxtyping` (e.g., `Float[Array, "b n d"]`) for all tensor operations.

### 2. Testing
*   Run tests before submitting:
    ```bash
    pytest tests/
    ```
*   Add new tests for any new functionality.

### 3. Documentation
*   Update docstrings for all public functions.
*   If changing the architecture, update `README.md`.

## Project Structure

*   `src/nova/models`: Neural network architectures (Flax).
*   `src/nova/core`: Core logic (Loss, Topology, Ops).
*   `src/nova/data`: Data pipelines (Streaming, Tokenization).
*   `scripts`: Entry points for Training and Generation.
*   `configs`: Hydra configuration files.

## Reporting Issues

Please use the GitHub Issues tab to report bugs or suggest enhancements. Provide:
*   Steps to reproduce.
*   Environment details (OS, JAX version, Hardware).
*   Expected vs. Actual behavior.
