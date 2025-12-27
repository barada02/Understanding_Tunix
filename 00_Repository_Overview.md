# Tunix Repository Overview

**Created:** December 27, 2025  
**Version:** 0.1.6  
**Status:** Alpha Release

---

## What is Tunix?

**Tunix (Tune-in-JAX)** is a JAX-based library for **Large Language Model (LLM) post-training**. It's designed specifically for efficient training on TPUs (Google's Tensor Processing Units).

### Core Purpose
Transform pre-trained LLMs into task-specific, aligned, and optimized models through:
- Fine-tuning
- Reinforcement Learning
- Knowledge Distillation

---

## Key Capabilities

### 1. **Supervised Fine-Tuning (SFT)**
- Full weight fine-tuning
- Parameter-Efficient Fine-Tuning (PEFT) with LoRA/QLoRA
- DPO (Direct Preference Optimization) for preference alignment

### 2. **Reinforcement Learning (RL)**
- PPO (Proximal Policy Optimization)
- GRPO (Group Relative Policy Optimization)
- GSPO-token (Token-level Group Sequence Policy Optimization)
- Agentic RL with multi-turn support

### 3. **Knowledge Distillation**
- Logit-based distillation
- Attention transfer strategies
- Feature pooling and projection strategies

### 4. **Efficient Text Generation**
- vLLM integration for fast inference on TPUs
- SGLang-Jax integration for structured generation
- Beam search and advanced sampling strategies

---

## Technology Stack

| Component | Technology |
|-----------|-----------|
| **Core Framework** | JAX (hardware-accelerated numerical computing) |
| **Neural Network Library** | Flax NNX (JAX-based neural network library) |
| **Hardware Target** | TPUs (Tensor Processing Units) |
| **Distributed Training** | Pathways (Google's distributed training system) |
| **Fast Inference** | vLLM & SGLang-Jax |
| **Model Execution** | MaxText integration for high-performance kernels |

---

## Repository Structure

```
tunix/
├── cli/              # Command-line interface and configuration
├── sft/              # Supervised Fine-Tuning implementation
├── rl/               # Reinforcement Learning algorithms
├── distillation/     # Knowledge distillation strategies
├── generate/         # Text generation (vLLM, SGLang, samplers)
├── models/           # Model implementations (Gemma, Llama3, Qwen)
├── utils/            # Common utilities
├── perf/             # Performance optimization tools
└── oss/              # Open-source specific utilities

examples/             # Jupyter notebooks and demo scripts
tests/                # Comprehensive test suite
docs/                 # Documentation source files
scripts/              # Setup and utility scripts
```

---

## Supported Models

- **Gemma** (Google's open model)
- **Gemma 3** (Latest version)
- **Llama 3** (Meta's model)
- **Qwen 2 & 3** (Alibaba's models)

Models can be loaded from:
- Hugging Face Hub
- SafeTensors format
- Custom implementations

---

## Key Design Principles

### 1. **Modularity**
Components are reusable and composable - mix and match different training strategies, models, and optimizations.

### 2. **Performance**
- Native TPU optimization
- Micro-batching support
- Integration with high-performance inference engines (vLLM, SGLang)
- MaxText kernel integration

### 3. **Scalability**
- Multi-host distributed training via Pathways
- Scales to thousands of TPU devices
- Efficient sharding strategies

### 4. **Flexibility**
- Plug-and-play logging backends (via Metrax protocol)
- Customizable training loops
- Configurable via YAML or Python API

---

## Installation Options

1. **PyPI (Recommended):** `pip install "google-tunix[prod]"`
2. **From GitHub:** `pip install git+https://github.com/google/tunix`
3. **Development:** Clone and `pip install -e ".[dev]"`

---

## Who Uses Tunix?

- **Researchers** training custom LLMs on TPUs
- **ML Engineers** fine-tuning models for production
- **Academic Labs** experimenting with RL and distillation
- **Open-source projects** like GRL (Game Reinforcement Learning from UCSD)

---

## Current Limitations (Alpha Status)

- Active development - APIs may change
- Documentation still being expanded
- Some features marked as experimental (agentic RL)
- Primary focus on TPU hardware

---

## Quick Start Examples

Tunix provides ready-to-run notebooks:
- `qlora_gemma.ipynb` - PEFT fine-tuning with QLoRA
- `grpo_gemma.ipynb` - RL training for math problems
- `logit_distillation.ipynb` - Model distillation
- `dpo_gemma.ipynb` - Direct Preference Optimization

---

## Next Steps

Proceed to the Learning Roadmap to understand the detailed structure and begin deep diving into specific components!
