# Understanding Tunix

Welcome to the Understanding Tunix project! This repository contains **comprehensive documentation** (~32,000 lines) for learning the [Tunix](https://github.com/google/tunix) framework - Google's JAX-based library for fine-tuning and aligning large language models on TPUs.

## 📚 Overview

Tunix is a powerful framework that provides:
- **Supervised Fine-Tuning (SFT)** with parameter-efficient methods (LoRA, QLoRA)
- **Reinforcement Learning (RL)** algorithms (GRPO, PPO, DPO)
- **Knowledge Distillation** for model compression
- **Optimized Generation** with vLLM integration
- **Multi-model support** (Gemma, Llama3, Qwen2, and more)
- **Distributed training** on TPU pods

This documentation provides deep-dive coverage of Tunix's architecture, APIs, examples, and best practices.

## 📖 Documentation Structure

### **Phase 1: High-Level Understanding**
Foundation concepts and architecture overview
- [`Phase_1_1_Core_Concepts.md`](Phase_1_1_Core_Concepts.md) - Training paradigms, JAX/Flax basics
- [`Phase_1_2_Architecture_Overview.md`](Phase_1_2_Architecture_Overview.md) - Repository structure, module organization
- [`Phase_1_3_Key_Technologies.md`](Phase_1_3_Key_Technologies.md) - JAX, Flax NNX, TPU fundamentals

### **Phase 2: Core Components Deep Dive** (~8,000 lines)
Detailed API documentation for each major module
- [`Phase_2_1_SFT_Module.md`](Phase_2_1_SFT_Module.md) - Supervised fine-tuning, PeftTrainer, checkpoint management
- [`Phase_2_2_RL_Module.md`](Phase_2_2_RL_Module.md) - GRPO, PPO, DPO, RLCluster, reward functions
- [`Phase_2_3_Distillation_Module.md`](Phase_2_3_Distillation_Module.md) - Logit distillation, feature extraction, strategies
- [`Phase_2_4_Generation_Module.md`](Phase_2_4_Generation_Module.md) - Samplers, vLLM integration, beam search
- [`Phase_2_5_Models_Module.md`](Phase_2_5_Models_Module.md) - Model loading, AutoModel, supported architectures

### **Phase 3: Advanced Topics** (~5,000 lines)
Production-grade training infrastructure
- [`Phase_3_Advanced_Topics.md`](Phase_3_Advanced_Topics.md)
  - Sharding strategies (FSDP, tensor parallel, data parallel)
  - Checkpoint management with Orbax
  - Metrics logging (Metrax, TensorBoard, WandB)
  - Multi-host distributed training
  - Grain data loading pipeline
  - System metrics and profiling

### **Phase 4: Examples & Practical Implementation** (~19,000 lines)
Complete runnable examples with detailed explanations
- [`Phase_4_Examples_and_Implementation.md`](Phase_4_Examples_and_Implementation.md)
  - **Jupyter Notebooks**: QLoRA, GRPO, Logit Distillation, DPO
  - **Production Scripts**: Multi-model GRPO, inference examples
  - **Agentic RL**: Multi-turn conversations, tool usage
  - **DeepScaler**: Math reasoning training and evaluation
  - **Usage Patterns**: Adapting examples, scaling hyperparameters
  - **Best Practices**: Development workflow, debugging, reproducibility

### **Supporting Documents**
- [`00_Repository_Overview.md`](00_Repository_Overview.md) - Repository structure and file organization
- [`01_Learning_Roadmap.md`](01_Learning_Roadmap.md) - Complete learning path with checkboxes
- [`Git_Commands.md`](Git_Commands.md) - Git workflow for this repository

## 🚀 Quick Start

### **1. For Beginners**
Start with Phase 1 to understand core concepts:
```bash
# Read foundation documents
Phase_1_1_Core_Concepts.md
Phase_1_2_Architecture_Overview.md
Phase_1_3_Key_Technologies.md
```

### **2. For Practitioners**
Jump to examples if you're familiar with LLM training:
```bash
# Practical implementation with code
Phase_4_Examples_and_Implementation.md
```

### **3. For API Reference**
Deep-dive into specific modules:
```bash
# Choose your area of interest
Phase_2_1_SFT_Module.md        # Fine-tuning
Phase_2_2_RL_Module.md          # Reinforcement learning
Phase_2_3_Distillation_Module.md  # Model compression
Phase_2_4_Generation_Module.md  # Inference
Phase_2_5_Models_Module.md      # Model loading
```

### **4. For Production Deployment**
Learn infrastructure and optimization:
```bash
# Advanced topics
Phase_3_Advanced_Topics.md
```

## 📊 Coverage Statistics

| Phase | Document | Lines | Topics Covered |
|-------|----------|-------|----------------|
| **Phase 1** | Core Concepts | ~1,000 | 3 documents covering foundations |
| **Phase 2** | Core Components | ~8,000 | 5 modules with API documentation |
| **Phase 3** | Advanced Topics | ~5,000 | 9 sections on infrastructure |
| **Phase 4** | Examples | ~19,000 | 4 example types + patterns + best practices |
| **Total** | | **~32,000** | **Comprehensive coverage** |

## 🎯 Learning Path

Follow the [**Learning Roadmap**](01_Learning_Roadmap.md) for a structured progression:

1. ✅ **Phase 0**: Setup & Prerequisites
2. 📖 **Phase 1**: High-Level Understanding  
3. 🔍 **Phase 2**: Core Components Deep Dive  
4. 🏗️ **Phase 3**: Advanced Topics  
5. 💡 **Phase 4**: Examples & Practical Implementation ← **Most Popular**
6. 🧪 **Phase 5**: Testing & Best Practices *(coming soon)*
7. 🚀 **Phase 6**: Advanced Use Cases *(coming soon)*

## 🔑 Key Features of This Documentation

### **Comprehensive Coverage**
- Complete API documentation for all major modules
- Real-world examples from Tunix repository
- Production-ready code patterns
- Hyperparameter tuning guides

### **Practical Focus**
- Runnable code examples
- Step-by-step walkthroughs
- Common pitfalls and solutions
- Hardware-specific adaptations (GPU/TPU)

### **Well-Organized**
- Progressive learning structure
- Markmap-compatible format
- Cross-referenced sections
- Searchable content

## 📝 Example Topics Covered

### **Fine-Tuning**
- QLoRA/LoRA implementation
- Parameter-efficient training
- Checkpoint management
- Memory optimization

### **Reinforcement Learning**
- GRPO (Group Relative Policy Optimization)
- DPO (Direct Preference Optimization)
- PPO (Proximal Policy Optimization)
- Custom reward functions
- Multi-turn agentic RL

### **Model Compression**
- Logit-based distillation
- Feature-based distillation
- Teacher-student setup
- Compression evaluation

### **Production Topics**
- Distributed training on TPU pods
- Sharding strategies (FSDP, TP, DP)
- Profiling and optimization
- Data pipeline design
- Checkpoint strategies

## 🛠️ Technologies Documented

- **JAX**: Functional programming, transformations, sharding
- **Flax NNX**: Neural network library, Module system
- **Orbax**: Checkpointing and state management
- **Grain**: Data loading and preprocessing
- **vLLM**: Fast inference integration
- **Metrax**: Distributed metrics aggregation
- **TPU**: Hardware optimization

## 💻 Code Examples

Each phase includes extensive code examples:
- **Phase 2**: API usage patterns for every module
- **Phase 3**: Infrastructure setup and configuration
- **Phase 4**: Complete training scripts and notebooks

Example from Phase 4 (GRPO training):
```python
# Configure GRPO
grpo_config = grpo_learner.GRPOConfig(
    num_iterations=1,
    beta=0.001,
    epsilon=0.2,
    learning_rate=1e-5,
)

# Create RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    model=model,
    optimizer=optimizer,
    rollout_config=rollout_config,
    reward_fn=compute_rewards,
    mesh=mesh,
)

# Train
learner = grpo_learner.GRPOLearner(config=grpo_config, rl_cluster=rl_cluster)
for step, batch in enumerate(train_ds):
    metrics = learner.train_step(batch)
```

## 🤝 Contributing

This is a living documentation project. Contributions are welcome!

### **How to Contribute**
1. Read [`Git_Commands.md`](Git_Commands.md) for workflow
2. Follow the existing documentation style
3. Add code examples where helpful
4. Cross-reference related sections
5. Submit clear, well-organized content

### **Areas for Contribution**
- Additional examples and use cases
- Clarifications and corrections
- Performance optimization tips
- Community best practices
- Phase 5 & 6 content

## 📚 Related Resources

- **Tunix GitHub**: https://github.com/google/tunix
- **Tunix Documentation**: Official docs in `docs/` folder
- **JAX Documentation**: https://jax.readthedocs.io/
- **Flax Documentation**: https://flax.readthedocs.io/

## 📮 Feedback

If you find this documentation helpful or have suggestions:
- Open an issue in this repository
- Submit improvements via pull request
- Share feedback on what's working well

## 📄 License

This project is for educational purposes. The Tunix framework itself is licensed under Apache 2.0.

---

**Last Updated**: January 3, 2026  
**Total Documentation**: ~32,000 lines  
**Status**: Phase 1-4 Complete | Phase 5-6 Coming Soon

**Start your Tunix journey today!** 🚀
