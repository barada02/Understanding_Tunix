---
markmap:
  initialExpandLevel: 2
---

# Tunix Learning Roadmap

> **Format:** This file is in Markmap format - you can visualize it as an interactive mindmap!  
> **How to use:** Copy this content to https://markmap.js.org/repl or use a Markmap VSCode extension

---

# Tunix Learning Journey

## ✅ Phase 0: Setup & Overview
- [x] Repository cloning
- [x] Create Understanding_Tunix folder
- [x] Read repository overview
- [x] Understand project scope
- [ ] Review installation requirements

## 📚 Phase 1: High-Level Understanding

### 1.1 Core Concepts
- [ ] What is post-training for LLMs?
- [ ] Why JAX and TPUs?
- [ ] Flax NNX fundamentals
- [ ] Training vs Inference lifecycle

### 1.2 Architecture Overview
- [ ] System architecture diagram
- [ ] Component interaction flow
- [ ] Data flow through training pipeline
- [ ] Multi-host distributed training concepts

### 1.3 Key Technologies Deep Dive
- [ ] JAX basics and transformations
  - [ ] `jax.jit` compilation
  - [ ] `jax.pmap` parallelization
  - [ ] `jax.grad` differentiation
- [ ] Flax NNX module system
  - [ ] Variables and state management
  - [ ] Module composition
- [ ] TPU architecture basics
  - [ ] TPU vs GPU differences
  - [ ] Memory layout considerations
- [ ] Pathways distributed training

---

## 🔧 Phase 2: Core Components Deep Dive

### 2.1 Supervised Fine-Tuning (SFT) Module
- [ ] `tunix/sft/` directory structure
- [ ] Full weight fine-tuning
  - [ ] Training loop implementation
  - [ ] Loss computation
  - [ ] Optimizer configuration
- [ ] PEFT with LoRA/QLoRA
  - [ ] LoRA layer injection
  - [ ] Parameter freezing strategies
  - [ ] Memory optimization
- [ ] Training utilities
  - [ ] `checkpoint_manager.py` - Model checkpointing
  - [ ] `metrics_logger.py` - Logging infrastructure
  - [ ] `progress_bar.py` - Training progress
  - [ ] `profiler.py` - Performance profiling
  - [ ] `sharding_utils.py` - Model sharding
- [ ] DPO implementation
  - [ ] Preference data handling
  - [ ] DPO loss computation
  - [ ] Policy-reference model setup

### 2.2 Reinforcement Learning (RL) Module
- [ ] `tunix/rl/` directory structure
- [ ] RL fundamentals in Tunix
  - [ ] `algorithm_config.py` - Algorithm configuration
  - [ ] `common.py` - Shared RL components
  - [ ] `rl_cluster.py` - Distributed RL setup
  - [ ] `rl_learner.py` - Learner implementation
- [ ] PPO (Proximal Policy Optimization)
  - [ ] Policy network
  - [ ] Value network
  - [ ] Advantage computation
  - [ ] Clipped objective
- [ ] GRPO (Group Relative Policy Optimization)
  - [ ] Group-based optimization
  - [ ] Relative reward computation
  - [ ] Implementation details
- [ ] GSPO-token
  - [ ] Token-level policy optimization
  - [ ] Sequence-level vs token-level differences
- [ ] Rollout infrastructure
  - [ ] `rollout/` directory
  - [ ] Trajectory collection
  - [ ] Reward computation
- [ ] Inference integration
  - [ ] `inference/` directory
  - [ ] vLLM rollout
  - [ ] SGLang rollout
- [ ] Agentic RL (Experimental)
  - [ ] `agentic/` directory
  - [ ] Multi-turn conversations
  - [ ] Tool usage integration
  - [ ] Async rollout

### 2.3 Knowledge Distillation Module
- [ ] `tunix/distillation/` directory structure
- [ ] `distillation_trainer.py` - Core trainer
- [ ] Distillation strategies
  - [ ] Logit strategy (output matching)
  - [ ] Attention transfer
  - [ ] Feature pooling and projection
- [ ] Feature extraction
  - [ ] `feature_extraction/` directory
  - [ ] Intermediate layer extraction
- [ ] Teacher-student model setup
- [ ] Loss functions for distillation

### 2.4 Generation Module
- [ ] `tunix/generate/` directory structure
- [ ] Sampler implementations
  - [ ] `base_sampler.py` - Base sampler interface
  - [ ] `sampler.py` - Standard sampling
  - [ ] `vllm_sampler.py` - vLLM integration
  - [ ] `sglang_jax_sampler.py` - SGLang integration
- [ ] `beam_search.py` - Beam search algorithm
- [ ] `tokenizer_adapter.py` - Tokenizer interface
- [ ] `mappings.py` - Token mappings
- [ ] `vllm_async_driver.py` - Async vLLM driver
- [ ] Sampling strategies
  - [ ] Temperature sampling
  - [ ] Top-k and top-p
  - [ ] Repetition penalty

### 2.5 Models Module
- [ ] `tunix/models/` directory structure
- [ ] `automodel.py` - Automatic model loading
- [ ] Model-specific implementations
  - [ ] Gemma architecture
  - [ ] Gemma 3 updates
  - [ ] Llama 3 architecture
  - [ ] Qwen 2 architecture
  - [ ] Qwen 3 architecture
- [ ] `safetensors_loader.py` - Loading from SafeTensors
- [ ] `safetensors_saver.py` - Saving to SafeTensors
- [ ] `naming.py` - Parameter naming conventions
- [ ] `dummy_model_creator.py` - Testing utilities

### 2.6 CLI & Configuration
- [ ] `tunix/cli/` directory structure
- [ ] `config.py` - Configuration system
- [ ] `base_config.yaml` - Default configurations
- [ ] YAML-based configuration
- [ ] OmegaConf integration
- [ ] Command-line interface
- [ ] Configuration composition and overrides

### 2.7 Utilities
- [ ] `tunix/utils/` directory
- [ ] Common utility functions
- [ ] JAX utilities
- [ ] Data processing helpers
- [ ] Logging and monitoring tools

### 2.8 Performance & Optimization
- [ ] `tunix/perf/` directory
- [ ] Performance profiling tools
- [ ] Export functionality
- [ ] Tracing utilities
- [ ] Memory optimization techniques
- [ ] Micro-batching implementation

---

## 🎯 Phase 3: Advanced Topics

### 3.1 Training Infrastructure
- [ ] Sharding strategies
  - [ ] FSDP (Fully Sharded Data Parallel)
  - [ ] Tensor parallelism
  - [ ] Pipeline parallelism
- [ ] Checkpoint management
  - [ ] Checkpoint formats
  - [ ] Incremental checkpointing
  - [ ] Checkpoint restoration
- [ ] Metrics and logging
  - [ ] Metrax protocol
  - [ ] Custom logging backends
  - [ ] WandB integration
  - [ ] TensorBoard integration
- [ ] System metrics
  - [ ] Memory monitoring
  - [ ] TPU utilization
  - [ ] Training throughput

### 3.2 Distributed Training
- [ ] Multi-host setup
- [ ] Pathways integration
- [ ] Cross-replica communication
- [ ] Gradient synchronization
- [ ] Load balancing

### 3.3 Data Pipeline
- [ ] Dataset loading (TensorFlow Datasets, Grain)
- [ ] Data preprocessing
- [ ] Tokenization strategies
- [ ] Batch construction
- [ ] Data shuffling and sampling

---

## 📖 Phase 4: Examples & Practical Implementation

### 4.1 Jupyter Notebook Examples
- [ ] `qlora_gemma.ipynb` - QLoRA fine-tuning
  - [ ] Setup and configuration
  - [ ] Training loop walkthrough
  - [ ] Results analysis
- [ ] `grpo_gemma.ipynb` - GRPO for math problems
  - [ ] RL environment setup
  - [ ] Reward function design
  - [ ] Training and evaluation
- [ ] `logit_distillation.ipynb` - Model distillation
  - [ ] Teacher-student setup
  - [ ] Distillation process
  - [ ] Performance comparison
- [ ] `dpo_gemma.ipynb` - Preference optimization
  - [ ] Preference dataset
  - [ ] DPO training
  - [ ] Alignment evaluation

### 4.2 Script Examples
- [ ] `scripts/grpo_demo_llama3_qwen2.py`
- [ ] `scripts/llama3_example.py`
- [ ] Setup scripts analysis
  - [ ] TPU notebook setup
  - [ ] vLLM installation

### 4.3 Agentic Examples
- [ ] `examples/agentic/` directory
- [ ] Multi-turn RL examples
- [ ] Tool usage patterns

### 4.4 DeepScaler Examples
- [ ] `examples/deepscaler/` directory
- [ ] Math evaluation
- [ ] Training workflows

---

## 🧪 Phase 5: Testing & Best Practices

### 5.1 Test Structure
- [ ] `tests/` directory organization
- [ ] Unit test patterns
- [ ] Integration test patterns
- [ ] Smoke tests

### 5.2 Test Coverage Areas
- [ ] SFT tests (`tests/sft/`)
- [ ] RL tests (`tests/rl/`)
- [ ] Distillation tests (`tests/distillation/`)
- [ ] Generation tests (`tests/generate/`)
- [ ] Model tests (`tests/models/`)
- [ ] CLI tests (`tests/cli/`)

### 5.3 Testing Best Practices
- [ ] Mocking strategies
- [ ] Fixture usage
- [ ] TPU testing considerations
- [ ] Performance testing

### 5.4 Contributing Guidelines
- [ ] Code style (Pyink formatting)
- [ ] Pull request process
- [ ] Issue reporting
- [ ] Documentation standards

---

## 🚀 Phase 6: Advanced Use Cases

### 6.1 Custom Training Loops
- [ ] Building custom trainers
- [ ] Custom loss functions
- [ ] Custom optimizers

### 6.2 Custom Models
- [ ] Adding new model architectures
- [ ] Model registration
- [ ] Parameter naming conventions

### 6.3 Custom RL Algorithms
- [ ] Extending base RL classes
- [ ] Custom reward functions
- [ ] Custom value functions

### 6.4 Production Deployment
- [ ] Model export strategies
- [ ] Inference optimization
- [ ] Serving patterns

---

## 📝 Phase 7: Documentation & Community

### 7.1 Documentation Deep Dive
- [ ] API documentation (`docs/api/`)
- [ ] Programming guide
- [ ] Gallery and examples
- [ ] ReadTheDocs setup

### 7.2 Community & Collaboration
- [ ] GitHub discussions
- [ ] Issue tracker
- [ ] Collaboration with GRL project
- [ ] Contributing to Tunix

---

## 🎓 Mastery Checklist

### Conceptual Understanding
- [ ] Understand all training paradigms (SFT, RL, Distillation)
- [ ] Grasp JAX and Flax fundamentals
- [ ] Know TPU optimization strategies
- [ ] Understand distributed training concepts

### Practical Skills
- [ ] Can run and modify examples
- [ ] Can configure training via YAML/CLI
- [ ] Can implement custom components
- [ ] Can debug training issues

### Advanced Capabilities
- [ ] Can design custom training pipelines
- [ ] Can optimize for performance
- [ ] Can contribute to the codebase
- [ ] Can deploy models to production

---

## 📊 Progress Tracking

**Current Phase:** Phase 0 ✅  
**Next Phase:** Phase 1 - High-Level Understanding  
**Overall Completion:** 5%

**Notes:**
- Mark items with [x] as you complete them
- Add your own sub-items as needed
- Track blockers or questions in your notes
- Revisit phases as needed for deeper understanding

---

**Last Updated:** December 27, 2025  
**Your Learning Journey Starts Here! 🚀**
