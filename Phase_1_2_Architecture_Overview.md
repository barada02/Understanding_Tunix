---
markmap:
  initialExpandLevel: 2
---

# Phase 1.2: Architecture Overview

**Learning Objective:** Understand how Tunix components fit together and interact to enable LLM post-training.

---

## 1. High-Level System Architecture

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                        Tunix Library                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   SFT    │  │    RL    │  │ Distill  │  │ Generate │  │
│  │  Module  │  │  Module  │  │  Module  │  │  Module  │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │             │              │             │         │
│       └─────────────┴──────────────┴─────────────┘         │
│                       │                                     │
│              ┌────────▼────────┐                           │
│              │  Core Trainer   │                           │
│              │   (PeftTrainer) │                           │
│              └────────┬────────┘                           │
│                       │                                     │
│       ┌───────────────┼───────────────┐                   │
│       │               │               │                    │
│  ┌────▼────┐    ┌────▼────┐    ┌────▼────┐              │
│  │ Models  │    │  Utils  │    │  Perf   │              │
│  │ Module  │    │ Module  │    │ Module  │              │
│  └─────────┘    └─────────┘    └─────────┘              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                         │
                ┌────────┴────────┐
                │                 │
        ┌───────▼────────┐  ┌────▼──────┐
        │  JAX/Flax NNX  │  │ Pathways  │
        │   (Compute)    │  │(Distributed)│
        └────────┬────────┘  └────┬──────┘
                 │                │
                 └────────┬───────┘
                          │
                    ┌─────▼─────┐
                    │    TPU    │
                    │  Hardware │
                    └───────────┘
```

### Design Philosophy

**Layered Architecture:**
1. **Application Layer:** High-level training paradigms (SFT, RL, Distillation)
2. **Core Training Layer:** Shared training infrastructure (PeftTrainer)
3. **Model Layer:** Model implementations and loading
4. **Framework Layer:** JAX, Flax NNX, Pathways
5. **Hardware Layer:** TPU execution

**Key Principles:**
- **Modularity:** Each component is self-contained and composable
- **Extensibility:** Easy to add new algorithms, models, strategies
- **Performance:** Optimized for TPU execution
- **Flexibility:** Support multiple training paradigms

---

## 2. Core Components Breakdown

### 2.1 PeftTrainer (Foundation)

**Location:** `tunix/sft/peft_trainer.py`

**Role:** The foundational trainer that ALL other trainers build upon

**Key Responsibilities:**
1. **Training Loop Management**
   - Execute train steps
   - Manage iteration and global steps
   - Handle gradient accumulation

2. **Model Management**
   - Parameter initialization
   - State management (params, optimizer state)
   - Model sharding across devices

3. **Checkpointing**
   - Save/restore model checkpoints
   - Save/restore optimizer state
   - Save/restore training state

4. **Metrics & Logging**
   - Log training metrics
   - Track training progress
   - Integration with logging backends

5. **Performance Optimization**
   - JIT compilation
   - Profiling integration
   - Memory optimization

**Core Components:**

```python
class PeftTrainer:
    model: nnx.Module           # The model being trained
    optimizer: nnx.Optimizer    # The optimizer (wraps optax)
    training_config: TrainingConfig
    
    # Key methods:
    def train(data_iterator) -> None
    def train_step(batch) -> loss
    def eval_step(batch) -> metrics
    def save_checkpoint() -> None
    def restore_checkpoint() -> None
```

**Training Flow:**

```
1. Initialize trainer with model + optimizer
   ↓
2. Load checkpoint (if resuming)
   ↓
3. For each training step:
   ├─ Get batch from iterator
   ├─ Prepare inputs (shard, tokenize)
   ├─ Run train_step (forward + backward)
   ├─ Accumulate gradients
   ├─ Update weights
   ├─ Log metrics
   └─ Save checkpoint (if needed)
   ↓
4. Training complete
```

### 2.2 SFT Module

**Location:** `tunix/sft/`

**Components:**

1. **PeftTrainer** (`peft_trainer.py`)
   - Base trainer implementation
   - Supports full fine-tuning and LoRA/QLoRA

2. **DPOTrainer** (`dpo/dpo_trainer.py`)
   - Extends PeftTrainer
   - Implements Direct Preference Optimization
   - Uses preference pairs for training

3. **ORPOTrainer** (`dpo/dpo_trainer.py`)
   - Variant of DPO
   - Odds Ratio Preference Optimization

4. **Support Components:**
   - `checkpoint_manager.py` - Checkpoint management
   - `metrics_logger.py` - Logging infrastructure
   - `sharding_utils.py` - Model/data sharding
   - `progress_bar.py` - Training progress display
   - `profiler.py` - Performance profiling
   - `utils.py` - Common utilities

**Data Flow:**

```
Dataset → DataLoader → Batch
              ↓
        Tokenization
              ↓
         Sharding (across devices)
              ↓
     PeftTrainer.train_step()
              ↓
    Forward Pass → Loss → Gradients → Update
              ↓
      Metrics Logging
```

### 2.3 RL Module

**Location:** `tunix/rl/`

**Architecture:** More complex than SFT due to RL requirements

**Key Components:**

1. **RLCluster** (`rl_cluster.py`)
   - **Central orchestrator** for RL training
   - Manages multiple models (actor, critic, reference, reward)
   - Handles different device meshes
   - Coordinates training and rollout

2. **RLLearner** (`rl_learner.py`)
   - Abstract base for RL algorithms
   - Manages rollout → training cycle
   - Handles advantage computation
   - Coordinates data flow

3. **Algorithm Implementations:**
   - `ppo/ppo_learner.py` - PPO algorithm
   - `grpo/grpo_learner.py` - GRPO algorithm
   - `grpo/drgrpo_learner.py` - Divergence Regularized GRPO
   - `grpo/dapo_learner.py` - Direct Advantage Policy Optimization

4. **Rollout System** (`rollout/`)
   - `base_rollout.py` - Abstract rollout interface
   - `vanilla_rollout.py` - Standard JAX rollout
   - vLLM integration - Fast inference engine
   - SGLang integration - Structured generation

5. **Support Components:**
   - `trainer.py` - RL-specific trainer (extends PeftTrainer)
   - `common.py` - Shared RL data structures
   - `algorithm_config.py` - Algorithm configuration
   - `utils.py` - RL utilities

**RLCluster Architecture:**

```
┌─────────────────────────────────────────────────────┐
│                   RLCluster                         │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │  Actor   │  │ Critic   │  │Reference │        │
│  │ Trainer  │  │ Trainer  │  │  Model   │        │
│  │  (Policy)│  │ (Value)  │  │  (Fixed) │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │             │              │               │
│  [Mesh: Actor]  [Mesh: Actor]  [Mesh: Ref]       │
│                                                     │
│  ┌──────────┐  ┌──────────────────────────┐      │
│  │  Reward  │  │     Rollout Engine       │      │
│  │  Model   │  │ (vLLM/SGLang/Vanilla)    │      │
│  └────┬─────┘  └────┬─────────────────────┘      │
│       │             │                              │
│  [Mesh: Reward] [Mesh: Rollout]                  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**RL Training Flow:**

```
1. RLCluster initialization
   ├─ Load actor model (policy to train)
   ├─ Load critic model (value estimator, for PPO)
   ├─ Load reference model (original policy, frozen)
   ├─ Load reward model (optional)
   └─ Setup rollout engine
   ↓
2. For each training iteration:
   │
   ├─ ROLLOUT PHASE (inference):
   │  ├─ Get prompts from dataset
   │  ├─ Generate completions with current policy
   │  ├─ Compute rewards for completions
   │  └─ Compute advantages (how good each completion was)
   │
   ├─ TRAINING PHASE:
   │  ├─ Shuffle rollout data
   │  ├─ Split into mini-batches
   │  ├─ For each mini-batch:
   │  │  ├─ Compute policy loss
   │  │  ├─ Compute value loss (PPO only)
   │  │  ├─ Update actor (and critic)
   │  │  └─ Clip updates to prevent large changes
   │  └─ Log metrics
   │
   └─ Checkpoint models
   ↓
3. Training complete
```

**Colocated vs Disaggregated Setup:**

**Colocated:** All components on same mesh
```
Actor, Critic, Reference, Rollout → Same TPU Pod
```
- Simpler setup
- Good for smaller models
- Less communication overhead

**Disaggregated:** Different meshes for different roles
```
Actor + Critic → Training Mesh (e.g., v4-128)
Rollout + Reference → Inference Mesh (e.g., v4-64)
```
- Better resource utilization
- Async rollout possible
- Scales better for large models

### 2.4 Distillation Module

**Location:** `tunix/distillation/`

**Components:**

1. **DistillationTrainer** (`distillation_trainer.py`)
   - Extends PeftTrainer
   - Manages teacher and student models
   - Applies distillation strategies

2. **Strategies** (`strategies/`)
   - Different distillation approaches
   - Logit matching, attention transfer, etc.
   - Feature extraction and projection

**Architecture:**

```
┌────────────────────────────────────────┐
│      DistillationTrainer               │
├────────────────────────────────────────┤
│                                        │
│  ┌──────────┐      ┌──────────┐      │
│  │ Teacher  │      │ Student  │      │
│  │  Model   │─────▶│  Model   │      │
│  │ (Frozen) │      │(Training)│      │
│  └────┬─────┘      └────┬─────┘      │
│       │                 │              │
│       │   ┌─────────────┘              │
│       │   │                            │
│  ┌────▼───▼─────┐                     │
│  │  Strategy    │                     │
│  │  (Logit/     │                     │
│  │  Attention/  │                     │
│  │  Feature)    │                     │
│  └──────┬───────┘                     │
│         │                              │
│  ┌──────▼───────┐                     │
│  │ Loss Compute │                     │
│  └──────────────┘                     │
│                                        │
└────────────────────────────────────────┘
```

**Training Flow:**

```
1. Initialize teacher (frozen) and student (trainable)
   ↓
2. For each batch:
   ├─ Forward pass through teacher
   ├─ Extract teacher outputs/features
   ├─ Forward pass through student
   ├─ Extract student outputs/features
   ├─ Compute distillation loss via strategy
   ├─ Backprop through student only
   └─ Update student parameters
   ↓
3. Student learns to mimic teacher
```

### 2.5 Generation Module

**Location:** `tunix/generate/`

**Purpose:** Text generation and sampling during inference/rollout

**Components:**

1. **Samplers:**
   - `sampler.py` - Standard JAX sampler
   - `vllm_sampler.py` - vLLM integration
   - `sglang_jax_sampler.py` - SGLang integration
   - `base_sampler.py` - Abstract interface

2. **Support:**
   - `tokenizer_adapter.py` - Unified tokenizer interface
   - `beam_search.py` - Beam search implementation
   - `mappings.py` - Token mapping utilities
   - `vllm_async_driver.py` - Async vLLM driver

**Why Multiple Samplers?**

| Sampler | Use Case | Speed | Features |
|---------|----------|-------|----------|
| **Vanilla (JAX)** | Simple use, full control | Medium | Full customization |
| **vLLM** | High-throughput inference | Fast | PagedAttention, batching |
| **SGLang** | Structured generation | Fast | Constrained decoding, DSL |

**Generation Pipeline:**

```
Input Prompt
     ↓
Tokenization
     ↓
┌────────────────┐
│ Sampler Engine │
│  (JAX/vLLM/    │
│   SGLang)      │
└────┬───────────┘
     ↓
Autoregressive Generation:
  ├─ Forward pass
  ├─ Sample next token
  ├─ Append to sequence
  └─ Repeat until EOS
     ↓
Detokenization
     ↓
Output Text
```

### 2.6 Models Module

**Location:** `tunix/models/`

**Purpose:** Model definitions and loading utilities

**Components:**

1. **Model Families:**
   - `gemma/` - Gemma model implementation
   - `gemma3/` - Gemma 3 implementation
   - `llama3/` - Llama 3 implementation
   - `qwen2/` - Qwen 2 implementation
   - `qwen3/` - Qwen 3 implementation

2. **Utilities:**
   - `automodel.py` - Automatic model loading
   - `safetensors_loader.py` - Load from SafeTensors
   - `safetensors_saver.py` - Save to SafeTensors
   - `naming.py` - Parameter naming conventions
   - `dummy_model_creator.py` - Testing utilities

**Model Loading Flow:**

```
Model Path/ID
     ↓
AutoModel.from_pretrained()
     ↓
┌─────────────────────┐
│ Detect model type   │
│ (Gemma/Llama/Qwen)  │
└─────┬───────────────┘
      ↓
┌─────────────────────┐
│ Load architecture   │
│ (config.json)       │
└─────┬───────────────┘
      ↓
┌─────────────────────┐
│ Load weights        │
│ (safetensors/       │
│  pytorch_model.bin) │
└─────┬───────────────┘
      ↓
┌─────────────────────┐
│ Shard across devices│
└─────┬───────────────┘
      ↓
Ready Model (nnx.Module)
```

### 2.7 CLI Module

**Location:** `tunix/cli/`

**Purpose:** Command-line interface for training

**Components:**
- `config.py` - Configuration parsing
- `base_config.yaml` - Default configurations
- `grpo_main.py` - GRPO training entry point

**Configuration System:**

```
YAML Config Files
     ↓
OmegaConf Parser
     ↓
Validated Config Objects
     ↓
Trainer Initialization
```

**Config Composition:**

```yaml
# base_config.yaml (defaults)
model:
  name: "google/gemma-2b"
  
training:
  max_steps: 1000
  learning_rate: 1e-4

# user_config.yaml (overrides)
training:
  max_steps: 5000
  
# Command line (final overrides)
$ tunix train --config base_config.yaml,user_config.yaml \
    --training.learning_rate=5e-5
```

---

## 3. Data Flow Through System

### 3.1 SFT Data Flow

```
Raw Dataset
     ↓
Dataset Loading (TF Datasets/Grain)
     ↓
Tokenization
     ↓
Batching
     ↓
Sharding across devices
     ↓
PeftTrainer.train_step()
     │
     ├─ Forward: model(input_tokens)
     ├─ Loss: cross_entropy(logits, targets)
     ├─ Backward: grad(loss)
     └─ Update: optimizer.update(grads)
     ↓
Metrics Logging
     ↓
Checkpoint Saving
```

### 3.2 RL Data Flow

```
Prompt Dataset
     ↓
RLCluster.rollout()
     │
     ├─ Load prompts
     ├─ Generate completions (inference)
     ├─ Compute rewards
     └─ Compute advantages
     ↓
TrainExample batch
     ↓
RLLearner.train()
     │
     ├─ Shuffle data
     ├─ Split into mini-batches
     └─ For each mini-batch:
         │
         ├─ Compute policy loss
         ├─ Compute value loss (PPO)
         ├─ Update actor/critic
         └─ KL regularization
     ↓
Metrics Logging
     ↓
Checkpoint Saving
```

### 3.3 Distillation Data Flow

```
Training Dataset
     ↓
Tokenization & Batching
     ↓
DistillationTrainer.train_step()
     │
     ├─ Teacher forward (inference)
     │   └─ Extract features/logits
     │
     ├─ Student forward (training)
     │   └─ Extract features/logits
     │
     ├─ Strategy.compute_loss()
     │   └─ Compare teacher vs student
     │
     ├─ Backward through student
     └─ Update student parameters
     ↓
Metrics Logging
     ↓
Checkpoint Saving
```

---

## 4. Component Interaction Patterns

### 4.1 Trainer Hierarchy

```
           PeftTrainer
           (Base class)
                 │
    ┌────────────┼────────────┐
    │            │            │
DPOTrainer  DistillationTrainer  RL Trainer
    │                             │
ORPOTrainer                      │
                         ┌───────┴───────┐
                         │               │
                    PPOLearner      GRPOLearner
                                        │
                                ┌───────┴───────┐
                                │               │
                          DrGRPOLearner    DAPOLearner
```

**Inheritance Benefits:**
- Code reuse (checkpointing, metrics, etc.)
- Consistent API across training paradigms
- Easy to extend with new algorithms

### 4.2 Model Management Pattern

```
User Code
    ↓
┌─────────────────────────┐
│   Model Specification   │
│ (path or config)        │
└─────────┬───────────────┘
          ↓
┌─────────────────────────┐
│    AutoModel.load()     │
└─────────┬───────────────┘
          ↓
    ┌─────┴─────┐
    │  Sharding │
    │  Strategy │
    └─────┬─────┘
          ↓
┌─────────────────────────┐
│  Distributed Model      │
│  (across devices)       │
└─────────┬───────────────┘
          ↓
     Trainer Usage
```

### 4.3 Logging Pattern

```
Training Event
     ↓
MetricsLogger.log()
     ↓
┌──────────────────────┐
│  Logger broadcasts   │
│  to all backends     │
└──────┬───────────────┘
       │
   ┌───┴────┬────────┬────────┐
   │        │        │        │
   ▼        ▼        ▼        ▼
Console  TensorBoard WandB  Custom
Backend  Backend     Backend Backend
```

**Protocol-based design allows pluggable backends**

### 4.4 Checkpoint Pattern

```
Trainer State:
├─ Model parameters
├─ Optimizer state
├─ Training step counter
├─ Data iterator state
└─ Random seeds

     ↓
CheckpointManager.save()
     ↓
┌──────────────────────┐
│  Orbax Checkpoint    │
│  (efficient format)  │
└──────┬───────────────┘
       │
   Saved to disk
       │
CheckpointManager.restore()
       ↓
Trainer resumes seamlessly
```

---

## 5. Multi-Host Distributed Training Architecture

### 5.1 Device Mesh Concept

**Logical view:**
```
Mesh shape: (2, 4)  # 2 FSDP replicas, 4 tensor parallel
Device IDs:
[0, 1, 2, 3]
[4, 5, 6, 7]
```

**Physical mapping:**
```
TPU Pod: 2 hosts × 4 chips each = 8 total chips

Host 0: Chips [0, 1, 2, 3]
Host 1: Chips [4, 5, 6, 7]

Mesh coordinates data/model sharding
```

### 5.2 Sharding Strategies

**FSDP (Fully Sharded Data Parallel):**
```
Model split across devices:

Device 0: Layers 0-3
Device 1: Layers 4-7
Device 2: Layers 8-11
Device 3: Layers 12-15

Each device:
- Holds full optimizer state for its layers
- Processes different data batch
- All-gathers parameters when needed
```

**Tensor Parallelism:**
```
Single layer split across devices:

Linear layer: [4096, 4096]
Device 0: [4096, 1024]
Device 1: [4096, 1024]
Device 2: [4096, 1024]
Device 3: [4096, 1024]
```

**Combined (FSDP + TP):**
```
Mesh: (fsdp=2, tensor=4)

Layer 0-7:  FSDP replica 0, split across TP 0-3
Layer 8-15: FSDP replica 1, split across TP 0-3
```

### 5.3 Communication Patterns

**Collective Operations:**

1. **All-Reduce:** Gradients aggregation
```
Each device: local gradients
     ↓
All-Reduce sum
     ↓
Each device: averaged gradients
```

2. **All-Gather:** Parameter reconstruction
```
Device 0: Params 0-3
Device 1: Params 4-7
     ↓
All-Gather
     ↓
All devices: Full params 0-7
```

3. **Reduce-Scatter:** Optimizer state distribution
```
All devices: Full gradients
     ↓
Reduce-Scatter
     ↓
Each device: Subset of reduced gradients
```

---

## 6. Memory Management

### 6.1 Memory Components

**During Training:**
```
Total Memory = Model Params + Optimizer State + Gradients + Activations

Example (7B model, bf16, Adam):
- Parameters: 14 GB
- Optimizer (Adam): 28 GB (2x params)
- Gradients: 14 GB
- Activations: varies by batch size
─────────────────────────────
Total: ~56 GB + activations
```

**FSDP Reduction:**
```
With 8-way FSDP:
Per-device memory: ~7 GB + activations/8

Enables training larger models!
```

### 6.2 QLoRA Memory Savings

```
Full Fine-tuning:
├─ All parameters trainable: 100%
├─ Full optimizer state: 100%
└─ Memory: VERY HIGH

LoRA:
├─ Base frozen: ~95% params
├─ LoRA trainable: ~5% params
├─ Optimizer state: 5% only
└─ Memory: ~25% of full

QLoRA:
├─ Base quantized (4-bit): ~25% memory
├─ LoRA trainable (16-bit): ~5% params
├─ Optimizer state: 5% only
└─ Memory: ~15% of full
```

---

## 7. Performance Optimization Layers

### 7.1 Compilation (XLA/JIT)

```
Python Code
     ↓
JAX traces function
     ↓
XLA Compiler
     ↓
Optimized TPU kernels
     ↓
Cached for reuse
```

**What JIT optimizes:**
- Operator fusion
- Memory layout optimization
- Constant folding
- Dead code elimination

### 7.2 Micro-batching

```
Global batch: 128 samples
Micro-batch size: 32

Process in 4 micro-batches:
├─ Micro-batch 1: Forward + Backward
├─ Micro-batch 2: Forward + Backward
├─ Micro-batch 3: Forward + Backward
├─ Micro-batch 4: Forward + Backward
└─ Aggregate gradients → Single update

Reduces peak memory!
```

### 7.3 Profiling Integration

```
Training Loop
     ↓
Profiler captures:
├─ Computation time per step
├─ Memory usage
├─ Communication time
├─ Compilation time
└─ Device utilization
     ↓
Export traces
     ↓
Analyze bottlenecks
```

---

## 8. Extension Points

### 8.1 Adding New Algorithms

```python
# Extend RLLearner for new RL algorithm
class MyCustomRLLearner(RLLearner):
    def _generate_and_compute_advantage(self, ...):
        # Custom advantage computation
        pass
    
    def _compute_policy_loss(self, ...):
        # Custom loss function
        pass
```

### 8.2 Adding New Models

```python
# Implement model architecture
class MyModel(nnx.Module):
    def __init__(self, config):
        # Define layers
        pass
    
    def __call__(self, inputs):
        # Forward pass
        pass

# Register with AutoModel
# Add to tunix/models/
```

### 8.3 Custom Distillation Strategies

```python
class MyStrategy(BaseStrategy):
    def compute_loss(self, teacher_out, student_out):
        # Custom distillation loss
        pass
```

### 8.4 Custom Logging Backends

```python
class MyBackend:
    def log_scalar(self, event, value, **kwargs):
        # Custom logging logic
        pass
    
    def close(self):
        pass

# Use with MetricsLoggerOptions
```

---

## 9. Architecture Diagrams

### 9.1 Complete RL Training Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Application                        │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      RLLearner                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │            Training Coordination Loop                 │  │
│  └──────────────────┬────────────┬──────────────────────┘  │
│                     │            │                          │
│         ┌───────────▼─────┐   ┌──▼──────────────┐         │
│         │  Rollout Phase  │   │ Training Phase  │         │
│         └─────────┬───────┘   └──┬──────────────┘         │
└───────────────────┼──────────────┼──────────────────────────┘
                    │              │
        ┌───────────▼──────┐   ┌──▼─────────────────┐
        │    RLCluster     │   │   Actor Trainer    │
        │  ┌────────────┐  │   │                    │
        │  │  Rollout   │  │   │  ┌──────────────┐ │
        │  │  Engine    │  │   │  │ PeftTrainer  │ │
        │  │(vLLM/JAX)  │  │   │  └──────────────┘ │
        │  └────────────┘  │   └────────────────────┘
        │  ┌────────────┐  │
        │  │ Reference  │  │
        │  │   Model    │  │
        │  └────────────┘  │
        │  ┌────────────┐  │
        │  │  Reward    │  │
        │  │   Model    │  │
        │  └────────────┘  │
        └──────────────────┘
```

### 9.2 Request Flow Diagram

```
User Request: "Fine-tune Gemma on my data"
         ↓
    PeftTrainer(model, optimizer, config)
         ↓
    trainer.train(data_iterator)
         ↓
    ┌────────────────────────┐
    │  Training Loop Start   │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Get next batch        │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Shard across devices  │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Forward pass (JIT)    │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Compute loss          │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Backward pass (grad)  │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  All-reduce gradients  │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Optimizer update      │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Log metrics           │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Save checkpoint?      │
    └──────────┬─────────────┘
               │
    ┌──────────▼─────────────┐
    │  Check if done         │
    └──────────┬─────────────┘
               │
         Repeat or End
```

---

## 🎯 Phase 1.2 Checklist

- [ ] Understand overall system architecture
- [ ] Know role of PeftTrainer as foundation
- [ ] Grasp how SFT module works
- [ ] Understand RL module complexity and RLCluster
- [ ] Know data flow through each training paradigm
- [ ] Understand multi-host distributed training
- [ ] Familiar with memory management strategies
- [ ] Ready for deep dive into JAX/TPU technologies (Phase 1.3)

---

**Previous:** [Phase 1.1 - Core Concepts](Phase_1_1_Core_Concepts.md)  
**Next:** [Phase 1.3 - Key Technologies Deep Dive](Phase_1_3_Key_Technologies.md)
