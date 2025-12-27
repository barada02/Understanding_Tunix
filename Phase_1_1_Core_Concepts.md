---
markmap:
  initialExpandLevel: 2
---

# Phase 1.1: Core Concepts

**Learning Objective:** Understand the fundamental concepts of LLM post-training and why Tunix's architecture choices matter.

---

## 1. What is Post-Training for LLMs?

### The LLM Lifecycle

```
Pre-training → Post-training → Deployment
   ↓               ↓              ↓
Learning      Specialization   Inference
Language      & Alignment      & Serving
```

### Pre-training vs Post-training

| Aspect | Pre-training | Post-training |
|--------|-------------|---------------|
| **Goal** | Learn general language patterns | Specialize for specific tasks/behaviors |
| **Data** | Massive unlabeled text (TB-PB) | Smaller curated datasets (GB-TB) |
| **Cost** | Extremely expensive (millions $) | More affordable (thousands-hundreds of thousands $) |
| **Duration** | Weeks to months | Days to weeks |
| **Output** | Base/Foundation model | Instruction-tuned/Aligned model |

### What Post-Training Achieves

1. **Task Specialization**
   - Convert general language model → task-specific expert
   - Examples: coding assistant, math solver, creative writer

2. **Alignment**
   - Make models follow instructions better
   - Reduce harmful/unwanted behaviors
   - Improve helpfulness and honesty

3. **Efficiency**
   - Compress knowledge (distillation)
   - Reduce model size while maintaining performance
   - Optimize for specific use cases

4. **Behavior Shaping**
   - Control output style and format
   - Enforce safety guidelines
   - Improve reasoning capabilities

---

## 2. Post-Training Methods in Tunix

### 2.1 Supervised Fine-Tuning (SFT)

**What it is:** Training on labeled input-output pairs

**When to use:**
- You have quality examples of desired behavior
- Need the model to follow specific formats
- Want to teach specific tasks

**Key Concepts:**
- **Full Fine-Tuning:** Update all model parameters
  - Pro: Maximum flexibility and performance
  - Con: Memory intensive, requires full model storage
  
- **PEFT (Parameter-Efficient Fine-Tuning):** Update only a small subset of parameters
  - **LoRA (Low-Rank Adaptation):** Adds trainable low-rank matrices
  - **QLoRA:** LoRA + quantization for extreme memory efficiency
  - Pro: 90%+ memory savings, faster training
  - Con: Slightly lower performance ceiling

**Example Flow:**
```
Input: "What is 2+2?"
Training: Model learns from examples
Output: "The sum of 2+2 is 4."
```

### 2.2 Reinforcement Learning (RL)

**What it is:** Training models to maximize rewards through trial and error

**When to use:**
- No perfect labeled examples exist
- Need to optimize for complex, multi-step reasoning
- Want to balance multiple objectives (helpfulness vs safety)
- Output quality is subjective or hard to define

**Key Concepts:**
- **Policy:** The model that generates text
- **Reward:** Scalar score indicating output quality
- **Value Function:** Estimates future rewards
- **Advantage:** How much better an action is than average

**RL Algorithms in Tunix:**

1. **PPO (Proximal Policy Optimization)**
   - Most stable RL algorithm
   - Uses value network to reduce variance
   - Clips updates to prevent large policy changes
   - **Use case:** General-purpose RL training

2. **GRPO (Group Relative Policy Optimization)**
   - Simpler than PPO (no value network needed)
   - Groups multiple completions per prompt
   - Compares completions relatively
   - **Use case:** When you can generate multiple responses per prompt

3. **GSPO-token (Token-level GRPO)**
   - Fine-grained token-by-token optimization
   - Better for reasoning tasks
   - **Use case:** Math, coding, step-by-step reasoning

**RL Training Loop:**
```
1. Generate responses with current policy
2. Compute rewards for each response
3. Calculate advantages (what was better/worse than expected)
4. Update policy to increase probability of high-reward actions
5. Repeat
```

### 2.3 Preference Optimization

**What it is:** Learning from human preferences (A vs B comparisons)

**When to use:**
- Have preference data (which output is better)
- Easier to judge than create perfect outputs
- Want to align with human values

**Methods:**

1. **DPO (Direct Preference Optimization)**
   - Direct optimization without explicit reward model
   - More stable than RLHF
   - Requires preference pairs: (prompt, chosen_response, rejected_response)
   
2. **ORPO (Odds Ratio Preference Optimization)**
   - Variant of DPO with different mathematical formulation
   - Can be more sample-efficient

**Example:**
```
Prompt: "Write a friendly greeting"
Option A: "Hey! How can I help you today? 😊"
Option B: "State your query."
Human Preference: A > B
→ Model learns to be friendlier
```

### 2.4 Knowledge Distillation

**What it is:** Transferring knowledge from a large "teacher" model to a smaller "student" model

**When to use:**
- Need faster inference (smaller model)
- Want to reduce deployment costs
- Have a strong teacher model
- Need to compress capabilities

**Distillation Strategies in Tunix:**

1. **Logit Distillation**
   - Student learns to match teacher's output probabilities
   - Most common approach
   - **Loss:** KL divergence between teacher and student distributions

2. **Attention Transfer**
   - Student learns to match teacher's attention patterns
   - Helps with understanding which tokens are important

3. **Feature Projection**
   - Match intermediate layer representations
   - Works even if architectures differ

**Distillation Flow:**
```
Teacher (large model) → Soft targets → Student (small model)
         ↓                              ↓
    Predictions               Learns to mimic teacher
```

---

## 3. Why JAX and TPUs?

### The JAX Advantage

**JAX = NumPy + Automatic Differentiation + XLA Compilation + Hardware Acceleration**

#### Key JAX Features Leveraged by Tunix:

1. **Functional Programming Paradigm**
   ```python
   # Pure functions = reproducible, testable, composable
   def loss_fn(params, batch):
       logits = model.apply(params, batch)
       return jnp.mean((logits - batch['targets'])**2)
   ```

2. **Automatic Differentiation**
   ```python
   # No manual gradient computation needed
   grad_fn = jax.grad(loss_fn)
   gradients = grad_fn(params, batch)
   ```

3. **JIT Compilation**
   ```python
   # Compile to optimized XLA code
   @jax.jit
   def train_step(params, batch):
       loss, grads = jax.value_and_grad(loss_fn)(params, batch)
       return loss, grads
   ```

4. **Automatic Parallelization**
   ```python
   # Distribute across multiple devices
   @jax.pmap  # Parallel map across devices
   def parallel_train_step(params, batch):
       return train_step(params, batch)
   ```

5. **Transformations Compose**
   ```python
   # Can combine jit, pmap, grad, vmap freely
   @jax.jit
   @jax.pmap
   @jax.grad
   def optimized_gradient(params, batch):
       return loss_fn(params, batch)
   ```

### Why Not PyTorch?

| Aspect | JAX | PyTorch |
|--------|-----|---------|
| **TPU Support** | First-class, native | Limited, through XLA bridge |
| **Compilation** | XLA-optimized, aggressive | TorchScript, less aggressive |
| **Functional Style** | Pure functions, immutable | Object-oriented, mutable |
| **Parallelization** | Built-in transformations | Manual implementation |
| **Performance on TPU** | ⚡ Excellent | 😐 Good but not optimal |

### TPUs vs GPUs for LLM Training

#### TPU (Tensor Processing Unit) Advantages:

1. **Matrix Multiplication Optimized**
   - LLM training = 90%+ matrix multiplications
   - TPUs excel at large matmuls (128x128 systolic arrays)

2. **High Bandwidth Memory (HBM)**
   - Faster memory access than GPU memory
   - Critical for large models

3. **Interconnect**
   - TPU Pods connected via ICI (Inter-Chip Interconnect)
   - 2D/3D mesh topology for efficient communication
   - Lower latency than GPU NVLink

4. **Cost Efficiency**
   - Google Cloud TPU v4: ~$1.35/hr per chip
   - Better price-performance for large-scale training

5. **Pathways Architecture**
   - Seamless multi-host scaling
   - Can scale to thousands of chips
   - Unified view across hosts

#### When to Use TPUs:

✅ **Good for:**
- Large-scale transformer training
- Batch-heavy workloads
- Production training pipelines
- Cost-sensitive projects
- Multi-host distributed training

❌ **Not ideal for:**
- Small experimental models
- Custom CUDA kernels
- GPU-optimized libraries
- Development without cloud access

### Pathways: Google's Distributed Training System

**What Pathways Does:**
- Manages multi-host TPU training
- Provides unified view of distributed system
- Handles fault tolerance and recovery
- Optimizes communication patterns

**Key Concepts:**
```
Single Host: 8 TPU chips
    ↓
TPU Pod: Multiple hosts interconnected
    ↓
Pathways: Treats entire pod as single machine
```

**Benefits for Tunix:**
- Transparent scaling (code looks single-host)
- No manual communication code
- Automatic sharding and replication
- Production-grade reliability

---

## 4. Flax NNX Fundamentals

### What is Flax NNX?

**Flax NNX (Neural Networks eXtended)** is JAX's neural network library that provides:
- Pythonic, object-oriented API
- State management for neural networks
- Module composition system
- Integration with JAX transformations

### Core NNX Concepts

#### 1. Modules

```python
from flax import nnx

class MyModel(nnx.Module):
    def __init__(self, features: int):
        # Parameters are automatically tracked
        self.dense = nnx.Linear(features, features)
        self.dropout = nnx.Dropout(0.1)
    
    def __call__(self, x):
        x = self.dense(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        return x
```

#### 2. Variables and State

NNX manages different types of state:
- **Params:** Trainable parameters
- **BatchStats:** Moving averages (e.g., BatchNorm)
- **Dropout:** RNG state
- **Cache:** KV cache for inference

```python
model = MyModel(128)

# Access parameters
params = nnx.state(model, nnx.Param)

# Access all state
full_state = nnx.state(model)
```

#### 3. Module Composition

```python
class Encoder(nnx.Module):
    def __init__(self):
        self.layer1 = MyModel(256)
        self.layer2 = MyModel(256)
    
    def __call__(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
```

#### 4. Training with NNX

```python
# Create model and optimizer
model = MyModel(features=128)
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# Training step
@nnx.jit
def train_step(model, optimizer, batch):
    def loss_fn(model):
        logits = model(batch['x'])
        return jnp.mean((logits - batch['y'])**2)
    
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss
```

### NNX vs Flax Linen

| Feature | NNX | Linen (old API) |
|---------|-----|-----------------|
| **Style** | Stateful, Pythonic | Functional, explicit |
| **State** | Implicit tracking | Manual passing |
| **API** | Modern, intuitive | Verbose |
| **Tunix Use** | ✅ Primary | ❌ Deprecated |

### Why NNX for Tunix?

1. **Simpler Code**
   - Less boilerplate
   - Easier to read and maintain
   - More Pythonic

2. **Better State Management**
   - Automatic parameter tracking
   - Clear separation of state types
   - Easier checkpointing

3. **Module Composition**
   - Easy to build complex architectures
   - Reusable components
   - Clear hierarchy

4. **Training Loops**
   - Integrated optimizer
   - Clean gradient computation
   - Seamless with JAX transformations

---

## 5. Training vs Inference Lifecycle

### Training Lifecycle

```
1. Data Loading
   ↓
2. Tokenization
   ↓
3. Batching & Sharding
   ↓
4. Forward Pass
   ↓
5. Loss Computation
   ↓
6. Backward Pass (Gradients)
   ↓
7. Optimizer Update
   ↓
8. Metrics Logging
   ↓
9. Checkpointing
   ↓
10. Repeat
```

**Key Characteristics:**
- Focus on gradient computation
- Backward pass takes ~2x forward pass time
- Memory intensive (stores activations)
- Batch processing for efficiency
- Regular checkpointing for recovery

### Inference Lifecycle

```
1. Load Model Checkpoint
   ↓
2. Receive Prompt
   ↓
3. Tokenization
   ↓
4. Forward Pass (autoregressive generation)
   ↓
5. Sampling/Decoding
   ↓
6. Repeat until EOS or max length
   ↓
7. Return Generated Text
```

**Key Characteristics:**
- No gradient computation
- Lower memory usage
- Sequential generation (token-by-token)
- Can use KV caching for speed
- Latency-sensitive

### RL: Training + Inference Combined

**Unique Challenge:** RL needs BOTH simultaneously

```
[Inference Phase - Rollout]
Generate completions with current policy
   ↓
[Reward Computation]
Score generated completions
   ↓
[Training Phase - Learning]
Update policy based on rewards
   ↓
Repeat
```

**Why This Matters in Tunix:**
- Need efficient inference during training
- vLLM/SGLang integration for fast rollouts
- Separate meshes for training and inference
- Async rollout to hide latency

### Memory Considerations

| Phase | Memory Usage | What's Stored |
|-------|--------------|---------------|
| **Training** | High | Parameters + Gradients + Optimizer state + Activations |
| **Inference** | Low | Parameters + KV cache |
| **RL Rollout** | Medium | Parameters + Generated samples + KV cache |
| **RL Training** | High | Training phase + Rollout data |

---

## 6. Key Terminology

### Model Terms

- **Base Model:** Pre-trained foundation model
- **Checkpoint:** Saved model weights at a point in time
- **LoRA Rank:** Dimensionality of LoRA adapter (typical: 8-64)
- **Context Length:** Maximum sequence length model can handle
- **Vocab Size:** Number of tokens in tokenizer

### Training Terms

- **Epoch:** One complete pass through dataset
- **Batch Size:** Number of examples per training step
- **Micro-batch:** Subset of batch processed at once (for memory)
- **Gradient Accumulation:** Summing gradients over micro-batches
- **Learning Rate:** Step size for optimizer updates
- **Warmup:** Gradual increase of learning rate at start

### Distributed Training Terms

- **Sharding:** Splitting model/data across devices
- **FSDP:** Fully Sharded Data Parallel (split everything)
- **Mesh:** Logical arrangement of devices
- **Replica:** Copy of model on different devices
- **Pipeline Parallelism:** Split model layers across devices
- **Tensor Parallelism:** Split individual layers across devices

### RL-Specific Terms

- **Rollout:** Generating samples with current policy
- **Trajectory:** Sequence of states and actions
- **Episode:** Complete rollout from start to end
- **Return:** Total reward over trajectory
- **Advantage:** How good an action is relative to average
- **PPO Clip:** Constraint on policy updates
- **KL Divergence:** Measure of policy change

### Inference Terms

- **Autoregressive:** Generating one token at a time
- **KV Cache:** Cached key-value pairs for attention
- **Beam Search:** Keeping top-k hypotheses during generation
- **Temperature:** Controls randomness in sampling
- **Top-k/Top-p:** Sampling strategies
- **Greedy Decoding:** Always pick highest probability token

---

## 7. Success Metrics

### SFT Metrics
- **Training Loss:** How well model fits training data
- **Validation Loss:** How well model generalizes
- **Perplexity:** exp(loss), lower is better
- **Throughput:** Tokens/second, samples/second

### RL Metrics
- **Average Reward:** Mean reward across rollouts
- **Reward Variance:** Stability of rewards
- **KL Divergence:** How much policy changed
- **Advantage Mean:** Expected improvement
- **Policy Loss:** RL optimization objective
- **Value Loss:** Critic error (PPO)

### Distillation Metrics
- **KL Divergence:** Student vs teacher output difference
- **Model Size Reduction:** Compression ratio
- **Performance Retention:** Student accuracy / Teacher accuracy
- **Inference Speedup:** Teacher time / Student time

---

## 🎯 Phase 1.1 Checklist

- [ ] Understand difference between pre-training and post-training
- [ ] Know when to use SFT vs RL vs Distillation
- [ ] Grasp why JAX + TPUs are powerful for LLM training
- [ ] Understand Flax NNX module system basics
- [ ] Know the difference between training and inference
- [ ] Familiar with key terminology
- [ ] Ready to dive into architecture details (Phase 1.2)

---

**Next:** [Phase 1.2 - Architecture Overview](Phase_1_2_Architecture_Overview.md)
