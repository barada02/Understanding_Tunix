---
markmap:
  initialExpandLevel: 2
---

# Phase 1.3: Key Technologies Deep Dive

**Learning Objective:** Deep understanding of JAX, Flax NNX, TPU architecture, and Pathways - the technologies that power Tunix.

---

## 1. JAX: The Foundation

### 1.1 What Makes JAX Special?

**JAX = NumPy + Autograd + XLA + Hardware Acceleration**

```python
import jax
import jax.numpy as jnp

# Regular NumPy-like code
def my_function(x):
    return jnp.sum(x ** 2)

# But with superpowers!
x = jnp.array([1.0, 2.0, 3.0])
result = my_function(x)  # 14.0
```

**Core Philosophy:**
- **Pure functions:** No side effects, deterministic
- **Functional transformations:** Transform functions, not data
- **Composable:** Transformations can be combined freely

### 1.2 Key JAX Transformations

#### 1. `jax.jit` - Just-In-Time Compilation

**What it does:** Compiles Python functions to optimized XLA code

```python
# Uncompiled - slow, retraces every time
def slow_function(x):
    return jnp.dot(x, x.T)

# Compiled - fast, compiled once
@jax.jit
def fast_function(x):
    return jnp.dot(x, x.T)

# First call: compilation + execution (slow)
x = jnp.ones((1000, 1000))
result = fast_function(x)  # ~100ms

# Subsequent calls: cached execution (fast)
result = fast_function(x)  # ~1ms
```

**How JIT works:**

```
Python Function
      ↓
JAX traces execution (abstract evaluation)
      ↓
Builds computation graph
      ↓
XLA compiler optimizes:
  ├─ Operator fusion
  ├─ Memory layout optimization
  ├─ Constant folding
  └─ Loop optimizations
      ↓
Optimized TPU/GPU kernels
      ↓
Cached for reuse
```

**Important JIT rules:**

```python
# ✅ Good - Pure function
@jax.jit
def good_function(x):
    return x * 2 + 1

# ❌ Bad - Side effects
@jax.jit
def bad_function(x):
    print(x)  # Side effect! Won't work with JIT
    return x * 2

# ❌ Bad - Python control flow with dynamic values
@jax.jit
def bad_control_flow(x):
    if x > 0:  # Can't JIT this!
        return x
    else:
        return -x

# ✅ Good - Use jnp.where for conditional
@jax.jit
def good_control_flow(x):
    return jnp.where(x > 0, x, -x)
```

#### 2. `jax.grad` - Automatic Differentiation

**What it does:** Computes gradients automatically

```python
# Define a loss function
def loss_fn(params, x, y):
    predictions = params['w'] * x + params['b']
    return jnp.mean((predictions - y) ** 2)

# Get gradient function
grad_fn = jax.grad(loss_fn)

# Compute gradients
params = {'w': 2.0, 'b': 1.0}
x = jnp.array([1.0, 2.0, 3.0])
y = jnp.array([3.0, 5.0, 7.0])

gradients = grad_fn(params, x, y)
# gradients = {'w': ..., 'b': ...}
```

**Gradient variants:**

```python
# Gradient with respect to first argument (default)
grad_fn = jax.grad(loss_fn)

# Gradient with respect to specific argument
grad_fn = jax.grad(loss_fn, argnums=0)

# Multiple arguments
grad_fn = jax.grad(loss_fn, argnums=(0, 1))

# Gradient + value (common in training)
value_and_grad_fn = jax.value_and_grad(loss_fn)
loss, gradients = value_and_grad_fn(params, x, y)
```

**How autodiff works:**

JAX uses **reverse-mode differentiation** (backpropagation):

```
Forward pass:
  x → f(x) → g(f(x)) → h(g(f(x))) = y

Backward pass:
  ∂y/∂h → ∂h/∂g → ∂g/∂f → ∂f/∂x = ∂y/∂x
```

**Why automatic differentiation is powerful:**
- No manual gradient computation
- Exact derivatives (not numerical approximations)
- Efficient (same complexity as forward pass)
- Works with complex control flow

#### 3. `jax.vmap` - Automatic Vectorization

**What it does:** Vectorizes functions across batch dimension

```python
# Function for single example
def predict_single(params, x):
    return params['w'] * x + params['b']

# Vectorize across batch
predict_batch = jax.vmap(predict_single, in_axes=(None, 0))

params = {'w': 2.0, 'b': 1.0}
x_batch = jnp.array([1.0, 2.0, 3.0])

# Automatically processes all examples
predictions = predict_batch(params, x_batch)
# [3.0, 5.0, 7.0]
```

**in_axes specification:**

```python
# in_axes=(None, 0) means:
# - First arg (params): don't vectorize
# - Second arg (x): vectorize over axis 0

# More complex example:
def complex_fn(a, b, c):
    return a * b + c

# Vectorize b and c, but not a
vectorized = jax.vmap(complex_fn, in_axes=(None, 0, 0))
```

**Why vmap matters:**
- Automatic batch processing
- No manual loop writing
- Compiler can optimize vectorized code
- Essential for mini-batch training

#### 4. `jax.pmap` - Parallel Map Across Devices

**What it does:** Distributes computation across multiple devices

```python
# Function to run on each device
def train_step_single_device(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    return loss, grads

# Parallelize across devices
train_step_parallel = jax.pmap(train_step_single_device)

# Data for 8 devices (8-way data parallelism)
params = replicate_to_devices(params, n_devices=8)
batches = shard_to_devices(batch, n_devices=8)

# Run on all devices in parallel
losses, grads = train_step_parallel(params, batches)
```

**Communication patterns with pmap:**

```python
# All-reduce example
@jax.pmap
def all_reduce_mean(x):
    return jax.lax.pmean(x, axis_name='devices')

# All-gather example  
@jax.pmap
def all_gather(x):
    return jax.lax.all_gather(x, axis_name='devices')
```

**pmap vs vmap:**

| Feature | vmap | pmap |
|---------|------|------|
| **Vectorizes over** | Batch dimension | Devices |
| **Memory** | Single device | Distributed |
| **Communication** | None | Cross-device |
| **Use case** | Batch processing | Multi-GPU/TPU |

### 1.3 Composing Transformations

**The magic of JAX: transformations compose!**

```python
# Combine jit + grad
@jax.jit
def fast_gradient(params, x, y):
    return jax.grad(loss_fn)(params, x, y)

# Combine jit + grad + vmap
@jax.jit
@jax.vmap
def fast_batched_gradient(params, x_batch, y_batch):
    return jax.grad(loss_fn)(params, x_batch, y_batch)

# Combine jit + grad + pmap (distributed training!)
@jax.jit
@jax.pmap
def distributed_training_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    # Average gradients across devices
    grads = jax.lax.pmean(grads, axis_name='devices')
    return loss, grads
```

**Order matters sometimes:**

```python
# ✅ Good: JIT outermost (compile everything)
@jax.jit
@jax.vmap
def good_order(x):
    return x * 2

# ❌ Less efficient: JIT innermost
@jax.vmap
@jax.jit
def less_efficient(x):
    return x * 2  # JIT compiles N times
```

### 1.4 JAX and Random Numbers

**JAX uses explicit PRNG keys (functional randomness):**

```python
import jax.random as random

# Create a key
key = random.PRNGKey(42)

# Split key for different random operations
key, subkey = random.split(key)

# Generate random numbers
random_array = random.normal(subkey, shape=(10,))

# For multiple operations, split multiple times
key, *subkeys = random.split(key, num=5)
dropout1 = random.bernoulli(subkeys[0], p=0.5, shape=(100,))
dropout2 = random.bernoulli(subkeys[1], p=0.5, shape=(100,))
```

**Why explicit keys?**
- Reproducibility: Same key → same random numbers
- Parallelization: Can split keys for parallel streams
- No global state: Pure functional approach

**Key splitting pattern:**

```
Original Key
     │
     ├─ Split
     │
     ├─────┬─────┐
     │     │     │
   Key1  Key2  Key3
     │     │     │
  Random Random Random
  Stream1 Stream2 Stream3
```

### 1.5 JAX Memory Model

**Device Arrays:**

```python
# Create array on default device
x = jnp.array([1, 2, 3])
print(x.device())  # TPU:0 or GPU:0

# Explicitly place on device
from jax import device_put
x_on_tpu = device_put(x, device=jax.devices('tpu')[0])
```

**Sharding:**

```python
from jax.sharding import PartitionSpec as P

# Define how to shard array
sharding = PositionalSharding(jax.devices()).reshape(4, 2)

# Shard array across devices
sharded_array = jax.device_put(array, sharding)
```

---

## 2. Flax NNX: Neural Networks in JAX

### 2.1 Core Concepts

**Module-based design:**

```python
from flax import nnx

class SimpleModel(nnx.Module):
    def __init__(self, features: int, rngs: nnx.Rngs):
        # Parameters automatically tracked
        self.linear1 = nnx.Linear(features, features, rngs=rngs)
        self.linear2 = nnx.Linear(features, 1, rngs=rngs)
        self.dropout = nnx.Dropout(rate=0.5, rngs=rngs)
    
    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

### 2.2 Variable Types

**NNX tracks different state types:**

```python
class ComplexModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        # Params: Trainable parameters
        self.linear = nnx.Linear(128, 128, rngs=rngs)
        
        # BatchStats: Moving averages
        self.batch_norm = nnx.BatchNorm(
            num_features=128,
            rngs=rngs,
            use_running_average=False
        )
        
        # Dropout: RNG state
        self.dropout = nnx.Dropout(0.1, rngs=rngs)
```

**Accessing state:**

```python
model = ComplexModel(rngs=nnx.Rngs(0))

# Get only trainable parameters
params = nnx.state(model, nnx.Param)

# Get all state
full_state = nnx.state(model)

# Get specific state types
batch_stats = nnx.state(model, nnx.BatchStat)
```

### 2.3 Training with NNX

**Complete training example:**

```python
from flax import nnx
import optax

# 1. Create model
model = SimpleModel(features=128, rngs=nnx.Rngs(0))

# 2. Create optimizer
learning_rate = 1e-3
tx = optax.adam(learning_rate)
optimizer = nnx.Optimizer(model, tx)

# 3. Define loss function
def loss_fn(model, batch):
    predictions = model(batch['x'])
    loss = jnp.mean((predictions - batch['y']) ** 2)
    return loss

# 4. Training step (with JIT)
@nnx.jit
def train_step(model, optimizer, batch):
    # Compute loss and gradients
    loss, grads = nnx.value_and_grad(loss_fn)(model, batch)
    
    # Update parameters
    optimizer.update(grads)
    
    return loss

# 5. Training loop
for epoch in range(num_epochs):
    for batch in data_loader:
        loss = train_step(model, optimizer, batch)
        print(f"Loss: {loss}")
```

### 2.4 Model Composition

**Building complex architectures:**

```python
class TransformerBlock(nnx.Module):
    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.attention = nnx.MultiHeadAttention(
            num_heads=8,
            qkv_features=dim,
            rngs=rngs
        )
        self.mlp = nnx.Sequential([
            nnx.Linear(dim, 4 * dim, rngs=rngs),
            nnx.relu,
            nnx.Linear(4 * dim, dim, rngs=rngs),
        ])
        self.ln1 = nnx.LayerNorm(num_features=dim, rngs=rngs)
        self.ln2 = nnx.LayerNorm(num_features=dim, rngs=rngs)
    
    def __call__(self, x):
        # Self-attention block
        h = self.ln1(x)
        h = self.attention(h)
        x = x + h
        
        # MLP block
        h = self.ln2(x)
        h = self.mlp(h)
        x = x + h
        
        return x

class Transformer(nnx.Module):
    def __init__(self, num_layers: int, dim: int, rngs: nnx.Rngs):
        # Stack transformer blocks
        self.blocks = [
            TransformerBlock(dim, rngs=rngs)
            for _ in range(num_layers)
        ]
    
    def __call__(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

### 2.5 Stateful Training (Dropout, BatchNorm)

**Handling mutable state:**

```python
class ModelWithBatchNorm(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.bn = nnx.BatchNorm(
            num_features=128,
            rngs=rngs,
            use_running_average=False  # Training mode
        )
    
    def __call__(self, x, *, train: bool = True):
        # Batch norm behavior changes based on mode
        self.bn.use_running_average = not train
        return self.bn(x)

# Usage
model = ModelWithBatchNorm(rngs=nnx.Rngs(0))

# Training
output = model(x, train=True)  # Updates running statistics

# Inference
output = model(x, train=False)  # Uses running statistics
```

### 2.6 NNX vs Linen

**Old way (Linen):**

```python
# Linen - functional, explicit state passing
class LinenModel(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=128)(x)
        return x

model = LinenModel()
variables = model.init(key, x)
output = model.apply(variables, x)  # Must pass state explicitly
```

**New way (NNX):**

```python
# NNX - stateful, Pythonic
class NNXModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.dense = nnx.Linear(128, rngs=rngs)
    
    def __call__(self, x):
        return self.dense(x)

model = NNXModel(rngs=nnx.Rngs(0))
output = model(x)  # State implicit, easier!
```

**Why Tunix uses NNX:**
- More Pythonic and intuitive
- Less boilerplate
- Easier debugging (can inspect model.params directly)
- Better for interactive development
- Cleaner integration with training loops

---

## 3. TPU Architecture Deep Dive

### 3.1 TPU Hardware Overview

**TPU Chip Components:**

```
┌─────────────────────────────────┐
│         TPU v4 Chip             │
├─────────────────────────────────┤
│                                 │
│  ┌─────────────────────────┐   │
│  │   Matrix Unit (MXU)     │   │
│  │   128x128 systolic array│   │
│  │   bf16/int8 operations  │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │  High Bandwidth Memory  │   │
│  │      (HBM)              │   │
│  │      32 GB              │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │  Vector Processing Unit │   │
│  │      (VPU)              │   │
│  └─────────────────────────┘   │
│                                 │
│  ┌─────────────────────────┐   │
│  │  Interconnect Interface │   │
│  │      (ICI)              │   │
│  └─────────────────────────┘   │
│                                 │
└─────────────────────────────────┘
```

**Key Specs (TPU v4):**
- **Matrix Unit:** 275 TFLOPs (bf16)
- **Memory:** 32 GB HBM per chip
- **Bandwidth:** 1.2 TB/s memory bandwidth
- **Interconnect:** 3D torus topology

### 3.2 Matrix Multiplication Unit (MXU)

**Why MXU is powerful:**

```
CPU/GPU: Sequential operations
  a[0]*b[0] → a[1]*b[1] → a[2]*b[2] → ...

TPU Systolic Array: Parallel pipeline
  128x128 = 16,384 operations simultaneously!
  
  Data flows through array:
  →→→→→→
  ↓↓↓↓↓↓  Each cell: multiply-accumulate
  →→→→→→
  ↓↓↓↓↓↓
```

**Perfect for transformers:**
- Attention: QK^T matmul
- Attention output: Softmax(QK^T)V matmul
- MLP: Two large matmuls
- ~90% of compute is matmul → TPU optimized!

### 3.3 Memory Hierarchy

```
┌─────────────────────────────────────┐
│  L1 Cache (on-chip SRAM)            │ Fastest
│  ~10 MB, ~0.5ns latency             │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  HBM (High Bandwidth Memory)        │ Fast
│  32 GB, ~100ns latency              │
│  1.2 TB/s bandwidth                 │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│  Host Memory (CPU RAM)              │ Slow
│  100s of GB, ~1μs latency           │
└─────────────────────────────────────┘ Slowest
```

**Memory optimization strategies:**
1. Keep activations in HBM
2. Recompute instead of store (gradient checkpointing)
3. Stream large weights from host if needed
4. Use reduced precision (bf16 vs fp32)

### 3.4 TPU Pod Architecture

**Single Host:** 4 TPU chips

```
Host Machine
  ├─ TPU Chip 0
  ├─ TPU Chip 1
  ├─ TPU Chip 2
  └─ TPU Chip 3
```

**TPU Pod:** Multiple hosts networked together

```
v4-32 Pod (4 hosts × 4 chips each = 16 chips):

Host 0     Host 1     Host 2     Host 3
[0 1]      [4 5]      [8  9]     [12 13]
[2 3]      [6 7]      [10 11]    [14 15]
  │          │          │            │
  └──────────┴──────────┴────────────┘
        ICI Network (2D/3D torus)
```

**ICI (Inter-Chip Interconnect):**
- Ultra-low latency (~5μs chip-to-chip)
- High bandwidth (100+ GB/s per link)
- Enables efficient data parallelism and model parallelism

### 3.5 TPU vs GPU Comparison

| Feature | TPU v4 | A100 GPU |
|---------|--------|----------|
| **Matrix Compute** | 275 TFLOPs (bf16) | 312 TFLOPs (bf16) |
| **Memory** | 32 GB HBM | 40/80 GB HBM |
| **Memory BW** | 1.2 TB/s | 1.5-2 TB/s |
| **Interconnect** | ICI (dedicated) | NVLink |
| **Precision** | bf16 native | bf16/fp16 |
| **Architecture** | Systolic array | CUDA cores |
| **Best for** | Large matmuls, LLMs | General compute, varied workloads |

**When TPU wins:**
- Transformer training (matmul-heavy)
- Large batch sizes
- Multi-host scaling
- Cost efficiency at scale

**When GPU wins:**
- Custom kernels (CUDA)
- Small models (latency)
- Diverse operations
- Existing CUDA ecosystem

---

## 4. Pathways: Distributed Training System

### 4.1 What is Pathways?

**Pathways = Google's infrastructure for distributed ML**

**Key Ideas:**
1. **Single Program, Multiple Data (SPMD):** Same program runs on all devices
2. **Virtual Machine:** Treats entire cluster as one machine
3. **Automatic Sharding:** Framework decides data/model placement
4. **Fault Tolerance:** Handles device failures gracefully

### 4.2 Pathways Architecture

```
┌──────────────────────────────────────────────────┐
│            Pathways Runtime                      │
├──────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐         ┌──────────────┐     │
│  │  Scheduler   │◄───────►│  Resource    │     │
│  │              │         │  Manager     │     │
│  └──────┬───────┘         └──────────────┘     │
│         │                                        │
│  ┌──────▼─────────────────────────────────┐    │
│  │         Task Coordinator               │    │
│  └──────┬─────────────────────────────────┘    │
│         │                                        │
├─────────┼────────────────────────────────────────┤
│         │    Distributed Execution              │
├─────────┼────────────────────────────────────────┤
│         │                                        │
│  ┌──────▼─────┐  ┌───────────┐  ┌───────────┐ │
│  │ TPU Pod 0  │  │ TPU Pod 1 │  │ TPU Pod N │ │
│  │ [Devices]  │  │ [Devices] │  │ [Devices] │ │
│  └────────────┘  └───────────┘  └───────────┘ │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 4.3 SPMD Programming Model

**Single Program:**

```python
# This same code runs on ALL devices
@jax.jit
def training_step(params, batch):
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    
    # Automatic cross-device communication
    grads = jax.lax.pmean(grads, axis_name='replicas')
    
    params = update_params(params, grads)
    return params, loss
```

**Multiple Data:**

```
Device 0: processes batch[0:32]
Device 1: processes batch[32:64]
Device 2: processes batch[64:96]
Device 3: processes batch[96:128]

All run same code on different data!
```

### 4.4 Automatic Sharding

**User specifies sharding strategy:**

```python
from jax.sharding import Mesh, PartitionSpec as P

# Create logical mesh
mesh = Mesh(devices, axis_names=('data', 'model'))

# Specify how to shard arrays
# (None, 'data') means: don't shard first dim, shard second dim across 'data' axis
data_sharding = NamedSharding(mesh, P(None, 'data'))

# Shard data
sharded_batch = jax.device_put(batch, data_sharding)
```

**Pathways handles:**
- Data transfer between devices
- Communication collectives (all-reduce, all-gather, etc.)
- Load balancing
- Memory management

### 4.5 Multi-Host Training

**Challenges:**
- Synchronize gradients across hosts
- Coordinate checkpointing
- Handle stragglers (slow hosts)

**Pathways solutions:**

```python
# 1. Initialize multi-host JAX
jax.distributed.initialize()

# 2. Create global mesh spanning all hosts
devices = jax.devices()  # All devices across all hosts
mesh = Mesh(devices, ('data', 'model'))

# 3. Code looks the same as single-host!
@jax.jit
def train_step(params, batch):
    # Gradients automatically synchronized across hosts
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    grads = jax.lax.pmean(grads, axis_name='replicas')
    return update(params, grads)
```

**Host coordination:**

```
Host 0              Host 1              Host 2
  ↓                   ↓                   ↓
Load data         Load data           Load data
  ↓                   ↓                   ↓
Train step        Train step          Train step
  ↓                   ↓                   ↓
  └───────────────────┴───────────────────┘
              All-reduce gradients
  ┌───────────────────┬───────────────────┐
  ↓                   ↓                   ↓
Update             Update              Update
  ↓                   ↓                   ↓
Checkpoint if       (coordinated)
process_index == 0
```

### 4.6 Fault Tolerance

**Pathways features:**
- Automatic device failure detection
- Checkpoint-restart mechanisms
- Straggler mitigation
- Redundant computation for critical operations

**Recovery flow:**

```
Normal Operation
      ↓
Device Failure Detected
      ↓
Pause Computation
      ↓
Restore from Last Checkpoint
      ↓
Redistribute Work
      ↓
Resume Training
```

---

## 5. XLA (Accelerated Linear Algebra)

### 5.1 What is XLA?

**XLA = Domain-specific compiler for linear algebra**

**Compilation Pipeline:**

```
Python/JAX Code
      ↓
JAX traces → HLO (High-Level Operations)
      ↓
XLA Optimizations:
  ├─ Operator fusion
  ├─ Buffer assignment
  ├─ Layout optimization
  └─ Algebraic simplification
      ↓
TPU/GPU Machine Code
      ↓
Fast Execution!
```

### 5.2 Key XLA Optimizations

#### 1. Operator Fusion

**Without fusion:**
```python
# Three separate kernels
x = jnp.exp(a)       # Kernel 1: load a, compute exp, store x
y = jnp.log(b)       # Kernel 2: load b, compute log, store y
z = x + y            # Kernel 3: load x, y, add, store z

# Memory traffic: 5 loads + 3 stores
```

**With fusion:**
```python
# Single fused kernel
z = jnp.exp(a) + jnp.log(b)

# Memory traffic: 2 loads + 1 store
# Much faster!
```

#### 2. Layout Optimization

**XLA chooses optimal memory layouts:**
```
Array: [batch, seq_len, hidden_dim]

Row-major: [b0s0h0 b0s0h1 ... b0s1h0 b0s1h1 ...]
Column-major: [b0s0h0 b1s0h0 ... b0s0h1 b1s0h1 ...]

XLA picks best layout for access patterns!
```

#### 3. Constant Folding

```python
# XLA detects constants at compile time
def model(x):
    scale = 2.0 * 3.14159  # Computed at compile time!
    return x * scale

# Optimized to:
def model(x):
    return x * 6.28318
```

### 5.3 HLO (High-Level Operations)

**XLA's intermediate representation:**

```
Python: z = x * 2 + y

HLO:
  %mul = multiply(%x, constant(2))
  %add = add(%mul, %y)
  
XLA optimizes HLO graph before generating code
```

---

## 6. Putting It All Together: Training Pipeline

### 6.1 Complete Stack

```
┌─────────────────────────────────────────┐
│         User Training Script            │  Python
├─────────────────────────────────────────┤
│              Tunix API                  │  Python
├─────────────────────────────────────────┤
│            Flax NNX                     │  Python
├─────────────────────────────────────────┤
│              JAX                        │  Python + C++
├─────────────────────────────────────────┤
│         XLA Compiler                    │  C++
├─────────────────────────────────────────┤
│          Pathways Runtime               │  C++
├─────────────────────────────────────────┤
│       TPU Driver/Firmware               │  Low-level
├─────────────────────────────────────────┤
│          TPU Hardware                   │  Silicon
└─────────────────────────────────────────┘
```

### 6.2 Training Step Execution Flow

```
1. Python: trainer.train_step(batch)
       ↓
2. Flax NNX: Model forward pass
       ↓
3. JAX: Trace operations, build computation graph
       ↓
4. JIT: Check cache for compiled version
       ├─ Hit: Use cached executable
       └─ Miss: Continue to compile
           ↓
5. XLA: Optimize HLO graph
       ├─ Fuse operations
       ├─ Optimize layouts
       └─ Allocate buffers
       ↓
6. XLA: Generate TPU machine code
       ↓
7. Pathways: Distribute to TPU devices
       ↓
8. TPU: Execute matrix operations
       ├─ MXU performs matmuls
       ├─ VPU performs element-wise ops
       └─ ICI communicates between chips
       ↓
9. Results: Gradients returned to Python
       ↓
10. Optimizer: Update parameters
       ↓
11. Metrics: Log progress
```

### 6.3 Memory and Compute Optimization

**The optimization stack:**

```
Level 1: Python/Tunix
  └─ Gradient accumulation, micro-batching

Level 2: Flax NNX
  └─ Efficient state management

Level 3: JAX
  └─ JIT compilation, automatic differentiation
  
Level 4: XLA
  └─ Operator fusion, layout optimization
  
Level 5: Pathways
  └─ Data sharding, communication optimization
  
Level 6: TPU
  └─ Hardware-level parallelism
```

---

## 7. Practical Implications for Tunix Users

### 7.1 Writing JAX-Friendly Code

**Do's:**
```python
# ✅ Pure functions
def loss_fn(params, batch):
    return compute_loss(params, batch)

# ✅ jnp instead of np
import jax.numpy as jnp
x = jnp.array([1, 2, 3])

# ✅ Explicit shapes
def forward(x: jax.Array) -> jax.Array:  # Clear types
    assert x.shape == (batch_size, seq_len, hidden_dim)
    return output
```

**Don'ts:**
```python
# ❌ Side effects
global_state = []
def bad_fn(x):
    global_state.append(x)  # Don't do this!
    return x

# ❌ Python control flow with JAX arrays
def bad_control(x):
    if x.sum() > 0:  # Traced as abstract value!
        return x
    else:
        return -x

# ❌ In-place mutations
def bad_update(x):
    x[0] = 10  # JAX arrays are immutable!
    return x
```

### 7.2 Debugging Tips

**1. Disable JIT for debugging:**
```python
with jax.disable_jit():
    result = train_step(params, batch)
    # Now can use print(), pdb, etc.
```

**2. Check shapes:**
```python
jax.debug.print("Shape: {}", x.shape)
```

**3. Visualize computation:**
```python
# Get HLO representation
hlo = jax.xla_computation(train_step)(params, batch).as_hlo_text()
print(hlo)
```

### 7.3 Performance Tips

**1. Batch size matters:**
```python
# Too small: Underutilizes TPU
batch_size = 4  # ❌ Slow

# Sweet spot: Fills TPU capacity
batch_size = 128  # ✅ Fast

# Too large: OOM
batch_size = 10000  # ❌ Out of memory
```

**2. Use gradient accumulation for large batches:**
```python
# Effective batch = micro_batch * accumulation_steps
TrainingConfig(
    mini_batch_size=128,
    train_micro_batch_size=32,  # Process in 4 steps
)
```

**3. Profile to find bottlenecks:**
```python
profiler_options = ProfilerOptions(
    enable=True,
    profile_dir="/tmp/profile"
)
# Analyze with TensorBoard or Cloud Profiler
```

---

## 🎯 Phase 1.3 Checklist

- [ ] Understand JAX core transformations (jit, grad, vmap, pmap)
- [ ] Know how transformations compose
- [ ] Grasp Flax NNX module system and variable types
- [ ] Understand TPU architecture (MXU, HBM, ICI)
- [ ] Know when TPUs are better than GPUs
- [ ] Understand Pathways distributed training
- [ ] Know XLA compilation pipeline and optimizations
- [ ] Understand complete training stack
- [ ] Can write JAX-friendly code
- [ ] Ready to dive into specific modules (Phase 2)

---

## 📚 Additional Resources

### JAX Resources
- [JAX Documentation](https://jax.readthedocs.io/)
- [JAX Quickstart](https://jax.readthedocs.io/en/latest/quickstart.html)
- [The Autodiff Cookbook](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)

### Flax Resources
- [Flax Documentation](https://flax.readthedocs.io/)
- [NNX Basics](https://flax.readthedocs.io/en/latest/nnx_basics.html)

### TPU Resources
- [Cloud TPU Documentation](https://cloud.google.com/tpu/docs)
- [TPU Best Practices](https://cloud.google.com/tpu/docs/performance-guide)

### XLA Resources
- [XLA Overview](https://www.tensorflow.org/xla)
- [XLA Compilation](https://www.tensorflow.org/xla/architecture)

---

**Previous:** [Phase 1.2 - Architecture Overview](Phase_1_2_Architecture_Overview.md)  
**Next:** Phase 2 - Core Components Deep Dive (Coming soon!)

---

**🎉 Congratulations! You've completed Phase 1!**

You now have a solid foundation of:
- LLM post-training concepts
- Tunix architecture
- JAX, Flax NNX, and TPU fundamentals

Ready to dive deep into specific components in Phase 2!
