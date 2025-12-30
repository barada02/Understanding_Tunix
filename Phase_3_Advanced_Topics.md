---
markmap:
  initialExpandLevel: 2
---

# Phase 3: Advanced Topics

## Overview

Phase 3 covers **advanced topics** for production-grade training in Tunix, focusing on distributed training infrastructure, efficient data pipelines, and system optimization. These components enable large-scale model training across multiple hosts, efficient resource utilization, and robust checkpoint management.

**Purpose**: Understand the infrastructure and optimization techniques required for production training at scale.

**Location**: Primarily in `tunix/sft/`, `tunix/rl/`, `tunix/utils/`, and `tunix/cli/utils/`

**Key Topics**:
1. **Training Infrastructure (3.1)**: Sharding strategies, checkpoint management, metrics logging, system monitoring
2. **Distributed Training (3.2)**: Multi-host coordination, resharding operations, Pathways integration
3. **Data Pipeline (3.3)**: Dataset loading with Grain, tokenization, batch construction

**Design Philosophy**:
- **Scalability**: Support for multi-host, multi-device training
- **Efficiency**: Minimize communication overhead, optimize memory usage
- **Flexibility**: Support various sharding strategies and data sources
- **Observability**: Comprehensive metrics and logging infrastructure

**Supported Training Scales**:
| Scale | Devices | Strategy | Use Case |
|-------|---------|----------|----------|
| Single-host | 1-8 TPUs/GPUs | Data parallel | Development, small models |
| Multi-host | 8-256 TPUs | FSDP, tensor parallel | Medium models (7B-70B) |
| Large-scale | 256+ TPUs | Pipeline + tensor parallel | Very large models (70B+) |

## 3.1 Training Infrastructure

### Sharding Strategies

**File**: `tunix/sft/sharding_utils.py`

Tunix uses JAX's sharding infrastructure to distribute model parameters and data across multiple devices.

#### Core Concepts

**1. JAX Sharding Primitives**:
```python
import jax.sharding as shd

# PartitionSpec defines how to shard each dimension
pspec = shd.PartitionSpec("fsdp", None)  # Shard dim 0 along "fsdp" axis

# NamedSharding combines mesh and spec
mesh = jax.sharding.Mesh(devices, ("fsdp", "tp"))
sharding = shd.NamedSharding(mesh, pspec)
```

**2. Mesh Definition**:
A mesh organizes devices into logical axes:
```python
# Example: 8 devices in FSDP layout
devices = jax.devices()
mesh = jax.sharding.Mesh(devices, ("fsdp",))

# Example: 8 devices in 2D layout (FSDP x Tensor Parallel)
mesh = jax.sharding.Mesh(
    devices.reshape(4, 2), 
    ("fsdp", "tp")
)
```

**3. Sharding Strategies**:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **FSDP** | Fully Sharded Data Parallel - shard model across all devices | Large models, memory constrained |
| **Data Parallel** | Replicate model, shard data | Small models, compute bound |
| **Tensor Parallel** | Shard individual layers along hidden dimension | Very large models (70B+) |
| **Pipeline Parallel** | Split model into stages across devices | Sequential processing |

#### Implementation

**`shard_input()` Function**:
```python
def shard_input(
    input_data: jax.Array, 
    data_sharding_axis: Tuple[str, ...]
) -> jax.Array:
  """Shards input data across available devices.
  
  Args:
    input_data: The input data to be sharded
    data_sharding_axis: Sharding axis, e.g. ("fsdp",)
  
  Returns:
    Sharded input
  """
  mesh = pxla.thread_resources.env.physical_mesh
  if mesh.empty:
    return input_data
  
  pspec = shd.PartitionSpec(*data_sharding_axis)
  
  # Check if already sharded to avoid re-sharding
  is_sharded = jax.tree.map(
      lambda x: isinstance(x, jax.Array)
      and hasattr(x, "sharding")
      and x.sharding.mesh == mesh
      and x.sharding.spec == pspec,
      input_data,
  )
  if all(jax.tree.leaves(is_sharded)):
    return input_data
  
  # Shard the data
  with jax.transfer_guard("allow"):
    return jax.tree.map(
        lambda x: jax.make_array_from_process_local_data(
            get_sharding(x, mesh=mesh, pspec=pspec), x
        ),
        input_data,
    )
```

**`get_sharding()` Function**:
```python
def get_sharding(x: jax.Array, mesh: shd.Mesh, pspec: shd.PartitionSpec):
  """Get a sharding for a tensor given mesh and partition spec."""
  # Only shard arrays with rank > 0
  if not isinstance(x, (np.ndarray, jax.Array)) or x.ndim == 0:
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated
  
  # Don't shard if rank is not sufficient
  if x.ndim < len(pspec):
    return shd.NamedSharding(mesh, shd.PartitionSpec())  # Replicated
  
  # Check divisibility for all sharded axes
  for i, axis_name in enumerate(pspec):
    if axis_name is not None:
      axis_names = axis_name if isinstance(axis_name, tuple) else (axis_name,)
      for name in axis_names:
        axis_size = mesh.shape[name]
        if x.shape[i] % axis_size != 0:
          # Replicate if not evenly divisible
          return shd.NamedSharding(mesh, shd.PartitionSpec())
  
  return shd.NamedSharding(mesh, pspec)
```

**Usage in PeftTrainer**:
```python
# In PeftTrainer configuration
@dataclasses.dataclass
class PeftTrainerConfig:
  data_sharding_axis: Tuple[str, ...] = ("fsdp",)
  # ... other config

# During training
train_example = sharding_utils.shard_input(
    train_example, 
    self.config.data_sharding_axis
)
```

#### Sharding Constraints

**Purpose**: Ensure computations happen on the correct devices.

```python
# Apply sharding constraint during computation
optimizer_sharded_state = jax.lax.with_sharding_constraint(
    optimizer_state, 
    optimizer_sharding
)
```

**When to Use**:
- After collective operations (all-gather, all-reduce)
- Before expensive computations
- To prevent unwanted data movement

### Checkpoint Management

**File**: `tunix/sft/checkpoint_manager.py`

Checkpoint management handles saving and restoring model state during training.

#### Architecture

**CheckpointManager Class**:
```python
class CheckpointManager:
  """Checkpoint manager for PEFT."""
  
  def __init__(
      self,
      root_directory: str | None = None,
      options: ocp.CheckpointManagerOptions | None = None,
  ):
    """Initialize checkpoint manager.
    
    Args:
      root_directory: Root directory for checkpoints (None to disable)
      options: Checkpoint manager options
    """
    self._checkpoint_manager: ocp.CheckpointManager | None = None
    if root_directory is not None:
      # Special handling for Pathways
      if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(
                use_ocdbt=False,
                use_zarr3=False,
            ),
        }
        logging.info('Using persistence APIs for checkpointing with Pathways.')
      else:
        item_handlers = {
            'model_params': ocp.PyTreeCheckpointHandler(),
        }
      item_handlers['custom_metadata'] = ocp.JsonCheckpointHandler()
      self._checkpoint_manager = ocp.CheckpointManager(
          root_directory,
          item_handlers=item_handlers,
          options=options or _DEFAULT_CHECKPOINTING_OPTIONS,
      )
```

#### Key Features

**1. Default Options**:
```python
_DEFAULT_CHECKPOINTING_OPTIONS = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,  # Save at most every 3 minutes
    ),
    max_to_keep=3,  # Keep only 3 most recent checkpoints
)
```

**2. Save Checkpoints**:
```python
def save(
    self,
    step: int,
    model: nnx.Module,
    save_only_lora_params: bool = False,
    force: bool = False,
    custom_metadata: dict[str, Any] | None = None,
) -> bool:
  """Save model parameters at given step.
  
  Args:
    step: Training step number
    model: Model to save
    save_only_lora_params: Whether to save only LoRA parameters
    force: Whether to force save (ignore save decision policy)
    custom_metadata: Additional metadata to save
  
  Returns:
    True if checkpoint was saved
  """
  if self._checkpoint_manager is None:
    return False
  if not force and not self._checkpoint_manager.should_save(step):
    return False
  
  # Extract parameters
  if save_only_lora_params:
    params = nnx.state(model, nnx.LoRAParam)
  else:
    params = nnx.state(model)
  
  # Create checkpoint args
  checkpoint_args = ocp.args.PyTreeSave(
      item=params, 
      save_args=jax.tree.map(lambda _: ocp.SaveArgs(), params)
  )
  
  # Save checkpoint
  return self._checkpoint_manager.save(
      step,
      args=ocp.args.Composite(model_params=checkpoint_args),
      custom_metadata=custom_metadata or {},
      force=force,
  )
```

**3. Restore Checkpoints**:
```python
def maybe_restore(
    self,
    model: nnx.Module,
    step: int | None = None,
    restore_only_lora_params: bool = False,
) -> Tuple[int, dict[str, Any]]:
  """Restore parameters from latest checkpoint if available.
  
  Args:
    model: Model to restore parameters into
    step: Specific step to restore (None for latest)
    restore_only_lora_params: Whether to restore only LoRA parameters
  
  Returns:
    Tuple of (restored_step, custom_metadata), (0, {}) if no checkpoint
  """
  if self._checkpoint_manager is None:
    return 0, {}
  
  if step is None:
    step = self._checkpoint_manager.latest_step()
    if step is None:
      return 0, {}
  
  # Get abstract parameters structure
  if restore_only_lora_params:
    abstract_params = nnx.state(model, nnx.LoRAParam)
  else:
    abstract_params = nnx.state(model)
  
  # Restore from checkpoint
  restored = self._checkpoint_manager.restore(
      step,
      args=ocp.args.Composite(
          model_params=ocp.args.PyTreeRestore(
              item=abstract_params,
              restore_args=jax.tree.map(
                  lambda data: ocp.type_handlers.ArrayRestoreArgs(
                      sharding=data.sharding
                  ),
                  abstract_params,
              ),
          )
      ),
  )
  
  # Update model with restored parameters
  nnx.update(model, restored['model_params'].item)
  
  return step, restored.get('custom_metadata', {})
```

#### Checkpoint Formats

**Orbax Checkpoint Structure**:
```
checkpoint_dir/
├── 0/                           # Step 0
│   ├── model_params/
│   │   └── default             # PyTree checkpoint
│   └── custom_metadata.json    # Training metadata
├── 100/                        # Step 100
│   ├── model_params/
│   └── custom_metadata.json
└── _CHECKPOINT_METADATA        # Checkpoint manager metadata
```

**Custom Metadata Examples**:
```python
custom_metadata = {
    "train_loss": 2.5,
    "eval_accuracy": 0.85,
    "learning_rate": 1e-4,
    "epoch": 3,
}
```

### Metrics and Logging

**File**: `tunix/sft/metrics_logger.py`

Metrics logging uses the **Metrax** protocol for unified backend support.

#### Architecture

**MetricsLogger Class**:
```python
class MetricsLogger:
  """Simple Metrics logger.
  
  Logs metrics to multiple backends (TensorBoard, WandB, CLU).
  """
  
  def __init__(
      self,
      metrics_logger_options: MetricsLoggerOptions | None = None,
  ):
    self._metrics = {}
    self._backends = (
        metrics_logger_options.create_backends()
        if metrics_logger_options
        else []
    )
    if metrics_logger_options and jax.process_index() == 0:
      for backend in self._backends:
        jax.monitoring.register_scalar_listener(backend.log_scalar)
```

#### Supported Backends

**1. TensorBoard** (OSS default):
```python
TensorboardBackend(
    log_dir=log_dir,
    flush_every_n_steps=100,
)
```

**2. WandB** (OSS optional):
```python
WandbBackend(
    project=project_name, 
    name=run_name
)
```

**3. CLU** (Google internal):
```python
CluBackend(log_dir=log_dir)
```

#### Configuration

**MetricsLoggerOptions**:
```python
@dataclasses.dataclass
class MetricsLoggerOptions:
  """Metrics Logger options."""
  
  log_dir: str                              # Directory for logs
  project_name: str = "tunix"              # Project name (for WandB)
  run_name: str = ""                       # Run name (for WandB)
  flush_every_n_steps: int = 100           # Flush frequency
  backend_factories: list[BackendFactory] | None = None  # Custom backends
  
  def create_backends(self) -> list[LoggingBackend]:
    """Factory method to create backends."""
    if jax.process_index() != 0:
      return []  # Only log on main process
    
    if self.backend_factories is not None:
      return [factory() for factory in self.backend_factories]
    
    # Default backends
    active_backends = []
    
    if env_utils.is_internal_env():
      active_backends.append(CluBackend(log_dir=self.log_dir))
    else:
      active_backends.append(
          TensorboardBackend(
              log_dir=self.log_dir,
              flush_every_n_steps=self.flush_every_n_steps,
          )
      )
      try:
        active_backends.append(
            WandbBackend(project=self.project_name, name=self.run_name)
        )
      except ImportError:
        logging.info("WandbBackend skipped: 'wandb' library not installed.")
    
    return active_backends
```

#### Logging Metrics

**1. Log Scalar Metrics**:
```python
def log(
    self,
    metrics_prefix: str,
    metric_name: str,
    scalar_value: float | np.ndarray,
    mode: Mode | str,
    step: int,
):
  """Log scalar metric value.
  
  Args:
    metrics_prefix: Prefix for metric (e.g., "training")
    metric_name: Name of metric (e.g., "loss")
    scalar_value: Metric value
    mode: Mode (TRAIN or EVAL)
    step: Training step
  """
  # Store in local history
  prefix_metrics = self._metrics.setdefault(metrics_prefix, {})
  mode_metrics = prefix_metrics.setdefault(
      mode, collections.defaultdict(list)
  )
  mode_metrics[metric_name].append(scalar_value)
  
  # Log via jax.monitoring (broadcasts to all backends)
  jax.monitoring.record_scalar(
      f"{metrics_prefix}/{mode}/{metric_name}", 
      scalar_value, 
      step=step
  )
```

**2. Retrieve Metrics**:
```python
def get_metric(self, metrics_prefix, metric_name: str, mode: Mode | str):
  """Returns mean metric value."""
  if not self.metric_exists(metrics_prefix, metric_name, mode):
    raise ValueError(f"Metric '{metrics_prefix}/{mode}/{metric_name}' not found.")
  
  values = np.stack(self._metrics[metrics_prefix][mode][metric_name])
  
  # Special handling for perplexity (geometric mean)
  if metric_name == "perplexity":
    return _calculate_geometric_mean(values)
  
  return np.mean(values)
```

**Usage Example**:
```python
# Initialize logger
metrics_logger = MetricsLogger(
    MetricsLoggerOptions(
        log_dir="/tmp/logs",
        project_name="my_project",
        run_name="experiment_1",
    )
)

# Log metrics during training
metrics_logger.log(
    metrics_prefix="training",
    metric_name="loss",
    scalar_value=2.5,
    mode=Mode.TRAIN,
    step=100,
)

# Retrieve aggregated metrics
avg_loss = metrics_logger.get_metric("training", "loss", Mode.TRAIN)
```

### System Metrics

**File**: `tunix/sft/system_metrics_calculator.py`, `tunix/sft/utils.py`

System metrics track hardware utilization and performance.

#### TFLOPs Measurement

**1. Static Analysis (Exact)**:
```python
def measure_tflops_per_step(
    train_step_fn: Callable[..., Any],
    model: nnx.Module,
    optimizer: nnx.ModelAndOptimizer,
    train_example: Any,
) -> float | None:
  """One-time static measurement of TFLOPs using JAX cost analysis.
  
  Returns:
    TFLOPs per training step, or None if measurement fails
  """
  if not hasattr(train_step_fn, "lower"):
    logging.warning("train_step_fn must be JIT-compiled")
    return None
  
  try:
    compiled = train_step_fn.lower(model, optimizer, train_example).compile()
    cost = compiled.cost_analysis()
    flops = cost.get("flops")
    if flops is None:
      return None
    return float(flops) / 1e12  # Convert to TFLOPs
  except Exception as e:
    logging.error("Could not measure TFLOPs: %s", e)
    return None
```

**2. Heuristic Approximation**:
```python
def approximate_tflops_per_second(
    total_model_params: int,
    global_batch_size: int,
    step_time_delta: float,
) -> float:
  """Approximate TFLOPS/s using 6*params heuristic.
  
  Heuristic: Forward + backward pass requires ~6 FLOPs per parameter.
  
  Args:
    total_model_params: Total number of model parameters
    global_batch_size: Batch size across all devices
    step_time_delta: Time taken for one training step (seconds)
  
  Returns:
    Approximate TFLOPS per second
  """
  if total_model_params <= 0 or step_time_delta <= 0:
    return 0.0
  
  # 6 * params for forward + backward pass
  flops_per_step = 6 * global_batch_size * total_model_params
  flops_per_second = flops_per_step / step_time_delta
  
  return flops_per_second / 1e12  # Convert to TFLOPS
```

#### HBM (Memory) Monitoring

**Show HBM Usage**:
```python
def show_hbm_usage(title=""):
  """Print current HBM usage across all devices."""
  fmt_size = functools.partial(humanize.naturalsize, binary=True)
  gc.collect()  # Force GC sweep
  
  if google_utils.pathways_available():
    logging.info("%s - Using Pathways compatible HBM stats", title)
    devices = jax.devices()
    hbm_stats = _pathways_hbm_usage_gb(devices)
    for i, (used, _) in enumerate(hbm_stats):
      logging.info("Using %s on %s", fmt_size(used), devices[i])
  else:
    devices = jax.local_devices()
    hbm_stats = _jax_hbm_usage_gb(devices)
    for i, (used, limit) in enumerate(hbm_stats):
      logging.info(
          "Using %s / %s (%.1f%%) on %s",
          fmt_size(used),
          fmt_size(limit),
          100 * used / limit,
          devices[i],
      )
```

**Pathways HBM Stats**:
```python
def _pathways_hbm_usage_gb(devices) -> List[Tuple[float, Optional[float]]]:
  """Returns HBM usage for each device when using Pathways."""
  live_arrays = jax.live_arrays()
  hbm_used = collections.defaultdict(int)
  hbm_limit = None  # Not available on Pathways
  
  for array in live_arrays:
    for device in array.sharding.device_set:
      hbm_used[device] += (
          array.dtype.itemsize * array.size // len(array.sharding.device_set)
      )
  
  return [(hbm_used[device], hbm_limit) for device in devices]
```

**JAX HBM Stats**:
```python
def _jax_hbm_usage_gb(devices) -> List[Tuple[float, float]]:
  """Returns HBM usage for each device using JAX."""
  hbm_used = []
  for device in devices:
    if device.platform == "cpu":
      logging.warning("Skipping non-TPU device: %s", device.platform)
      return []
    stats = device.memory_stats()
    used = stats["bytes_in_use"]
    limit = stats["bytes_limit"]
    hbm_used.append((used, limit))
  return hbm_used
```

#### Profiling

**File**: `tunix/sft/profiler.py`

**Profiler Class**:
```python
@dataclasses.dataclass(frozen=True)
class ProfilerOptions:
  log_dir: str                          # Directory for profile traces
  skip_first_n_steps: int              # Steps to skip before profiling
  profiler_steps: int                  # Number of steps to profile
  set_profile_options: bool = True     # Whether to set profile options
  host_tracer_level: int = 2           # 2 to capture HBM profiles
  python_tracer_level: int = 1         # Python tracing level

class Profiler:
  """Activate/deactivate profiler based on options."""
  
  _lock = threading.Lock()
  _is_active: bool = False
  
  def maybe_activate(self, step: int):
    """Start profiler at configured step."""
    if self._do_not_profile or step != self._first_profile_step:
      return
    
    with Profiler._lock:
      if Profiler._is_active:
        logging.warning("Profiler already active, skipping")
        return
      
      logging.info("Starting JAX profiler at step %d.", step)
      if self._profiler_options.set_profile_options:
        profile_options = jax.profiler.ProfileOptions()
        profile_options.host_tracer_level = (
            self._profiler_options.host_tracer_level
        )
        profile_options.python_tracer_level = (
            self._profiler_options.python_tracer_level
        )
        jax.profiler.start_trace(
            log_dir=self._output_path, 
            profiler_options=profile_options
        )
      else:
        jax.profiler.start_trace(log_dir=self._output_path)
      
      Profiler._is_active = True
      self._started_by_this_instance = True
  
  def maybe_deactivate(self, step: int):
    """Stop profiler at configured step."""
    if not self._started_by_this_instance or step != self._last_profile_step:
      return
    
    with Profiler._lock:
      logging.info("Stopping JAX profiler at step %d.", step)
      jax.profiler.stop_trace()
      Profiler._is_active = False
```

**Usage**:
```python
# Configure profiler
profiler_options = ProfilerOptions(
    log_dir="/tmp/profiles",
    skip_first_n_steps=10,  # Skip warmup
    profiler_steps=5,       # Profile 5 steps
    host_tracer_level=2,    # Capture HBM
)

profiler = Profiler(
    initial_step=0,
    max_step=1000,
    profiler_options=profiler_options,
)

# In training loop
for step in range(1000):
  profiler.maybe_activate(step)   # Activates at step 10
  
  # Training step
  loss = train_step(model, batch)
  
  profiler.maybe_deactivate(step)  # Deactivates at step 15
```

## 3.2 Distributed Training

### Multi-Host Setup

**Purpose**: Coordinate training across multiple machines (hosts), each with multiple devices (TPUs/GPUs).

#### Architecture

**Multi-Host Terminology**:
- **Host**: A single machine with multiple devices
- **Process**: One Python process per host
- **Device**: Individual TPU/GPU accelerator
- **Global Mesh**: Spans devices across all hosts

**Example: 4-Host Setup (32 TPUs total)**:
```
Host 0: 8 TPUs (devices 0-7)
Host 1: 8 TPUs (devices 8-15)
Host 2: 8 TPUs (devices 16-23)
Host 3: 8 TPUs (devices 24-31)

Global Mesh: (32,) or (4, 8) or (8, 4) depending on sharding strategy
```

#### JAX Multi-Host Initialization

**1. Initialize JAX Distributed**:
```python
import jax
import jax.distributed

# Initialize JAX distributed runtime
jax.distributed.initialize(
    coordinator_address=coordinator_address,  # e.g., "10.0.0.1:1234"
    num_processes=4,                          # Total number of hosts
    process_id=process_id,                    # This host's ID (0-3)
)

# Verify setup
print(f"Process {jax.process_index()} of {jax.process_count()}")
print(f"Local devices: {jax.local_devices()}")
print(f"Global devices: {jax.devices()}")
```

**2. Create Global Mesh**:
```python
# All processes must create the same mesh
devices = jax.devices()  # Global device array

# Example: FSDP across all 32 devices
mesh = jax.sharding.Mesh(devices, ("fsdp",))

# Example: 2D mesh (FSDP x TP)
mesh = jax.sharding.Mesh(
    devices.reshape(16, 2),  # 16-way FSDP, 2-way TP
    ("fsdp", "tp")
)
```

#### Cross-Host Communication

**Collective Operations**:
- **All-Reduce**: Sum/average across all devices
- **All-Gather**: Gather data from all devices
- **Reduce-Scatter**: Reduce and distribute results

**Implicit in JAX**:
```python
# JAX handles cross-host communication automatically
@jax.jit
def train_step(model, batch):
  # Gradients are automatically all-reduced across hosts
  loss, grads = jax.value_and_grad(loss_fn)(model, batch)
  
  # Apply gradients (synchronized)
  updates, opt_state = optimizer.update(grads, opt_state)
  model = optax.apply_updates(model, updates)
  
  return model, loss
```

**Explicit Communication**:
```python
# Explicit all-reduce
global_loss = jax.lax.pmean(local_loss, axis_name="batch")

# All-gather across hosts
all_predictions = jax.lax.all_gather(predictions, axis_name="batch")
```

#### Synchronization

**Barrier**:
```python
# Wait for all hosts to reach this point
jax.experimental.multihost_utils.sync_global_devices("checkpoint_saved")
```

**Process Index Guards**:
```python
# Only execute on main process (host 0)
if jax.process_index() == 0:
  print("Saving checkpoint...")
  checkpoint_manager.save(step, model)

# Wait for checkpoint to be saved
jax.experimental.multihost_utils.sync_global_devices("checkpoint_done")
```

### Resharding Operations

**File**: `tunix/rl/reshard.py`

Resharding transfers arrays between different sharding layouts, critical for distributed RL where models move between learner and inference workers.

#### Core Concepts

**Why Reshard?**:
- **Different mesh sizes**: Learner uses 8 TPUs, inference uses 32 TPUs
- **Different sharding strategies**: FSDP for training, replicated for inference
- **Memory constraints**: Large models need careful resharding to avoid OOM

**Resharding Challenges**:
- **All-gather overhead**: Expensive when gathering large models
- **Memory spikes**: Intermediate arrays can exceed HBM
- **Cross-host bandwidth**: Limited network bandwidth between hosts

#### Intermediate Sharding Optimization

**Problem**: Direct resharding from heavily sharded (e.g., 32-way FSDP) to less sharded (e.g., 8-way FSDP) requires expensive all-gather.

**Solution**: Use intermediate sharding that splits shards and replicates:

```python
# Original: [fsdp: 32]
# Target:   [fsdp: 8]
# 
# Direct approach: All-gather 32 shards → reshard to 8 (memory spike!)
#
# Optimized approach:
# Step 1: [fsdp: 32] → [fsdp_split: 8, fsdp_replica: 4]
#         (Split each target shard into 4 replicas)
# Step 2: [fsdp_split: 8, fsdp_replica: 4] → [fsdp: 8]
#         (Use any replica, no all-gather needed)
```

**`_maybe_find_intermediate_sharding()` Function**:
```python
def _maybe_find_intermediate_sharding(source_sharding, target_sharding):
  """Find intermediate sharding to avoid expensive all-gather.
  
  Args:
    source_sharding: Source NamedSharding (e.g., [fsdp: 32])
    target_sharding: Target NamedSharding (e.g., [fsdp: 8])
  
  Returns:
    Intermediate NamedSharding or None
  """
  if not isinstance(source_sharding, jax.sharding.NamedSharding) or \
     not isinstance(target_sharding, jax.sharding.NamedSharding):
    return None
  
  src_mesh = source_sharding.mesh
  dst_mesh = target_sharding.mesh
  
  # Get sharding dimensions and largest shard factor
  src_sharding_dims, src_largest_shards = _get_sharding_dims(
      source_sharding, src_mesh
  )
  dst_sharding_dims, dst_largest_shards = _get_sharding_dims(
      target_sharding, dst_mesh
  )
  
  # Check if optimization is applicable
  if src_largest_shards % dst_largest_shards != 0:
    return None  # Not divisible
  
  total_src = math.prod(list(src_sharding_dims.values()))
  total_dst = math.prod(list(dst_sharding_dims.values()))
  
  if total_src <= total_dst or total_src % total_dst != 0:
    return None  # Source not more sharded than target
  
  # Calculate replica factor
  replicas = src_largest_shards // dst_largest_shards
  
  # Find axis to split
  new_split_dim_shards = None
  new_split_axis = None
  
  for (sharding_mesh_axis_idx, src_dim_shards), (_, dst_dim_shards) in zip(
      src_sharding_dims.items(), dst_sharding_dims.items()
  ):
    gcd_dim_shards = math.gcd(src_dim_shards, dst_dim_shards)
    if gcd_dim_shards == 1:
      if src_dim_shards > dst_dim_shards and \
         src_dim_shards == src_largest_shards:
        new_split_axis = sharding_mesh_axis_idx
        new_split_dim_shards = (src_dim_shards // replicas, replicas)
  
  if new_split_axis is None:
    return None
  
  # Create intermediate mesh with split + replica axes
  new_split_mesh_axis_name = (
      src_mesh.axis_names[new_split_axis[1]] + INTERMEDIATE_SPLIT_SUFFIX
  )
  new_split_mesh_replica_axis_name = (
      src_mesh.axis_names[new_split_axis[1]] + INTERMEDIATE_REPLICA_SUFFIX
  )
  
  intermediate_mesh = jax.sharding.Mesh(
      src_mesh.devices.reshape(
          tuple(
              list(src_mesh.devices.shape[:new_split_axis[1]])
              + [new_split_dim_shards[0], new_split_dim_shards[1]]
              + list(src_mesh.devices.shape[new_split_axis[1] + 1:])
          )
      ),
      tuple(
          list(src_mesh.axis_names[:new_split_axis[1]])
          + [new_split_mesh_axis_name, new_split_mesh_replica_axis_name]
          + list(src_mesh.axis_names[new_split_axis[1] + 1:])
      ),
  )
  
  # Create intermediate partition spec
  intermediate_spec = list(source_sharding.spec)
  intermediate_spec[new_split_axis[0]] = (
      new_split_mesh_axis_name,
      new_split_mesh_replica_axis_name,
  )
  
  intermediate_sharding = jax.sharding.NamedSharding(
      intermediate_mesh,
      jax.sharding.PartitionSpec(*intermediate_spec),
  )
  
  return intermediate_sharding
```

#### Resharding Functions

**1. Standard Resharding**:
```python
def reshard_arrays(
    arrays: jaxtyping.PyTree,
    source_sharding: jax.sharding.Sharding,
    target_sharding: jax.sharding.Sharding,
) -> jaxtyping.PyTree:
  """Reshard arrays from source to target sharding.
  
  Automatically uses intermediate sharding if beneficial.
  """
  intermediate_sharding = _maybe_find_intermediate_sharding(
      source_sharding, target_sharding
  )
  
  if intermediate_sharding is not None:
    logging.info("Using intermediate sharding for optimization")
    # Step 1: Source → Intermediate
    arrays = jax.tree.map(
        lambda x: jax.lax.with_sharding_constraint(x, intermediate_sharding),
        arrays
    )
    # Step 2: Intermediate → Target
    arrays = jax.tree.map(
        lambda x: jax.lax.with_sharding_constraint(x, target_sharding),
        arrays
    )
  else:
    # Direct resharding
    arrays = jax.tree.map(
        lambda x: jax.lax.with_sharding_constraint(x, target_sharding),
        arrays
    )
  
  return arrays
```

**2. Model Resharding**:
```python
def reshard_model_to_mesh(model: nnx.Module, mesh: jax.sharding.Mesh):
  """Reshard entire model to new mesh.
  
  Args:
    model: Flax NNX model
    mesh: Target mesh
  
  Returns:
    Model with parameters resharded to target mesh
  """
  # Get current parameters
  state = nnx.state(model)
  
  # Create target shardings (replicate all parameters)
  target_shardings = jax.tree.map(
      lambda x: jax.sharding.NamedSharding(
          mesh, jax.sharding.PartitionSpec()
      ),
      state
  )
  
  # Reshard parameters
  resharded_state = reshard_arrays(
      state,
      source_sharding=None,  # Infer from arrays
      target_sharding=target_shardings,
  )
  
  # Update model with resharded parameters
  nnx.update(model, resharded_state)
  
  return model
```

#### Pathways Resharding

**PathwaysUtils Integration**:
```python
def _get_reshard_fn_pathwaysutils(
    source_sharding, 
    target_sharding, 
    intermediate_sharding=None
):
  """Get reshard function using PathwaysUtils for optimized cross-host transfer.
  
  PathwaysUtils provides:
  - Optimized cross-pod communication
  - Reduced memory overhead
  - Better performance on large models
  """
  try:
    from pathwaysutils.experimental import reshard as experimental_reshard
    from pathwaysutils.experimental import split_by_mesh_axis
    from pathwaysutils import jax as pw_jax
  except ImportError:
    logging.error('Cannot import PathwaysUtils')
    return None
  
  if intermediate_sharding is not None:
    # Two-stage resharding with PathwaysUtils
    def reshard_fn(x):
      # Stage 1: Source → Intermediate
      x = experimental_reshard(x, intermediate_sharding)
      # Stage 2: Intermediate → Target
      x = experimental_reshard(x, target_sharding)
      return x
  else:
    # Direct resharding
    reshard_fn = lambda x: experimental_reshard(x, target_sharding)
  
  return reshard_fn
```

### Cross-Device Communication

#### Communication Primitives

**1. All-Reduce** (Sum across devices):
```python
# Reduce loss across all devices
global_loss = jax.lax.psum(local_loss, axis_name="batch")

# Average gradients
avg_grads = jax.tree.map(
    lambda g: jax.lax.pmean(g, axis_name="batch"),
    grads
)
```

**2. All-Gather** (Collect from all devices):
```python
# Gather predictions from all devices
all_predictions = jax.lax.all_gather(
    local_predictions, 
    axis_name="batch"
)
# Shape: [num_devices, local_batch_size, ...] → [global_batch_size, ...]
```

**3. Reduce-Scatter** (Reduce and distribute):
```python
# Each device gets a slice of the reduced result
scattered_result = jax.lax.psum_scatter(
    local_data,
    axis_name="batch",
    scatter_dimension=0,
    tiled=True,
)
```

#### Axis Names and pmap

**Using `pmap` for Data Parallelism**:
```python
# Define parallel training step
@functools.partial(jax.pmap, axis_name="batch")
def parallel_train_step(model, batch):
  # Compute loss and gradients
  loss, grads = jax.value_and_grad(loss_fn)(model, batch)
  
  # Average gradients across devices
  grads = jax.tree.map(
      lambda g: jax.lax.pmean(g, axis_name="batch"),
      grads
  )
  
  # Update model
  model = update_model(model, grads)
  
  # Average loss across devices
  loss = jax.lax.pmean(loss, axis_name="batch")
  
  return model, loss

# Replicate model across devices
replicated_model = jax.device_put_replicated(model, jax.local_devices())

# Run parallel training
replicated_model, loss = parallel_train_step(replicated_model, batch)
```

#### Communication Costs

**Bandwidth Considerations**:
- **Intra-host**: ~600 GB/s (between TPUs on same chip)
- **Inter-host**: ~100 GB/s (network bandwidth)

**Minimizing Communication**:
```python
# ❌ Bad: Sync every micro-step
for micro_batch in batches:
  grads = compute_grads(model, micro_batch)
  grads = jax.lax.pmean(grads, "batch")  # Sync!
  model = apply_grads(model, grads)

# ✅ Good: Accumulate, then sync
accumulated_grads = init_grads()
for micro_batch in batches:
  grads = compute_grads(model, micro_batch)
  accumulated_grads = add_grads(accumulated_grads, grads)

# Single sync at end
accumulated_grads = jax.lax.pmean(accumulated_grads, "batch")
model = apply_grads(model, accumulated_grads)
```

### Pathways Integration

**Pathways** is Google's infrastructure for orchestrating distributed ML workloads across multiple hosts and accelerators.

#### Key Features

**1. Unified Resource Management**:
- Allocates TPU/GPU resources across pods
- Handles multi-host coordination
- Optimized cross-pod communication

**2. Special Handling in Tunix**:

**Checkpoint Manager**:
```python
# Pathways requires special persistence APIs
if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
  item_handlers = {
      'model_params': ocp.PyTreeCheckpointHandler(
          use_ocdbt=False,  # Disable OCDBT on Pathways
          use_zarr3=False,  # Disable Zarr3 on Pathways
      ),
  }
  logging.info('Using persistence APIs for checkpointing with Pathways.')
```

**HBM Monitoring**:
```python
# Pathways doesn't expose device.memory_stats()
if google_utils.pathways_available():
  hbm_stats = _pathways_hbm_usage_gb(devices)
  # Must calculate HBM usage from live arrays
else:
  hbm_stats = _jax_hbm_usage_gb(devices)
  # Use device.memory_stats()
```

**3. Resharding Optimization**:
```python
# PathwaysUtils provides experimental resharding API
from pathwaysutils.experimental import reshard as experimental_reshard

# Optimized cross-pod resharding
resharded_array = experimental_reshard(array, target_sharding)
```

#### Environment Detection

**Check if Pathways is Active**:
```python
def pathways_available() -> bool:
  """Check if running on Pathways."""
  return 'proxy' in os.getenv('JAX_PLATFORMS', '')

# Usage
if pathways_available():
  # Use Pathways-specific code paths
  pass
else:
  # Use standard JAX code paths
  pass
```

## 3.3 Data Pipeline

### Dataset Sources

Tunix supports multiple dataset sources for training data.

#### Supported Sources

**1. TensorFlow Datasets (TFDS)**:
```python
import tensorflow_datasets as tfds

# Load dataset from TFDS
train_ds, eval_ds = tfds.data_source(
    "mtnt/en-fr",
    split=("train", "valid"),
    download=True,  # Download if not cached
)
```

**2. Hugging Face Datasets**:
```python
import datasets

# Load from Hugging Face Hub
train_ds, eval_ds = datasets.load_dataset(
    "Helsinki-NLP/opus-100",
    data_dir="en-fr",
    split=("train", "validation")
)
```

**3. Google Cloud Storage (GCS)**:
```python
import fsspec

# Load from GCS
with fsspec.open('gs://bucket/path/data.json', 'r') as f:
  data = json.load(f)
```

**4. Local Files**:
```python
# JSON files
with open('/path/to/data.json', 'r') as f:
  data = json.load(f)

# Custom Python modules
from myproject.data import my_dataset
```

#### Dataset Module Specification

**File**: `tunix/cli/utils/data.py`

**Format**: `module_path:function_name(args)`

**Examples**:
```python
# Default function (create_dataset)
dataset = get_dataset_from_module("data.coding", tokenizer)

# Specific function
dataset = get_dataset_from_module("data.coding:create_dataset", tokenizer)

# With arguments
dataset = get_dataset_from_module(
    "data.coding:create_dataset(name='coding_v0')", 
    tokenizer
)

# From file path
dataset = get_dataset_from_module(
    "/home/user/project/data/coding.py:get_dataset",
    tokenizer
)
```

**Implementation**:
```python
def get_dataset_from_module(specifier: str, tokenizer: TokenizerAdapter):
  """Get dataset from module specifier.
  
  Args:
    specifier: Module specifier with optional function and args
    tokenizer: Tokenizer to apply
  
  Returns:
    Grain MapDataset with chat templates applied
  """
  # Parse specifier into parts
  if "(" in specifier and ":" in specifier:
    specifier, args_part = specifier.rsplit("(", 1)
  else:
    args_part = ""
  
  if ":" in specifier:
    specifier, func_spec = specifier.rsplit(":", 1)
  else:
    func_spec = ""
  
  # Load module
  if os.path.exists(specifier) and specifier.endswith(".py"):
    # Load from file
    module_name = os.path.splitext(os.path.basename(specifier))[0]
    spec = importlib.util.spec_from_file_location(module_name, specifier)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
  else:
    # Import as Python module
    module = importlib.import_module(specifier)
  
  # Get function
  if func_spec:
    func = getattr(module, func_spec)
    if args_part:
      args_part = args_part.rstrip(")")
      args, kwargs = parse_call_string(args_part)
  else:
    func = module.create_dataset
    args, kwargs = [], {}
  
  # Create dataset and apply chat template
  dataset = func(*args, **kwargs)
  return dataset.map(
      functools.partial(apply_chat_template, tokenizer=tokenizer)
  )
```

### Data Loading with Grain

**Grain** is Google's data loading library for ML, providing efficient, scalable data pipelines.

**File**: `tunix/examples/data/translation_dataset.py`, `tunix/utils/script_utils.py`

#### Core Components

**1. DataSource** (Random access to examples):
```python
# TFDS as data source
train_ds = tfds.data_source("gsm8k", split="train")

# List/array as data source
data = [{"question": "...", "answer": "..."}, ...]
dataset = grain.MapDataset.source(data)
```

**2. Sampler** (Controls iteration order):
```python
sampler = grain.IndexSampler(
    num_records=len(data_source),
    num_epochs=3,                      # Repeat for 3 epochs
    shard_options=grain.NoSharding(),  # No sharding (single process)
    shuffle=True,                      # Shuffle data
    seed=42,                           # Random seed
)
```

**3. Transformations** (Process data):
```python
class Tokenize(grain.MapTransform):
  """Tokenize text."""
  
  def __init__(self, tokenizer):
    self._tokenizer = tokenizer
  
  def map(self, element):
    """Apply tokenization to one example."""
    return {
        'input_tokens': self._tokenizer.tokenize(element['text']),
        'label': element['label'],
    }

class FilterShort(grain.FilterTransform):
  """Filter out short sequences."""
  
  def __init__(self, min_length):
    self._min_length = min_length
  
  def filter(self, element):
    """Return True to keep element."""
    return len(element['input_tokens']) >= self._min_length
```

**4. DataLoader** (Combines everything):
```python
dataloader = grain.DataLoader(
    data_source=train_ds,
    sampler=sampler,
    operations=[
        Tokenize(tokenizer),
        FilterShort(min_length=10),
        grain.Batch(batch_size=32, drop_remainder=True),
    ],
)

# Iterate
for batch in dataloader:
  # batch is a dict with batched arrays
  print(batch['input_tokens'].shape)  # (32, seq_len)
```

#### Complete Example: Translation Dataset

**`create_datasets()` Function**:
```python
def create_datasets(
    dataset_name: str,
    global_batch_size: int,
    max_target_length: int,
    num_train_epochs: int | None,
    tokenizer: tokenizer_lib.Tokenizer,
    instruct_tuned: bool = False,
    tfds_download: bool = True,
    input_template: dict[str, str] | None = None,
) -> tuple[Iterable[TrainingInput], Iterable[TrainingInput]]:
  """Create train and eval data iterators.
  
  Args:
    dataset_name: Dataset name (e.g., "mtnt/en-fr")
    global_batch_size: Batch size across all devices
    max_target_length: Maximum sequence length
    num_train_epochs: Number of training epochs (None = infinite)
    tokenizer: Tokenizer instance
    instruct_tuned: Whether to use instruction-tuned template
    tfds_download: Download flag for TFDS
    input_template: Custom input template
  
  Returns:
    Tuple of (train_loader, eval_loader)
  """
  # Load data source
  if dataset_name == "mtnt/en-fr":
    train_ds, eval_ds = tfds.data_source(
        dataset_name, 
        split=("train", "valid"), 
        download=tfds_download
    )
  elif dataset_name == "Helsinki-NLP/opus-100":
    train_ds, eval_ds = datasets.load_dataset(
        dataset_name, 
        data_dir="en-fr", 
        split=("train", "validation")
    )
  else:
    raise ValueError(f"Unsupported dataset: {dataset_name}")
  
  # Select input template
  input_template = INPUT_TEMPLATE_IT if instruct_tuned else INPUT_TEMPLATE
  
  # Build data loaders
  train_loader = _build_data_loader(
      data_source=train_ds,
      batch_size=global_batch_size,
      num_epochs=num_train_epochs,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  
  eval_loader = _build_data_loader(
      data_source=eval_ds,
      batch_size=global_batch_size,
      num_epochs=1,
      max_seq_len=max_target_length,
      tokenizer=tokenizer,
      input_template=input_template,
  )
  
  return train_loader, eval_loader
```

**`_build_data_loader()` Function**:
```python
def _build_data_loader(
    *,
    data_source: grain.RandomAccessDataSource,
    batch_size: int,
    num_epochs: int | None,
    max_seq_len: int,
    tokenizer: tokenizer_lib.Tokenizer,
    input_template: dict[str, str],
) -> grain.DataLoader:
  """Build a data loader for the given data source."""
  return grain.DataLoader(
      data_source=data_source,
      sampler=grain.IndexSampler(
          num_records=len(data_source),
          num_epochs=num_epochs,
          shard_options=grain.NoSharding(),
      ),
      operations=[
          _Tokenize(tokenizer, input_template),
          _BuildTrainInput(max_seq_len, tokenizer.pad_id()),
          _FilterOverlength(max_seq_len),
          grain.Batch(batch_size=batch_size, drop_remainder=True),
      ],
  )
```

#### Custom Transformations

**1. Tokenization**:
```python
class _Tokenize(grain.MapTransform):
  """Tokenize the input."""
  
  def __init__(
      self, 
      tokenizer: tokenizer_lib.Tokenizer, 
      input_template: dict[str, str]
  ):
    self._tokenizer = tokenizer
    self._input_template = input_template
  
  def map(self, element: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize source and target text."""
    if "src" in element.keys():  # MTNT dataset
      src_tokens = self._tokenizer.tokenize(
          element["src"].decode(),
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["dst"].decode(), 
          add_eos=True
      )
    else:  # OPUS-100 dataset
      src_tokens = self._tokenizer.tokenize(
          element["translation"]["en"],
          prefix=self._input_template["prefix"],
          suffix=self._input_template["suffix"],
          add_eos=False,
      )
      dst_tokens = self._tokenizer.tokenize(
          element["translation"]["fr"], 
          add_eos=True
      )
    
    return src_tokens, dst_tokens
```

**2. Build Training Input**:
```python
class _BuildTrainInput(grain.MapTransform):
  """Build TrainingInput from source and destination tokens."""
  
  def __init__(self, max_seq_len: int, pad_value: int | bool):
    self._max_seq_len = max_seq_len
    self._pad_value = pad_value
  
  def map(self, tokens: tuple[np.ndarray, np.ndarray]) -> TrainingInput:
    src_tokens, dst_tokens = tokens
    
    # Concatenate source and destination
    tokens = np.concat([src_tokens, dst_tokens], axis=0)
    
    # Create mask (don't train on source tokens)
    q_mask = np.zeros_like(src_tokens, dtype=np.bool)
    a_mask = np.ones_like(dst_tokens, dtype=np.bool)
    mask = np.concat([q_mask, a_mask], axis=0)
    
    # Pad to max length
    tokens = self._pad_up_to_max_len(tokens, self._pad_value)
    mask = self._pad_up_to_max_len(mask, 0)
    
    return TrainingInput(input_tokens=tokens, input_mask=mask)
  
  def _pad_up_to_max_len(
      self, 
      input_tensor: np.ndarray, 
      pad_value: int
  ) -> np.ndarray:
    """Pad tensor up to max sequence length."""
    seq_len = input_tensor.shape[0]
    to_pad = np.maximum(self._max_seq_len - seq_len, 0)
    return np.pad(
        input_tensor,
        [[0, to_pad]],
        mode="constant",
        constant_values=pad_value,
    )
```

**3. Filtering**:
```python
class _FilterOverlength(grain.FilterTransform):
  """Filter out sequences longer than max length."""
  
  def __init__(self, max_seq_len: int):
    self._max_seq_len = max_seq_len
  
  def filter(self, element: TrainingInput) -> bool:
    """Return True to keep element."""
    return len(element.input_tokens) <= self._max_seq_len
```

### Tokenization Strategies

#### Chat Templates

**File**: `tunix/cli/utils/data.py`

**Apply Chat Template**:
```python
def apply_chat_template(x, tokenizer: TokenizerAdapter) -> dict[str, Any]:
  """Apply chat template to prompt field.
  
  Converts:
    {"prompt": [{"role": "user", "content": "Hello"}]}
  To:
    {"prompts": "<start_of_turn>user\nHello\n<end_of_turn>\n..."}
  """
  return {
      "prompts": tokenizer.apply_chat_template(
          x["prompt"], 
          tokenize=False, 
          add_generation_prompt=True
      ),
      **{k: v for k, v in x.items() if k != "prompt"},
  }
```

**Usage**:
```python
dataset = dataset.map(
    functools.partial(apply_chat_template, tokenizer=tokenizer)
)
```

#### Input Templates

**Standard Template**:
```python
INPUT_TEMPLATE = {
    "prefix": "Translate this into French:\n",
    "suffix": "\n",
}
```

**Instruction-Tuned Template** (Gemma-style):
```python
INPUT_TEMPLATE_IT = {
    "prefix": "<start_of_turn>user\nTranslate this into French:\n",
    "suffix": "\n<end_of_turn>\n<start_of_turn>model\n",
}
```

**Usage**:
```python
src_tokens = tokenizer.tokenize(
    text,
    prefix=input_template["prefix"],
    suffix=input_template["suffix"],
    add_eos=False,
)
```

### Batch Construction

#### Post-Processing

**File**: `tunix/cli/utils/data.py`

**`post_init_dataset()` Function**:
```python
def post_init_dataset(
    dataset,
    tokenizer: Tokenizer,
    batch_size: int,
    num_batches: int | None,
    max_prompt_length: int | None,
    fraction: float = 1.0,
    num_epochs: int = 1,
):
  """Apply post-initialization transformations to dataset.
  
  Args:
    dataset: Input Grain dataset
    tokenizer: Tokenizer instance
    batch_size: Batch size
    num_batches: Maximum number of batches (None = unlimited)
    max_prompt_length: Maximum prompt length (filter longer)
    fraction: Fraction of dataset to use (for debugging)
    num_epochs: Number of epochs to repeat
  
  Returns:
    Transformed dataset
  """
  # Filter by prompt length
  if max_prompt_length is not None:
    dataset = dataset.filter(
        lambda x: len(tokenizer.tokenize(x["prompts"])) <= max_prompt_length
    )
  
  # Take fraction of dataset
  if fraction < 1.0:
    num_examples = int(len(dataset) * fraction)
    dataset = dataset[:num_examples]
  
  # Repeat for multiple epochs
  if num_epochs > 1:
    dataset = dataset.repeat(num_epochs)
  
  # Batch
  dataset = dataset.batch(batch_size, drop_remainder=True)
  
  # Limit number of batches
  if num_batches is not None:
    dataset = dataset[:num_batches]
  
  return dataset
```

#### Train/Eval Split

**`get_train_and_eval_datasets()` Function**:
```python
def get_train_and_eval_datasets(
    data_path: str,
    split: str,
    seed: int,
    system_prompt: str,
    batch_size: int,
    num_batches: int | None,
    train_fraction: float,
    num_epochs: int | None,
    *,
    answer_extractor: Callable[[str], str | None],
    dataset_name: str = 'gsm8k',
) -> tuple[grain.MapDataset, grain.MapDataset | None]:
  """Create train and eval datasets from single split.
  
  Args:
    data_path: Path to dataset
    split: Split to use (e.g., "train")
    seed: Random seed for shuffling
    system_prompt: System prompt to prepend
    batch_size: Batch size
    num_batches: Max batches per dataset
    train_fraction: Fraction to use for training (rest for eval)
    num_epochs: Number of epochs
    answer_extractor: Function to extract answer from text
    dataset_name: Name of dataset
  
  Returns:
    Tuple of (train_dataset, eval_dataset)
  """
  # Load base dataset
  dataset = get_dataset(
      path=data_path,
      split=split,
      seed=seed,
      system_prompt=system_prompt,
      answer_extractor=answer_extractor,
      dataset_name=dataset_name,
  )
  
  # Calculate split size
  total_size = len(dataset)
  train_size = int(total_size * train_fraction)
  
  # Split into train and eval
  train_dataset = dataset[:train_size]
  eval_dataset = dataset[train_size:] if train_fraction < 1.0 else None
  
  # Repeat train dataset for multiple epochs
  if num_epochs is not None:
    train_dataset = train_dataset.repeat(num_epochs)
  
  # Batch both datasets
  train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
  if eval_dataset is not None:
    eval_dataset = eval_dataset.batch(batch_size, drop_remainder=True)
  
  # Limit number of batches
  if num_batches is not None:
    train_dataset = train_dataset[:num_batches]
    if eval_dataset is not None:
      eval_dataset = eval_dataset[:num_batches]
  
  return train_dataset, eval_dataset
```

#### Sharding for Multi-Process

**For Distributed Training**:
```python
# Each process gets a different shard
sampler = grain.IndexSampler(
    num_records=len(data_source),
    num_epochs=num_epochs,
    shard_options=grain.ShardByJaxProcess(),  # Shard by JAX process
)

# Alternative: Manual sharding
sampler = grain.IndexSampler(
    num_records=len(data_source),
    num_epochs=num_epochs,
    shard_options=grain.ShardOptions(
        shard_index=jax.process_index(),
        shard_count=jax.process_count(),
    ),
)
```

## Usage Examples

### Example 1: Single-Host Training with FSDP

```python
import jax
import jax.sharding as shd
from tunix.sft import peft_trainer, sharding_utils

# Initialize mesh
devices = jax.local_devices()  # 8 TPUs
mesh = shd.Mesh(devices, ("fsdp",))

# Configure trainer
config = peft_trainer.PeftTrainerConfig(
    data_sharding_axis=("fsdp",),
    learning_rate=1e-4,
    num_train_steps=1000,
)

# Create trainer
with mesh:
  trainer = peft_trainer.PeftTrainer(config)
  
  # Shard input data
  for step, batch in enumerate(train_loader):
    batch = sharding_utils.shard_input(batch, ("fsdp",))
    
    # Training step
    loss = trainer.train_step(model, batch)
    
    if step % 100 == 0:
      print(f"Step {step}, Loss: {loss}")
```

### Example 2: Multi-Host Training Setup

```python
import jax
import jax.distributed

# Initialize distributed JAX (on each host)
jax.distributed.initialize(
    coordinator_address="10.0.0.1:1234",
    num_processes=4,
    process_id=os.environ["JAX_PROCESS_ID"],  # 0-3
)

# Create global mesh across all hosts
devices = jax.devices()  # 32 TPUs total
mesh = jax.sharding.Mesh(
    devices.reshape(16, 2),  # 16-way FSDP, 2-way TP
    ("fsdp", "tp")
)

# Training loop
with mesh:
  for step, batch in enumerate(train_loader):
    # Only main process logs
    if jax.process_index() == 0 and step % 10 == 0:
      print(f"Step {step}")
    
    # All processes train
    loss = train_step(model, batch)
    
    # Save checkpoint (main process only)
    if jax.process_index() == 0 and step % 100 == 0:
      checkpoint_manager.save(step, model)
    
    # Wait for checkpoint to complete
    jax.experimental.multihost_utils.sync_global_devices("checkpoint_done")
```

### Example 3: Checkpoint Management

```python
from tunix.sft import checkpoint_manager
import orbax.checkpoint as ocp

# Initialize checkpoint manager
ckpt_options = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=300,  # Save every 5 minutes
    ),
    max_to_keep=5,  # Keep 5 most recent checkpoints
)

ckpt_mgr = checkpoint_manager.CheckpointManager(
    root_directory="/tmp/checkpoints",
    options=ckpt_options,
)

# Restore from checkpoint
restored_step, metadata = ckpt_mgr.maybe_restore(model)
print(f"Restored from step {restored_step}")

# Training loop
for step in range(restored_step, num_steps):
  loss = train_step(model, batch)
  
  # Save checkpoint (respects save_decision_policy)
  saved = ckpt_mgr.save(
      step=step,
      model=model,
      custom_metadata={"loss": float(loss)},
  )
  
  if saved:
    print(f"Saved checkpoint at step {step}")
```

### Example 4: Metrics Logging

```python
from tunix.sft import metrics_logger

# Initialize logger
logger = metrics_logger.MetricsLogger(
    metrics_logger.MetricsLoggerOptions(
        log_dir="/tmp/logs",
        project_name="tunix_experiments",
        run_name="gemma2_sft_run1",
        flush_every_n_steps=50,
    )
)

# Training loop
for step in range(num_steps):
  # Training
  train_loss = train_step(model, train_batch)
  logger.log(
      metrics_prefix="training",
      metric_name="loss",
      scalar_value=float(train_loss),
      mode=metrics_logger.Mode.TRAIN,
      step=step,
  )
  
  # Evaluation
  if step % 100 == 0:
    eval_loss = evaluate(model, eval_loader)
    logger.log(
        metrics_prefix="evaluation",
        metric_name="loss",
        scalar_value=float(eval_loss),
        mode=metrics_logger.Mode.EVAL,
        step=step,
    )
    
    # Get average metrics
    avg_train_loss = logger.get_metric("training", "loss", "train")
    print(f"Step {step}, Avg Train Loss: {avg_train_loss:.4f}")
```

### Example 5: System Metrics Monitoring

```python
from tunix.sft import system_metrics_calculator, utils
import time

# Measure TFLOPs (one-time static analysis)
tflops_per_step = system_metrics_calculator.measure_tflops_per_step(
    train_step_fn=jax.jit(train_step),
    model=model,
    optimizer=optimizer,
    train_example=sample_batch,
)
print(f"TFLOPs per step: {tflops_per_step:.2f}")

# Training loop with performance tracking
for step in range(num_steps):
  # Track HBM before step
  if step % 100 == 0:
    utils.show_hbm_usage(f"Step {step} - Before")
  
  # Time the step
  start_time = time.time()
  loss = train_step(model, batch)
  step_time = time.time() - start_time
  
  # Calculate TFLOPS/s
  tflops_per_sec = system_metrics_calculator.approximate_tflops_per_second(
      total_model_params=model_params_count,
      global_batch_size=batch_size * jax.device_count(),
      step_time_delta=step_time,
  )
  
  print(f"Step {step}: {step_time:.3f}s, {tflops_per_sec:.2f} TFLOPS/s")
  
  # Track HBM after step
  if step % 100 == 0:
    utils.show_hbm_usage(f"Step {step} - After")
```

### Example 6: Profiling Training

```python
from tunix.sft import profiler

# Configure profiler
profiler_opts = profiler.ProfilerOptions(
    log_dir="/tmp/profiles",
    skip_first_n_steps=10,    # Skip warmup steps
    profiler_steps=5,         # Profile 5 steps
    host_tracer_level=2,      # Capture HBM
    python_tracer_level=1,    # Capture Python calls
)

prof = profiler.Profiler(
    initial_step=0,
    max_step=1000,
    profiler_options=profiler_opts,
)

# Training loop
for step in range(1000):
  prof.maybe_activate(step)    # Activates at step 10
  
  loss = train_step(model, batch)
  
  prof.maybe_deactivate(step)   # Deactivates at step 15

# View profile: tensorboard --logdir=/tmp/profiles
```

### Example 7: Data Pipeline with Grain

```python
from grain import python as grain
from tunix.generate import tokenizer_adapter

# Load tokenizer
tokenizer = tokenizer_adapter.TokenizerAdapter.from_pretrained(
    "google/gemma-2-2b"
)

# Create dataset
data = [
    {"text": "Hello, world!", "label": 0},
    {"text": "How are you?", "label": 1},
    # ... more examples
]

# Define transformations
class Tokenize(grain.MapTransform):
  def __init__(self, tokenizer):
    self._tokenizer = tokenizer
  
  def map(self, element):
    tokens = self._tokenizer.tokenize(element["text"])
    return {
        "input_tokens": tokens,
        "label": element["label"],
    }

# Build data loader
dataset = grain.MapDataset.source(data)
dataloader = grain.DataLoader(
    data_source=dataset,
    sampler=grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=3,
        shuffle=True,
        seed=42,
    ),
    operations=[
        Tokenize(tokenizer),
        grain.Batch(batch_size=32, drop_remainder=True),
    ],
)

# Iterate
for batch in dataloader:
  print(batch["input_tokens"].shape)  # (32, seq_len)
```

### Example 8: Model Resharding

```python
from tunix.rl import reshard

# Source: Model on learner mesh (8 devices, FSDP)
learner_mesh = jax.sharding.Mesh(jax.devices()[:8], ("fsdp",))
with learner_mesh:
  # Train model
  model = train(model)

# Target: Model on inference mesh (32 devices, replicated)
inference_mesh = jax.sharding.Mesh(jax.devices(), ("replica",))

# Reshard model to inference mesh
model = reshard.reshard_model_to_mesh(model, inference_mesh)

# Now model can be used on inference mesh
with inference_mesh:
  predictions = model(inputs)
```

### Example 9: Custom Data Module

```python
# File: my_dataset.py
from grain import python as grain
import numpy as np

def create_dataset(name: str = "default", split: str = "train"):
  """Create custom dataset.
  
  Args:
    name: Dataset variant name
    split: Data split (train/eval)
  
  Returns:
    Grain MapDataset
  """
  # Load your data
  data = load_my_data(name, split)
  
  # Process into format expected by Tunix
  processed = []
  for item in data:
    processed.append({
        "prompt": [
            {"role": "user", "content": item["question"]},
        ],
        "response": item["answer"],
    })
  
  return grain.MapDataset.source(processed)

# Usage in CLI:
# --dataset_name my_dataset:create_dataset(name='v2', split='train')
```

### Example 10: Multi-Process Data Loading

```python
import jax
from grain import python as grain

# Initialize JAX distributed
jax.distributed.initialize(...)

# Create data loader with sharding
dataloader = grain.DataLoader(
    data_source=train_ds,
    sampler=grain.IndexSampler(
        num_records=len(train_ds),
        num_epochs=3,
        # Each process gets different shard
        shard_options=grain.ShardByJaxProcess(),
    ),
    operations=[
        Tokenize(tokenizer),
        grain.Batch(batch_size=32, drop_remainder=True),
    ],
)

# Each process iterates over its own shard
for batch in dataloader:
  # Process ID 0 gets examples 0, 4, 8, ...
  # Process ID 1 gets examples 1, 5, 9, ...
  # etc.
  loss = train_step(model, batch)
```

## Best Practices

### 1. Sharding Strategy Selection

**Choose FSDP for**:
- Large models (7B+ parameters)
- Memory-constrained scenarios
- Single mesh axis suffices

**Choose Tensor Parallelism for**:
- Very large models (70B+ parameters)
- When FSDP alone isn't enough
- Combined with FSDP in 2D mesh

**Choose Data Parallelism for**:
- Small models (< 3B parameters)
- Compute-bound workloads
- When memory is not a concern

**Example Configuration**:
```python
# Small model (2B): Data parallel
mesh = jax.sharding.Mesh(devices, ("data",))
# Replicate model, shard data

# Medium model (7B): FSDP
mesh = jax.sharding.Mesh(devices, ("fsdp",))
# Shard model across all devices

# Large model (70B): FSDP + TP
mesh = jax.sharding.Mesh(
    devices.reshape(16, 2),
    ("fsdp", "tp")
)
# 16-way FSDP, 2-way tensor parallel
```

### 2. Checkpoint Best Practices

**✅ DO**:
- Set reasonable save intervals (3-5 minutes minimum)
- Limit `max_to_keep` to avoid filling disk
- Save custom metadata for reproducibility
- Test restoration early in training
- Use `force=True` for important milestones

**❌ DON'T**:
- Save every step (too expensive)
- Keep unlimited checkpoints (disk space)
- Skip testing checkpoint restoration
- Forget to sync processes after save

**Example**:
```python
# ✅ Good checkpoint configuration
options = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=300,  # 5 minutes
    ),
    max_to_keep=5,
)

# ✅ Save with metadata
ckpt_mgr.save(
    step=step,
    model=model,
    custom_metadata={
        "train_loss": float(train_loss),
        "eval_accuracy": float(eval_acc),
        "learning_rate": float(lr),
        "config": dataclasses.asdict(config),
    },
)

# ✅ Test restoration early
restored_step, metadata = ckpt_mgr.maybe_restore(model)
assert restored_step > 0 or step == 0, "Checkpoint restoration failed!"
```

### 3. Metrics Logging Guidelines

**Structure**:
- Use meaningful prefixes (`training`, `evaluation`, `system`)
- Consistent metric names across runs
- Log at appropriate frequencies
- Flush periodically to avoid data loss

**What to Log**:
- **Training**: loss, learning_rate, gradient_norm
- **Evaluation**: loss, accuracy, perplexity
- **System**: tflops_per_sec, hbm_usage, step_time

**Example**:
```python
# ✅ Good logging structure
logger.log("training", "loss", loss, Mode.TRAIN, step)
logger.log("training", "learning_rate", lr, Mode.TRAIN, step)
logger.log("training", "gradient_norm", grad_norm, Mode.TRAIN, step)

if step % 100 == 0:
  logger.log("evaluation", "loss", eval_loss, Mode.EVAL, step)
  logger.log("evaluation", "accuracy", eval_acc, Mode.EVAL, step)
  logger.log("system", "tflops_per_sec", tflops, Mode.TRAIN, step)

# ❌ Avoid inconsistent naming
logger.log("train", "training_loss", loss, Mode.TRAIN, step)  # Different prefix
logger.log("training", "train_loss", loss, Mode.TRAIN, step)   # Different name
```

### 4. Multi-Host Training Tips

**Initialization**:
- Start coordinator first (one designated host)
- Ensure all processes use same coordinator address
- Verify `jax.device_count() == num_processes * devices_per_host`

**Synchronization**:
- Use barriers after collective operations
- Only main process saves checkpoints
- Sync before exiting training loop

**Debugging**:
- Log process ID with all messages
- Test with 2 hosts before scaling to many
- Monitor network bandwidth usage

**Example**:
```python
# ✅ Good multi-host setup
def main():
  # Initialize
  jax.distributed.initialize(
      coordinator_address=f"{coordinator_host}:1234",
      num_processes=jax.process_count(),
      process_id=jax.process_index(),
  )
  
  # Verify setup
  if jax.process_index() == 0:
    print(f"Initialized {jax.process_count()} processes")
    print(f"Total devices: {jax.device_count()}")
  
  # Training loop
  for step in range(num_steps):
    # All processes train
    loss = train_step(model, batch)
    
    # Main process saves
    if jax.process_index() == 0 and step % 100 == 0:
      ckpt_mgr.save(step, model)
    
    # Wait for save to complete
    jax.experimental.multihost_utils.sync_global_devices("ckpt_done")
  
  # Final sync before exit
  jax.experimental.multihost_utils.sync_global_devices("training_done")
```

### 5. Data Pipeline Optimization

**Performance**:
- Use `grain.DataLoader` for efficient iteration
- Shard data across processes
- Use `drop_remainder=True` for consistent batch sizes
- Pre-tokenize large datasets

**Memory**:
- Filter before batching
- Use streaming for large datasets
- Limit cache size for shuffled datasets

**Debugging**:
- Start with small fraction of data
- Test transformations on single example
- Verify batch shapes before training

**Example**:
```python
# ✅ Optimized data pipeline
dataloader = grain.DataLoader(
    data_source=dataset,
    sampler=grain.IndexSampler(
        num_records=len(dataset),
        num_epochs=3,
        shuffle=True,
        seed=42,
        shard_options=grain.ShardByJaxProcess(),  # Shard across processes
    ),
    operations=[
        Tokenize(tokenizer),
        FilterShort(min_length=10),           # Filter before batch
        grain.Batch(batch_size=32, drop_remainder=True),  # Consistent sizes
    ],
)

# ✅ Verify first batch
first_batch = next(iter(dataloader))
print(f"Batch shape: {first_batch['input_tokens'].shape}")
assert first_batch['input_tokens'].shape[0] == 32
```

### 6. Memory Management

**Monitor HBM Usage**:
```python
# Show HBM at key points
utils.show_hbm_usage("Before loading model")
model = load_model()
utils.show_hbm_usage("After loading model")

# During training
if step % 100 == 0:
  utils.show_hbm_usage(f"Step {step}")
```

**Reduce Memory**:
- Use FSDP for large models
- Enable gradient checkpointing
- Reduce batch size
- Use mixed precision (bfloat16)

**Detect Leaks**:
```python
# Track memory over time
import gc

for step in range(num_steps):
  loss = train_step(model, batch)
  
  if step % 100 == 0:
    gc.collect()  # Force GC
    utils.show_hbm_usage(f"Step {step}")
    # Memory should be stable
```

### 7. Profiling Strategy

**When to Profile**:
- After major code changes
- When investigating performance issues
- Before scaling to many hosts

**What to Profile**:
- Skip warmup steps (10-20)
- Profile 3-5 representative steps
- Capture both host and device traces

**Analysis**:
- Look for unexpected HBM usage
- Identify communication bottlenecks
- Check for inefficient ops

**Example**:
```python
# ✅ Good profiling setup
profiler_opts = profiler.ProfilerOptions(
    log_dir="/tmp/profiles",
    skip_first_n_steps=20,     # Skip warmup
    profiler_steps=5,          # Profile 5 steps
    host_tracer_level=2,       # Full traces
)

prof = profiler.Profiler(0, 1000, profiler_opts)

for step in range(1000):
  prof.maybe_activate(step)
  loss = train_step(model, batch)
  prof.maybe_deactivate(step)

# Analyze: tensorboard --logdir=/tmp/profiles
```

### 8. Common Pitfalls

**❌ Forgetting to shard input data**:
```python
# Wrong: Data not sharded
for batch in dataloader:
  loss = train_step(model, batch)  # May cause OOM

# ✅ Right: Shard data
for batch in dataloader:
  batch = sharding_utils.shard_input(batch, ("fsdp",))
  loss = train_step(model, batch)
```

**❌ Inconsistent mesh across processes**:
```python
# Wrong: Different mesh shapes per process
if jax.process_index() == 0:
  mesh = jax.sharding.Mesh(devices.reshape(8, 1), ("fsdp", "tp"))
else:
  mesh = jax.sharding.Mesh(devices.reshape(4, 2), ("fsdp", "tp"))

# ✅ Right: Same mesh shape all processes
mesh = jax.sharding.Mesh(
    jax.devices().reshape(8, 1),  # All processes same shape
    ("fsdp", "tp")
)
```

**❌ Not waiting for checkpoint save**:
```python
# Wrong: Race condition
if jax.process_index() == 0:
  ckpt_mgr.save(step, model)
# Non-main processes may try to load before save completes

# ✅ Right: Synchronize
if jax.process_index() == 0:
  ckpt_mgr.save(step, model)
jax.experimental.multihost_utils.sync_global_devices("ckpt_done")
```

**❌ Logging on all processes**:
```python
# Wrong: Duplicate logs from all processes
for step in range(num_steps):
  loss = train_step(model, batch)
  print(f"Step {step}, Loss: {loss}")  # Printed 4x on 4 hosts

# ✅ Right: Log only on main process
if jax.process_index() == 0 and step % 10 == 0:
  print(f"Step {step}, Loss: {loss}")
```
