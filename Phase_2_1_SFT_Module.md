---
markmap:
  initialExpandLevel: 2
---

# Phase 2.1: Supervised Fine-Tuning Module

**Module:** `tunix/sft/`  
**Core Class:** `PeftTrainer`  
**Purpose:** Foundation for all training in Tunix - handles parameter-efficient and full fine-tuning

---

## 1. Module Overview

### What is the SFT Module?

The **Supervised Fine-Tuning (SFT) module** is the **foundational training infrastructure** in Tunix. Every other trainer (DPO, Distillation, RL) extends or uses components from this module.

**Key Responsibilities:**
1. Training loop execution
2. Gradient computation and optimizer updates
3. Checkpoint management (save/restore)
4. Metrics logging and monitoring
5. Data sharding across devices
6. Performance profiling
7. Progress tracking

**Location:** `tunix/sft/`

```
tunix/sft/
├── peft_trainer.py          # Core trainer implementation
├── dpo/
│   └── dpo_trainer.py        # Direct Preference Optimization
├── checkpoint_manager.py     # Checkpoint management
├── metrics_logger.py         # Metrics logging infrastructure
├── sharding_utils.py         # Data/model sharding
├── profiler.py               # Performance profiling
├── progress_bar.py           # Training progress display
├── hooks.py                  # Training lifecycle hooks
├── inflight_throttler.py     # Async computation control
├── system_metrics_calculator.py  # System metrics
└── utils.py                  # Utility functions
```

---

## 2. PeftTrainer: The Foundation

### 2.1 Class Structure

```python
class PeftTrainer:
    """PEFT trainer for LoRA. Only LoRA parameters are updated."""
    
    # Core attributes
    model: nnx.Module                    # The model being trained
    optimizer: nnx.Optimizer             # Wraps optax optimizer
    config: TrainingConfig               # Training configuration
    
    # State tracking
    _train_steps: int                    # Number of optimizer updates
    _iter_steps: int                     # Number of loop iterations
    
    # Management components
    checkpoint_manager: CheckpointManager
    metrics_logger: MetricsLogger
    
    # Functions
    loss_fn: Callable                    # Training loss function
    eval_loss_fn: Callable               # Evaluation loss function
    gen_model_input_fn: Callable         # Input preprocessing
```

**Key Insight:** The trainer tracks TWO step counters:
- `_train_steps`: Actual model updates (after gradient accumulation)
- `_iter_steps`: Loop iterations (includes gradient accumulation steps)

### 2.2 Training Configuration

```python
@dataclasses.dataclass
class TrainingConfig:
    """Configuration for the trainer."""
    
    # Required
    eval_every_n_steps: int              # Evaluation frequency
    
    # Training control
    max_steps: int | None = None         # Maximum training steps
    gradient_accumulation_steps: int | None = None
    
    # Checkpointing
    checkpoint_root_directory: str | None = None
    checkpointing_options: ocp.CheckpointManagerOptions | None = None
    
    # Logging and monitoring
    metrics_logging_options: MetricsLoggerOptions | None = None
    profiler_options: ProfilerOptions | None = None
    
    # Distributed training
    data_sharding_axis: Tuple[str, ...] = ("fsdp",)
    
    # Performance
    max_inflight_computations: int = 2   # Async control
    
    # UI
    pbar_description: str | None = "Training"
    metrics_prefix: str = ""
```

**Configuration Highlights:**

1. **Gradient Accumulation:**
   ```python
   # If set, splits mini-batch into micro-batches
   gradient_accumulation_steps: int = 4
   
   # Effect:
   # - 4 forward/backward passes
   # - Gradients accumulated
   # - Single optimizer update
   # - Reduces memory usage
   ```

2. **Data Sharding:**
   ```python
   # Specifies how to shard data across devices
   data_sharding_axis: Tuple[str, ...] = ("fsdp",)
   
   # Common patterns:
   # ("fsdp",)           → Shard along FSDP axis
   # ("data",)           → Shard along data axis
   # ("fsdp", "tensor")  → Multiple sharding axes
   ```

3. **Inflight Computations:**
   ```python
   max_inflight_computations: int = 2
   
   # Allows async execution:
   # - Step N+1 starts before Step N finishes
   # - Hides communication latency
   # - Improves throughput
   ```

### 2.3 Initialization Process

```python
def __init__(
    self,
    model: nnx.Module,
    optimizer: optax.GradientTransformation,
    training_config: TrainingConfig,
    metrics_logger: Optional[MetricsLogger] = None,
    perf_tracer: Optional[perf_trace.Tracer] = None,
):
    # 1. Store model and config
    self.model = model
    self.config = training_config
    
    # 2. Detect LoRA mode
    self._lora_enabled = utils.is_lora_enabled(self.model)
    
    # 3. Wrap optimizer with gradient accumulation if needed
    if training_config.gradient_accumulation_steps is not None:
        optimizer = optax.MultiSteps(
            optimizer,
            training_config.gradient_accumulation_steps
        )
    
    # 4. Create NNX optimizer (only trains specified params)
    if self._lora_enabled:
        self.optimizer = nnx.Optimizer(
            self.model, 
            optimizer, 
            wrt=nnx.LoRAParam  # Only update LoRA parameters!
        )
    else:
        self.optimizer = nnx.Optimizer(
            self.model, 
            optimizer, 
            wrt=nnx.Param  # Update all parameters
        )
    
    # 5. Set default loss and input functions
    self.loss_fn = _default_loss_fn
    self.eval_loss_fn = _default_loss_fn
    self.gen_model_input_fn = lambda x: x
    
    # 6. Initialize management components
    self.checkpoint_manager = CheckpointManager(...)
    self.metrics_logger = metrics_logger or MetricsLogger(...)
    
    # 7. Initialize state
    self._train_steps = 0
    self._iter_steps = 0
    
    # 8. Restore from checkpoint if available
    restored_steps, metadata = self.checkpoint_manager.maybe_restore(
        self.model,
        restore_only_lora_params=self._lora_enabled
    )
    self._train_steps = restored_steps
    self._iter_steps = restored_steps * gradient_accumulation_steps
    
    # 9. Initialize profiler, progress bar, etc.
    self._prof = profiler.Profiler(...)
    self._pbar = None
```

**Initialization Flow:**

```
User creates trainer
      ↓
Detect LoRA mode
      ↓
Wrap optimizer with gradient accumulation
      ↓
Create NNX optimizer (wrt=LoRAParam or Param)
      ↓
Set up checkpoint manager
      ↓
Try to restore from checkpoint
      ↓
Initialize monitoring components
      ↓
Ready for training!
```

---

## 3. Training Loop Deep Dive

### 3.1 Main Training Method

```python
def train(
    self,
    train_ds: Iterable[Any],
    eval_ds: Iterable[Any] | None = None,
    skip_jit: bool = False,
    cache_nnx_graph: bool = False,
) -> None:
    """Training loop."""
    
    # 1. JIT compile train and eval steps
    train_step, eval_step = self.jit_train_and_eval_step(skip_jit)
    
    # 2. Cache partial functions for performance (optional)
    if cache_nnx_graph:
        partial_train_step = nnx.cached_partial(
            train_step, self.model, self.optimizer
        )
        partial_eval_step = nnx.cached_partial(eval_step, self.model)
    else:
        partial_train_step = lambda inputs: train_step(
            self.model, self.optimizer, inputs
        )
        partial_eval_step = lambda inputs: eval_step(self.model, inputs)
    
    # 3. Initial evaluation if eval_ds provided
    if eval_ds:
        self._run_eval(eval_ds, partial_eval_step)
    
    # 4. Setup progress bar
    if self.config.max_steps is not None:
        self._pbar = ProgressBar(...)
    
    # 5. Call training start hook
    if self.training_hooks:
        self.training_hooks.on_train_start(self)
    
    # 6. Main training loop
    train_iterator = iter(train_ds)
    index = 0
    last_step_completion_time = time.perf_counter()
    
    while True:
        # 6a. Activate profiler if needed
        self._prof.maybe_activate(self._iter_steps)
        
        # 6b. Get next training example
        train_example = next(train_iterator, None)
        if train_example is None:
            break
        
        # 6c. Check if max_steps reached
        if self.config.max_steps and self._train_steps >= self.config.max_steps:
            break
        
        # 6d. Prepare and shard inputs
        train_example = self._prepare_inputs(train_example)
        train_example = sharding_utils.shard_input(
            train_example,
            self.config.data_sharding_axis
        )
        
        # 6e. Measure TFLOPs (first step only)
        if not self._flops_measured and not skip_jit:
            tflops_per_step = measure_tflops_per_step(...)
            self.metrics_logger.log("tflops_per_step", ...)
            self._flops_measured = True
        
        # 6f. Wait for previous computation if needed (throttling)
        self._throttler.wait_for_next()
        
        # 6g. Training step hook
        if self.training_hooks:
            self.training_hooks.on_train_step_start(self)
        
        # 6h. EXECUTE TRAINING STEP
        with self._perf_tracer.span("peft_train_step", ...) as span:
            train_loss, aux = partial_train_step(train_example)
            span.device_end([train_loss])
        
        # 6i. Track step time
        current_time = time.perf_counter()
        step_time_delta = current_time - last_step_completion_time
        last_step_completion_time = current_time
        
        # 6j. Add async computation to throttler
        self._throttler.add_computation(train_loss)
        
        # 6k. Buffer metrics
        self._buffered_train_metrics = self._buffer_metrics(
            self._buffered_train_metrics,
            loss=train_loss,
            step=self._train_steps,
            step_time_delta=step_time_delta,
        )
        
        # 6l. Post-process auxiliary data
        self._post_process_train_step(aux)
        
        # 6m. Increment iteration step
        self._iter_steps += 1
        
        # 6n. Check if it's time for optimizer update
        if self._iter_steps % gradient_accumulation_steps == 0:
            self._train_steps += 1
            
            # Write metrics
            self._write_train_metrics()
            
            # Save checkpoint (respects checkpointing policy)
            self.checkpoint_manager.save(
                self._train_steps,
                self.model,
                save_only_lora_params=self._lora_enabled,
            )
            
            # Run evaluation if needed
            if eval_ds and self._train_steps % eval_every_n_steps == 0:
                self._run_eval(eval_ds, partial_eval_step)
        
        # 6o. Deactivate profiler if needed
        self._prof.maybe_deactivate(self._iter_steps)
    
    # 7. Wait for all async computations to finish
    self._throttler.wait_for_all()
    
    # 8. Training end hook
    if self.training_hooks:
        self.training_hooks.on_train_end(self)
    
    # 9. Cleanup
    if not self.is_managed_externally:
        self.close()
```

**Training Loop Flow:**

```
Initialize
    ↓
JIT compile train/eval steps
    ↓
Run initial eval (optional)
    ↓
┌─────────────────┐
│  For each batch │
└────────┬────────┘
         │
    Get batch
         ↓
    Prepare & shard
         ↓
    Train step (forward + backward)
         ↓
    Buffer metrics
         ↓
    Increment iter_steps
         ↓
    ┌─────────────────────────┐
    │ Gradient accumulated?   │
    └────┬──────────┬─────────┘
         NO        YES
         │          │
         │     Increment train_steps
         │          ↓
         │     Log metrics
         │          ↓
         │     Save checkpoint
         │          ↓
         │     Run eval (if scheduled)
         │          │
         └──────────┘
              ↓
         Repeat or end
```

### 3.2 Train Step Execution

```python
def _train_step(
    self,
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    inputs: Any
) -> ArrayLike | Tuple[ArrayLike, Any]:
    """Main body for one train step."""
    
    # 1. Generate model inputs
    inputs = self.gen_model_input_fn(inputs)
    
    # 2. Define gradient function
    grad_fn = nnx.value_and_grad(
        self.loss_fn,
        argnums=nnx.DiffState(0, nnx.LoRAParam) if self._lora_enabled else 0,
        has_aux=self._has_aux,
    )
    
    # 3. Compute loss and gradients
    out, grads = grad_fn(model, **inputs)
    
    # 4. Update model parameters
    optimizer.update(model, grads)
    
    # 5. Return loss (and aux if available)
    if self._has_aux:
        loss, aux = out
        return loss, aux
    else:
        return out, None
```

**What Happens Under the Hood:**

```
inputs → gen_model_input_fn → processed_inputs
                                    ↓
                          Forward pass through model
                                    ↓
                            Compute loss_fn
                                    ↓
                          ┌─────────┴─────────┐
                          │                   │
                       Loss                Aux data
                          │                   
                          ↓
                    Automatic differentiation (JAX grad)
                          ↓
                      Gradients
                          ↓
        ┌─────────────────┴──────────────────┐
        │  LoRA mode?                         │
        ├─────YES────────┬────────NO─────────┤
        │                │                    │
  Gradient of      Gradient of         Gradient of
  LoRA params      LoRA params         all params
  only             (ignored)           
        │                                     │
        └─────────────────┬───────────────────┘
                          ↓
                   optimizer.update()
                          ↓
                  Parameters updated!
```

### 3.3 Default Loss Function

```python
def _default_loss_fn(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> ArrayLike:
    """Default loss function for PEFT training."""
    
    # 1. Forward pass
    logits, _ = model(input_tokens, positions, None, attention_mask)
    
    # 2. Shift logits and targets for next-token prediction
    logits = logits[:, :-1, :]        # Exclude last token
    target_tokens = input_tokens[:, 1:]  # Exclude first token
    target_mask = input_mask[:, 1:]      # Exclude first token
    
    # 3. Convert targets to one-hot
    one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])
    
    # 4. Mask unwanted tokens (padding, etc.)
    one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]
    
    # 5. Compute normalization factor
    norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)
    
    # 6. Compute negative log likelihood (NLL) loss
    loss = -jnp.sum(jax.nn.log_softmax(logits) * one_hot) * norm_factor
    
    return loss
```

**Loss Computation Breakdown:**

```
Input: "The cat sat on the mat"
Tokens: [1, 2, 3, 4, 5, 6, 7]

Next-token prediction task:
Input:  [1, 2, 3, 4, 5, 6]
Target: [2, 3, 4, 5, 6, 7]

For each position:
- Predict probability distribution over vocabulary
- Compute cross-entropy with target token
- Sum cross-entropy losses (weighted by mask)
- Normalize by number of valid tokens

Result: Average negative log-likelihood
```

---

## 4. Checkpoint Management

### 4.1 CheckpointManager Class

**Location:** `tunix/sft/checkpoint_manager.py`

```python
class CheckpointManager:
    """Checkpoint manager for PEFT."""
    
    def __init__(
        self,
        root_directory: str | None = None,
        options: ocp.CheckpointManagerOptions | None = None,
    ):
        if root_directory is None:
            # Checkpointing disabled
            self._checkpoint_manager = None
            return
        
        # Detect Pathways environment
        if 'proxy' in os.getenv('JAX_PLATFORMS', ''):
            # Use persistence APIs for Pathways
            item_handlers = {
                'model_params': ocp.PyTreeCheckpointHandler(
                    use_ocdbt=False,
                    use_zarr3=False,
                ),
            }
        else:
            # Standard checkpointing
            item_handlers = {
                'model_params': ocp.PyTreeCheckpointHandler(),
            }
        
        item_handlers['custom_metadata'] = ocp.JsonCheckpointHandler()
        
        # Create Orbax checkpoint manager
        self._checkpoint_manager = ocp.CheckpointManager(
            root_directory,
            item_handlers=item_handlers,
            options=options or _DEFAULT_CHECKPOINTING_OPTIONS,
        )
```

### 4.2 Saving Checkpoints

```python
def save(
    self,
    step: int,
    model: nnx.Module,
    save_only_lora_params: bool = False,
    force: bool = False,
    custom_metadata: dict[str, Any] | None = None,
) -> bool:
    """Saves the params for the given step."""
    
    if self._checkpoint_manager is None:
        return False
    
    # Check save policy (unless forced)
    if not force and not self._checkpoint_manager.should_save(step):
        return False
    
    # Extract parameters to save
    if save_only_lora_params:
        params = nnx.state(model, nnx.LoRAParam)  # Only LoRA params
    else:
        params = nnx.state(model)  # All parameters
    
    # Prepare checkpoint args
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

### 4.3 Restoring Checkpoints

```python
def maybe_restore(
    self,
    model: nnx.Module,
    step: int | None = None,
    restore_only_lora_params: bool = False,
) -> Tuple[int, dict[str, Any]]:
    """Restores params from latest checkpoint if available."""
    
    if self._checkpoint_manager is None:
        return 0, {}
    
    # Find latest checkpoint if step not specified
    if step is None:
        step = self._checkpoint_manager.latest_step()
        if step is None:
            return 0, {}  # No checkpoint available
    
    # Get abstract parameter structure
    if restore_only_lora_params:
        abstract_params = nnx.state(model, nnx.LoRAParam)
    else:
        abstract_params = nnx.state(model)
    
    # Create restore args with sharding info
    def map_to_pspec(data):
        return ocp.type_handlers.ArrayRestoreArgs(
            sharding=data.sharding
        )
    
    restore_args_dict = jax.tree_util.tree_map(
        map_to_pspec,
        abstract_params
    )
    
    checkpoint_args = ocp.args.PyTreeRestore(
        item=abstract_params,
        restore_args=restore_args_dict
    )
    
    # Restore checkpoint
    ckpt = self._checkpoint_manager.restore(
        step,
        args=ocp.args.Composite(model_params=checkpoint_args),
    )
    
    # Update model with restored parameters
    nnx.update(model, ckpt.model_params)
    
    # Get custom metadata
    metadata = self._checkpoint_manager.metadata(step)
    custom_metadata = metadata.custom_metadata if metadata else {}
    
    return step, custom_metadata
```

### 4.4 Default Checkpointing Policy

```python
_DEFAULT_CHECKPOINTING_OPTIONS = ocp.CheckpointManagerOptions(
    save_decision_policy=ocp.checkpoint_managers.ContinuousCheckpointingPolicy(
        minimum_interval_secs=180,  # Save at most every 3 minutes
    ),
    max_to_keep=3,  # Keep only 3 most recent checkpoints
)
```

**Checkpoint Structure:**

```
checkpoint_root_directory/
├── 0/                          # Step 0 checkpoint
│   ├── model_params/
│   │   ├── layers/
│   │   └── ...
│   └── custom_metadata.json
├── 1000/                       # Step 1000 checkpoint
│   ├── model_params/
│   └── custom_metadata.json
└── 2000/                       # Step 2000 checkpoint
    ├── model_params/
    └── custom_metadata.json
```

**Checkpointing Flow:**

```
Training step N
      ↓
Should save? (check policy)
      ↓
Extract parameters
   ├─ Full mode: All params
   └─ LoRA mode: Only LoRA params
      ↓
Serialize to Orbax format
      ↓
Write to disk
      ↓
Cleanup old checkpoints (keep max_to_keep)
```

---

## 5. Metrics Logging

### 5.1 MetricsLogger Class

**Location:** `tunix/sft/metrics_logger.py`

```python
class MetricsLogger:
    """Simple Metrics logger with unified, protocol-based backend system."""
    
    def __init__(
        self,
        metrics_logger_options: MetricsLoggerOptions | None = None,
    ):
        self._metrics = {}  # Local history
        
        # Create backends (only on main process)
        self._backends = (
            metrics_logger_options.create_backends()
            if metrics_logger_options
            else []
        )
        
        # Register with JAX monitoring
        if metrics_logger_options and jax.process_index() == 0:
            for backend in self._backends:
                jax.monitoring.register_scalar_listener(
                    backend.log_scalar
                )
```

### 5.2 Logging Backends

**Protocol-Based Design:**

```python
# All backends must implement this protocol
class LoggingBackend(Protocol):
    def log_scalar(
        self,
        event: str,
        value: float | np.ndarray,
        **kwargs
    ):
        """Logs a scalar value."""
        ...
    
    def close(self):
        """Closes the logger."""
        ...
```

**Available Backends:**

1. **TensorBoard Backend**
   ```python
   TensorboardBackend(
       log_dir="/path/to/logs",
       flush_every_n_steps=100
   )
   ```

2. **Weights & Biases Backend**
   ```python
   WandbBackend(
       project="my-project",
       name="experiment-1"
   )
   ```

3. **CLU Backend** (Google internal)
   ```python
   CluBackend(log_dir="/path/to/logs")
   ```

4. **Custom Backend**
   ```python
   class MyCustomBackend:
       def log_scalar(self, event, value, **kwargs):
           # Custom logging logic
           print(f"{event}: {value}")
       
       def close(self):
           pass
   ```

### 5.3 Metrics Logging Configuration

```python
@dataclasses.dataclass
class MetricsLoggerOptions:
    """Metrics Logger options."""
    
    log_dir: str
    project_name: str = "tunix"
    run_name: str = ""
    flush_every_n_steps: int = 100
    backend_factories: list[BackendFactory] | None = None
    
    def create_backends(self) -> list[LoggingBackend]:
        """Factory method to create backends."""
        
        # Only create on main process
        if jax.process_index() != 0:
            return []
        
        # Case 1: User-provided backends
        if self.backend_factories is not None:
            return [factory() for factory in self.backend_factories]
        
        # Case 2: Default backends
        active_backends = []
        
        if env_utils.is_internal_env():
            # Internal: Use CLU
            active_backends.append(CluBackend(log_dir=self.log_dir))
        else:
            # External: Use TensorBoard + WandB
            active_backends.append(
                TensorboardBackend(
                    log_dir=self.log_dir,
                    flush_every_n_steps=self.flush_every_n_steps,
                )
            )
            try:
                active_backends.append(
                    WandbBackend(
                        project=self.project_name,
                        name=self.run_name
                    )
                )
            except ImportError:
                logging.info("WandB not available, skipping.")
        
        return active_backends
```

### 5.4 Logging Metrics

```python
def log(
    self,
    metrics_prefix: str,
    metric_name: str,
    scalar_value: float | np.ndarray,
    mode: Mode | str,  # "train" or "eval"
    step: int,
):
    """Logs the scalar metric value."""
    
    # 1. Store in local history
    prefix_metrics = self._metrics.setdefault(metrics_prefix, {})
    mode_metrics = prefix_metrics.setdefault(
        mode,
        collections.defaultdict(list)
    )
    mode_metrics[metric_name].append(scalar_value)
    
    # 2. Broadcast to all backends via JAX monitoring
    jax.monitoring.record_scalar(
        f"{metrics_prefix}/{mode}/{metric_name}",
        scalar_value,
        step=step
    )
```

**Metric Flow:**

```
trainer.metrics_logger.log(
    metrics_prefix="experiment1",
    metric_name="loss",
    scalar_value=0.5,
    mode="train",
    step=100
)
      ↓
Store in local history
      ↓
jax.monitoring.record_scalar()
      ↓
┌─────────────┴────────────┐
│                          │
TensorBoard            WandB
Backend               Backend
  ↓                      ↓
events.out          wandb.log()
.tfevents
```

### 5.5 Built-in Metrics

**Automatically Logged by PeftTrainer:**

```python
def _log_metrics(
    self,
    loss: ArrayLike,
    step: int | None = None,
    step_time_delta: float | None = None,
    additional_metrics: dict[str, ArrayLike] | None = None,
):
    """Logs the metrics to the metrics logger."""
    
    # 1. Loss
    self.metrics_logger.log(
        self.metrics_prefix, "loss", loss, self._mode, step
    )
    
    # 2. Perplexity (exp(loss))
    perplexity = np.exp(jax.device_get(loss))
    self.metrics_logger.log(
        self.metrics_prefix, "perplexity", perplexity, self._mode, step
    )
    
    # 3. Learning rate (if available)
    learning_rate = self._try_get_learning_rate()
    if learning_rate is not None:
        self.metrics_logger.log(
            self.metrics_prefix,
            "learning_rate",
            jax.device_get(learning_rate),
            self._mode,
            step,
        )
    
    # 4. Step time and throughput
    if step_time_delta is not None:
        self.metrics_logger.log(
            self.metrics_prefix,
            "step_time_sec",
            step_time_delta,
            self._mode,
            step,
        )
        self.metrics_logger.log(
            self.metrics_prefix,
            "steps_per_sec",
            1.0 / (step_time_delta + 1e-9),
            self._mode,
            step,
        )
    
    # 5. Additional metrics
    for k, v in (additional_metrics or {}).items():
        self.metrics_logger.log(
            self.metrics_prefix, k, v, self._mode, step
        )
```

**Default Metrics:**

| Metric | Description | Formula |
|--------|-------------|---------|
| `loss` | Training/eval loss | Cross-entropy loss |
| `perplexity` | Model uncertainty | `exp(loss)` |
| `learning_rate` | Current LR | From optimizer state |
| `step_time_sec` | Time per step | Measured |
| `steps_per_sec` | Throughput | `1 / step_time_sec` |
| `tflops_per_step` | Compute utilization | Measured once |

---

## 6. Data Sharding

### 6.1 Sharding Utility

**Location:** `tunix/sft/sharding_utils.py`

```python
def shard_input(
    input_data: jax.Array,
    data_sharding_axis: Tuple[str, ...]
) -> jax.Array:
    """Shards the input data across the available devices."""
    
    # Get current mesh
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty:
        return input_data  # No sharding if no mesh
    
    # Create partition spec
    pspec = shd.PartitionSpec(*data_sharding_axis)
    
    # Check if already sharded correctly
    is_sharded = jax.tree.map(
        lambda x: (
            isinstance(x, jax.Array) and
            hasattr(x, "sharding") and
            x.sharding.mesh == mesh and
            x.sharding.spec == pspec
        ),
        input_data,
    )
    if all(jax.tree.leaves(is_sharded)):
        return input_data  # Already sharded correctly
    
    # Shard the data
    with jax.transfer_guard("allow"):
        return jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(
                get_sharding(x, mesh=mesh, pspec=pspec),
                x
            ),
            input_data,
        )
```

### 6.2 Sharding Strategy

```python
def get_sharding(
    x: jax.Array,
    mesh: shd.Mesh,
    pspec: shd.PartitionSpec
):
    """Get a sharding for a tensor given a mesh and partition spec."""
    
    # Only shard arrays with rank > 0
    if not isinstance(x, (np.ndarray, jax.Array)) or x.ndim == 0:
        return shd.NamedSharding(mesh, shd.PartitionSpec())
    
    # Don't shard if rank is insufficient
    if x.ndim < len(pspec):
        return shd.NamedSharding(mesh, shd.PartitionSpec())
    
    # Check divisibility for all sharded axes
    for i, axis_name in enumerate(pspec):
        if axis_name is not None:
            axis_names = (
                axis_name if isinstance(axis_name, tuple)
                else (axis_name,)
            )
            for name in axis_names:
                axis_size = mesh.shape[name]
                if x.shape[i] % axis_size != 0:
                    # Not evenly divisible → replicate
                    return shd.NamedSharding(
                        mesh,
                        shd.PartitionSpec()
                    )
    
    # All checks passed → shard
    return shd.NamedSharding(mesh, pspec)
```

**Sharding Examples:**

```python
# Example 1: FSDP sharding
mesh = Mesh(devices, axis_names=('fsdp',))
batch = jnp.ones((32, 128, 768))  # [batch, seq_len, hidden]

sharded = shard_input(
    batch,
    data_sharding_axis=("fsdp",)
)
# Result: Batch dimension sharded across 'fsdp' axis

# Example 2: No sharding (replicated)
sharded = shard_input(
    batch,
    data_sharding_axis=()
)
# Result: Replicated across all devices

# Example 3: Multiple axes
mesh = Mesh(
    devices.reshape(4, 2),
    axis_names=('data', 'model')
)
sharded = shard_input(
    batch,
    data_sharding_axis=("data",)
)
# Result: Batch sharded across 'data' axis,
#         replicated across 'model' axis
```

---

## 7. DPO Trainer

### 7.1 Direct Preference Optimization

**Location:** `tunix/sft/dpo/dpo_trainer.py`

**What is DPO?**
- Preference tuning method for aligning LLMs
- More efficient alternative to RLHF
- No reward model needed
- No text generation in training loop
- Uses preference pairs: (prompt, chosen, rejected)

```python
class DPOTrainer(peft_trainer.PeftTrainer):
    """Direct Preference Optimization trainer."""
    
    def __init__(
        self,
        model: nnx.Module,              # Policy model to train
        ref_model: nnx.Module | None,   # Reference model (frozen)
        optimizer: optax.GradientTransformation,
        training_config: DPOTrainingConfig,
        tokenizer: Any | None = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.dpo_config = training_config
        self.algorithm = training_config.algorithm  # "dpo" or "orpo"
        
        # Call parent constructor
        super().__init__(model, optimizer, training_config)
        
        # Set tokenizer
        self.tokenizer = (
            None if tokenizer is None
            else tokenizer_adapter.TokenizerAdapter(tokenizer)
        )
        
        # Set DPO loss function
        self.with_loss_fn(dpo_loss_fn, has_aux=True)
        
        # Configure input generation based on algorithm
        if self.algorithm == "orpo":
            self.with_gen_model_input_fn(
                lambda x: {
                    "train_example": x,
                    "algorithm": "orpo",
                    "lambda_orpo": self.dpo_config.lambda_orpo,
                    "label_smoothing": self.dpo_config.label_smoothing,
                }
            )
        else:  # DPO
            self.with_gen_model_input_fn(
                lambda x: {
                    "train_example": x,
                    "algorithm": "dpo",
                    "beta": self.dpo_config.beta,
                    "label_smoothing": self.dpo_config.label_smoothing,
                }
            )
```

### 7.2 DPO Training Config

```python
@dataclasses.dataclass
class DPOTrainingConfig(peft_trainer.TrainingConfig):
    """DPO/ORPO Training Config."""
    
    algorithm: str = "dpo"  # "dpo" or "orpo"
    
    # DPO hyperparameters
    beta: float = 0.1       # KL penalty coefficient
    
    # ORPO hyperparameters
    lambda_orpo: float = 0.1  # Preference loss weight
    
    # Common
    label_smoothing: float = 0.0
    
    # Tokenization (if using string inputs)
    max_prompt_length: int | None = None
    max_response_length: int | None = None
```

### 7.3 DPO Input Formats

**Option 1: Raw Strings (DataInput)**

```python
@flax.struct.dataclass
class DataInput:
    """Training data input for DPO (raw strings)."""
    
    prompts: list[str]               # ["Write a poem"]
    chosen_responses: list[str]      # ["Roses are red..."]
    rejected_responses: list[str]    # ["asdf jkl..."]
```

**Option 2: Tokenized (TrainingInput)**

```python
@flax.struct.dataclass
class TrainingInput:
    """Tokenized training input for DPO."""
    
    # Prompts (left-padded)
    prompt_ids: jax.Array
    prompt_mask: jax.Array
    
    # Chosen responses (right-padded)
    chosen_ids: jax.Array
    chosen_mask: jax.Array
    
    # Rejected responses (right-padded)
    rejected_ids: jax.Array
    rejected_mask: jax.Array
```

### 7.4 DPO Loss Function

**Conceptual Formula:**

```
DPO Loss = -log(σ(β * (log π_θ(y_w | x) - log π_ref(y_w | x)
                       - log π_θ(y_l | x) + log π_ref(y_l | x))))

Where:
- π_θ: Policy model (being trained)
- π_ref: Reference model (frozen)
- y_w: Chosen (winning) response
- y_l: Rejected (losing) response
- x: Prompt
- β: KL penalty coefficient
- σ: Sigmoid function
```

**Intuition:**
- Increase probability of chosen responses
- Decrease probability of rejected responses
- Regularize with KL divergence from reference model
- Binary classification-style loss

### 7.5 ORPO Trainer

**ORPO (Odds Ratio Preference Optimization):**
- Memory-efficient variant of DPO
- No reference model needed (~50% memory savings)
- Combines SFT loss with preference loss

```python
class ORPOTrainer(DPOTrainer):
    """ORPO trainer (alias for DPOTrainer with algorithm='orpo')."""
    
    def __init__(
        self,
        model: nnx.Module,
        optimizer: optax.GradientTransformation,
        training_config: ORPOTrainingConfig,
        tokenizer: Any | None = None,
    ):
        # Note: No ref_model needed!
        super().__init__(
            model=model,
            ref_model=None,  # ORPO doesn't use reference model
            optimizer=optimizer,
            training_config=training_config,
            tokenizer=tokenizer,
        )
```

**ORPO Loss:**

```
ORPO Loss = NLL Loss + λ * Preference Loss

Where:
- NLL Loss: Standard next-token prediction loss
- Preference Loss: Based on odds ratios
- λ: Weight for preference loss (lambda_orpo)
```

---

## 8. Performance Optimization

### 8.1 JIT Compilation

```python
def jit_train_and_eval_step(
    self,
    skip_jit: bool = False
):
    """Creates and returns JIT-compiled train and eval step functions."""
    
    train_step = self.create_train_step_fn()
    eval_step = self.create_eval_step_fn()
    
    if skip_jit:
        return train_step, eval_step
    
    # Check cache
    if self._jitted_train_step_fn is None:
        # Shard optimizer state before JIT
        self._shard_optimizer(pxla.thread_resources.env.physical_mesh)
        
        # JIT compile with buffer donation
        self._jitted_train_step_fn = nnx.jit(
            train_step,
            donate_argnames=("optimizer",)  # Reuse optimizer buffer
        )
        self._jitted_eval_step_fn = nnx.jit(eval_step)
    
    return self._jitted_train_step_fn, self._jitted_eval_step_fn
```

**Buffer Donation:**
```python
# Without donation:
optimizer_old = optimizer_state
optimizer_new = update(optimizer_old)
# Both optimizer_old and optimizer_new in memory

# With donation:
optimizer = optimizer_state
optimizer = update(optimizer)  # Reuses buffer!
# Only one copy in memory
```

### 8.2 Gradient Accumulation

```python
# Configuration
TrainingConfig(
    gradient_accumulation_steps=4
)

# Effect:
# Before: 1 batch → 1 forward → 1 backward → 1 update
# After:  4 batches → 4 forward → 4 backward → 1 update

# Memory savings:
# - Only need activations for 1 micro-batch
# - Gradients accumulated across micro-batches
# - Single optimizer update

# Implementation (via optax.MultiSteps):
if gradient_accumulation_steps is not None:
    optimizer = optax.MultiSteps(
        optimizer,
        gradient_accumulation_steps
    )
```

### 8.3 Inflight Throttler

```python
class InflightThrottler:
    """Controls how many computations can be scheduled ahead."""
    
    def __init__(self, max_inflight: int = 2):
        self.max_inflight = max_inflight
        self.futures = []
    
    def wait_for_next(self):
        """Wait if too many computations in flight."""
        while len(self.futures) >= self.max_inflight:
            # Block until oldest computation finishes
            self.futures.pop(0).result()
    
    def add_computation(self, future):
        """Add a new computation."""
        self.futures.append(future)
    
    def wait_for_all(self):
        """Wait for all computations to finish."""
        for future in self.futures:
            future.result()
        self.futures.clear()
```

**Benefits:**
- Hides communication latency
- Overlaps computation and data transfer
- Improves throughput without increasing memory

### 8.4 Profiling

**Location:** `tunix/sft/profiler.py`

```python
@dataclasses.dataclass
class ProfilerOptions:
    log_dir: str                    # Where to save profiles
    skip_first_n_steps: int         # Warmup steps
    profiler_steps: int             # Steps to profile
    set_profile_options: bool = True
    host_tracer_level: int = 2      # Capture HBM profiles
    python_tracer_level: int = 1

class Profiler:
    """Activate/deactivate profiler based on ProfilerOptions."""
    
    def maybe_activate(self, step: int):
        """Start profiler at configured step."""
        if step == self._first_profile_step:
            jax.profiler.start_trace(
                log_dir=self._output_path,
                profiler_options=profile_options
            )
    
    def maybe_deactivate(self, step: int):
        """Stop profiler after configured steps."""
        if step == self._last_profile_step:
            jax.profiler.stop_trace()
```

**Usage:**

```python
config = TrainingConfig(
    profiler_options=ProfilerOptions(
        log_dir="/tmp/profiles",
        skip_first_n_steps=10,  # Skip compilation steps
        profiler_steps=10,      # Profile 10 steps
    )
)

# Profiler automatically activates/deactivates
# View profiles with: tensorboard --logdir /tmp/profiles
```

---

## 9. Training Hooks

### 9.1 Hook System

**Location:** `tunix/sft/hooks.py`

```python
class TrainingHooks:
    """Hooks for training lifecycle events."""
    
    def on_train_start(self, trainer):
        """Called once at the start of training."""
        pass
    
    def on_train_step_start(self, trainer):
        """Called before each training step."""
        pass
    
    def on_train_step_end(
        self,
        trainer,
        step: int,
        loss: float,
        step_time: float
    ):
        """Called after each training step."""
        pass
    
    def on_eval_step_start(self, trainer):
        """Called before each eval step."""
        pass
    
    def on_eval_step_end(self, trainer, eval_loss: float):
        """Called after evaluation."""
        pass
    
    def on_train_end(self, trainer):
        """Called once at the end of training."""
        pass

class DataHooks:
    """Hooks for custom data loading."""
    
    def load_next_train_batch(self, trainer):
        """Custom training batch loading."""
        pass
    
    def load_next_eval_batch(self, trainer):
        """Custom eval batch loading."""
        pass
```

### 9.2 Using Hooks

```python
class MyCustomHooks(TrainingHooks):
    def on_train_start(self, trainer):
        print(f"Starting training from step {trainer.train_steps}")
    
    def on_train_step_end(self, trainer, step, loss, step_time):
        if step % 100 == 0:
            print(f"Step {step}: loss={loss:.4f}")
    
    def on_train_end(self, trainer):
        print(f"Training complete! Final step: {trainer.train_steps}")

# Use hooks
trainer = PeftTrainer(...)
trainer.with_training_hooks(MyCustomHooks())
trainer.train(train_ds)
```

---

## 10. Complete Training Example

### 10.1 Full Fine-Tuning Example

```python
import jax
from flax import nnx
import optax
from tunix import PeftTrainer, TrainingConfig, MetricsLoggerOptions

# 1. Load model
model = load_my_model()  # Your model loading logic

# 2. Create optimizer
learning_rate = 1e-4
optimizer = optax.adam(learning_rate)

# 3. Configure training
config = TrainingConfig(
    max_steps=1000,
    eval_every_n_steps=100,
    gradient_accumulation_steps=4,
    checkpoint_root_directory="/tmp/checkpoints",
    metrics_logging_options=MetricsLoggerOptions(
        log_dir="/tmp/logs",
        project_name="my-experiment",
        run_name="full-finetune",
    ),
    data_sharding_axis=("fsdp",),
)

# 4. Create trainer
trainer = PeftTrainer(
    model=model,
    optimizer=optimizer,
    training_config=config,
)

# 5. Prepare datasets
train_ds = create_training_dataset()
eval_ds = create_eval_dataset()

# 6. Train!
trainer.train(train_ds, eval_ds)
```

### 10.2 LoRA/QLoRA Example

```python
from tunix.models import automodel

# 1. Load model with LoRA
model = automodel.load_model_with_lora(
    model_path="google/gemma-2b",
    lora_rank=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
)

# 2. Create optimizer (same as before)
optimizer = optax.adam(1e-4)

# 3. Configure training
config = TrainingConfig(
    max_steps=1000,
    eval_every_n_steps=100,
    checkpoint_root_directory="/tmp/lora_checkpoints",
)

# 4. Create trainer (automatically detects LoRA)
trainer = PeftTrainer(model, optimizer, config)

# 5. Train (only LoRA params will be updated!)
trainer.train(train_ds, eval_ds)

# Note: Checkpoints will only save LoRA parameters
# Much smaller checkpoint files!
```

### 10.3 DPO Example

```python
from tunix import DPOTrainer, DPOTrainingConfig
from tunix.sft.dpo.dpo_trainer import DataInput

# 1. Load policy and reference models
policy_model = load_model("google/gemma-2b")
ref_model = load_model("google/gemma-2b")  # Same architecture

# 2. Configure DPO training
config = DPOTrainingConfig(
    max_steps=500,
    eval_every_n_steps=50,
    beta=0.1,  # KL penalty
    algorithm="dpo",
    max_prompt_length=128,
    max_response_length=256,
)

# 3. Load tokenizer
tokenizer = load_tokenizer("google/gemma-2b")

# 4. Create DPO trainer
trainer = DPOTrainer(
    model=policy_model,
    ref_model=ref_model,
    optimizer=optax.adam(5e-5),
    training_config=config,
    tokenizer=tokenizer,
)

# 5. Prepare preference dataset
train_ds = [
    DataInput(
        prompts=["Write a poem about AI"],
        chosen_responses=["AI so bright and wise..."],
        rejected_responses=["asdf jkl qwerty..."],
    ),
    # ... more examples
]

# 6. Train!
trainer.train(train_ds)
```

---

## 11. Common Patterns and Best Practices

### 11.1 Custom Loss Function

```python
def my_custom_loss(
    model: nnx.Module,
    input_tokens: jax.Array,
    input_mask: jax.Array,
    positions: jax.Array,
    attention_mask: jax.Array,
) -> jax.Array:
    """Custom loss function."""
    
    # Forward pass
    logits, _ = model(input_tokens, positions, None, attention_mask)
    
    # Your custom loss computation
    # ... (implement your logic)
    
    return loss

# Use custom loss
trainer = PeftTrainer(model, optimizer, config)
trainer.with_loss_fn(my_custom_loss)
trainer.train(train_ds)
```

### 11.2 Custom Input Preprocessing

```python
def preprocess_inputs(training_input):
    """Custom input preprocessing."""
    
    # Extract fields
    text = training_input['text']
    
    # Tokenize
    tokens = tokenizer(text)
    
    # Build positions and masks
    positions = build_positions(tokens)
    attention_mask = make_causal_mask(tokens)
    
    return {
        'input_tokens': tokens,
        'input_mask': mask,
        'positions': positions,
        'attention_mask': attention_mask,
    }

# Use custom preprocessing
trainer.with_gen_model_input_fn(preprocess_inputs)
```

### 11.3 Resume Training

```python
# Automatic resume from checkpoint
config = TrainingConfig(
    checkpoint_root_directory="/tmp/checkpoints",
    # ... other config
)

trainer = PeftTrainer(model, optimizer, config)

# Automatically restores from latest checkpoint
# trainer._train_steps set to restored step
# Training continues from where it left off

trainer.train(train_ds)
```

### 11.4 Multi-Host Training

```python
# 1. Initialize JAX distributed
jax.distributed.initialize()

# 2. Create mesh spanning all hosts
devices = jax.devices()  # All devices across all hosts
mesh = jax.sharding.Mesh(
    devices.reshape(num_hosts, chips_per_host),
    axis_names=('data', 'model')
)

# 3. Set mesh context
with mesh:
    # 4. Create trainer (mesh automatically detected)
    trainer = PeftTrainer(model, optimizer, config)
    
    # 5. Train (automatic synchronization)
    trainer.train(train_ds)

# Checkpointing only happens on process 0
# Metrics logging only happens on process 0
```

---

## 🎯 Phase 2.1 Checklist

- [ ] Understand PeftTrainer as foundation of all training
- [ ] Know difference between `train_steps` and `iter_steps`
- [ ] Grasp training loop execution flow
- [ ] Understand checkpoint save/restore mechanism
- [ ] Know metrics logging system and backends
- [ ] Understand data sharding strategy
- [ ] Know how gradient accumulation works
- [ ] Familiar with DPO and ORPO trainers
- [ ] Can customize loss functions and preprocessing
- [ ] Ready to explore RL module (Phase 2.2)

---

**Previous:** [Phase 1.3 - Key Technologies](Phase_1_3_Key_Technologies.md)  
**Next:** Phase 2.2 - RL Module (Coming next!)
