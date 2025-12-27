---
markmap:
  initialExpandLevel: 2
---


# Phase 2.2: RL Module Deep Dive

## 1. Module Overview
### Purpose and Scope
- Reinforcement Learning training infrastructure
- Actor-Critic architectures (PPO, GRPO)
- Disaggregated multi-model setup
- Policy optimization with rewards

### Key Files Structure
- `rl_cluster.py` (1004 lines) - Central cluster orchestrator
- `rl_learner.py` (780 lines) - Abstract base learner
- `trainer.py` (200 lines) - RL-specific trainer wrapper
- `ppo/ppo_learner.py` (639 lines) - PPO implementation
- `grpo/grpo_learner.py` (531 lines) - GRPO implementation
- `common.py` (424 lines) - Shared utilities
- `algorithm_config.py` - Algorithm configurations
- `rollout/` - Generation engines
- `inference/` - Critic/reference/reward inference

### Design Philosophy
- **Disaggregated Architecture**: Separate meshes for actor/critic/rollout/reference
- **Flexible Rollout Engines**: Vanilla, vLLM, SGLang-JAX support
- **CPU Offloading**: Memory optimization for multi-model training
- **Async Execution**: Parallel rollout and training when disaggregated
- **Backbone Sharing**: LoRA models share base weights across roles

## 2. RLCluster - Central Orchestrator
### Cluster Architecture
#### Role-to-Mesh Mapping
- **Role Enum**: ACTOR, CRITIC, REFERENCE, REWARD, ROLLOUT
- **ClusterConfig**: Maps each role to JAX mesh
- **Colocated vs Disaggregated**
  - Colocated: Same mesh for multiple roles (memory sharing)
  - Disaggregated: Separate meshes (parallel execution)
- **Logical Axis Rules**: Support for logical sharding

```python
class Role(enum.Enum):
    ACTOR = "actor"      # Policy model being trained
    CRITIC = "critic"    # Value model (PPO only)
    REFERENCE = "reference"  # Fixed reference for KL penalty
    REWARD = "reward"    # Reward model for scoring
    ROLLOUT = "rollout"  # Generation/sampling model
```

#### ClusterConfig Structure
```python
@dataclasses.dataclass(kw_only=True, frozen=True)
class ClusterConfig:
    role_to_mesh: dict[Role, Mesh]  # Core mapping
    role_to_logical_axis_rule: dict[Role, flax.typing.LogicalRules] | None
    rollout_engine: str | type[BaseRollout]  # "vanilla", "vllm", "sglang_jax"
    offload_to_cpu: bool  # CPU offloading for memory
    training_config: RLTrainingConfig
    rollout_config: dict[Mode, RolloutConfig] | RolloutConfig
```

### Model Loading and Initialization
#### Multi-Model Setup
- **Input Models**: actor, critic, reference, reward
- **Model or Path**: Supports nnx.Module or checkpoint path
- **Automatic Resharding**: Moves models to target meshes
- **Data Type Casting**: Optional dtype conversion

#### Backbone Sharing Map
```python
# LoRA scenario: Actor and Reference share backbone
self._backbone_sharing_map[Role.ACTOR] = [Role.REFERENCE, Role.ROLLOUT]
# Updates to LoRA params propagate to all sharing roles
```

**Sharing Logic**:
- If actor/rollout on same mesh → share model instance
- If LoRA enabled + reference on same mesh → share backbone (non-LoRA params)
- Critic can share backbone with actor if colocated

#### CPU Offloading Strategy
```python
def _maybe_offload_model_to_cpu(model: nnx.Module, role: Role):
    """Moves model params to pinned_host memory"""
    if self.cluster_config.offload_to_cpu:
        self._put_model_on_memory_kind(model, "pinned_host")
        self._update_models_sharing_weights(nnx.state(model), role)
```

**When Used**:
- After loading model to mesh (offload immediately)
- Before using model (load to device)
- Enables fitting multiple large models in limited HBM

### Component Initialization
#### 1. Rollout Initialization
**Engine Selection**:
- `"vanilla"`: VanillaRollout (pure JAX, KV cache)
- `"vllm"`: VllmRollout (optimized, paged attention)
- `"sglang_jax"`: SglangJaxRollout (RadixCache support)
- Custom: Subclass of BaseRollout

```python
# Vanilla rollout setup
self._rollout = vanilla_rollout.VanillaRollout(
    self.rollout_actor,
    self.tokenizer,
    cache_config_or_size=base_rollout.CacheConfig(
        cache_size=max_kv_cache_size,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    ),
)
```

**Rollout Config Modes**:
- TRAIN mode: Higher temperature, larger batch
- EVAL mode: Lower temperature, deterministic sampling

#### 2. Inference Worker Initialization
```python
inference_models = {}
if self.critic is not None:
    inference_models["critic"] = self.critic
if self.reference is not None:
    inference_models["reference"] = self.reference
if self.reward is not None:
    inference_models["reward"] = self.reward

self._inference_worker = InferenceWorker(inference_models)
```

**Purpose**: Centralized inference for:
- Reference model log probabilities
- Critic model value predictions
- Reward model scoring

#### 3. Trainer Initialization
**Actor Trainer**:
```python
self._actor_trainer = rl_trainer.Trainer(
    model=self.train_actor,
    optimizer=config.training_config.actor_optimizer,
    training_config=actor_config,
    custom_checkpoint_metadata_fn=lambda: {"global_step": self.global_steps + 1},
    metrics_logger=self._rl_metrics_logger,
    perf_tracer=self._perf,
)
```

**Critic Trainer** (PPO only):
- Separate optimizer for value function
- Independent checkpoint directory
- Same pattern as actor trainer

### Core Operations
#### Generation (generate)
```python
def generate(
    prompts: list[str],
    apply_chat_template: bool = False,
    mode: Mode = Mode.TRAIN,
    micro_batch_size: int | None = None,
) -> RolloutOutput:
```

**Process**:
1. Apply chat template if requested
2. Load rollout model from CPU if offloaded
3. Generate in micro-batches
4. Concatenate outputs across batches
5. Offload model back to CPU
6. Return RolloutOutput (text, tokens, logits, logprobs)

**Micro-batching**: Splits large prompt list to fit memory

#### Reference Log Probabilities (get_ref_per_token_logps)
```python
def get_ref_per_token_logps(
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    micro_batch_size: int | None = None,
) -> jax.Array:
```

**Purpose**: Compute KL divergence penalty
- Uses reference model (frozen)
- Computes log P(completion | prompt) per token
- Micro-batched inference
- Returns shape: [batch_size, completion_length]

#### Old Policy Log Probabilities (get_old_per_token_logps)
**Purpose**: For multi-iteration algorithms (PPO, GRPO with μ > 1)
- Uses current policy model snapshot
- Computed before policy updates
- Enables importance sampling correction

#### Value Predictions (get_values)
**PPO Only**: Critic model predicts state values
- Used for Generalized Advantage Estimation (GAE)
- Returns values for all tokens in completion

#### Weight Synchronization (sync_weights)
```python
def sync_weights():
    """Syncs weights from actor trainer to rollout model"""
    if should_sync_weights:
        filter_types = nnx.LoRAParam if is_lora else nnx.Param
        src_params = nnx.state(self.actor_trainer.model, filter_types)
        self.rollout.update_params(src_params, filter_types)
    self.global_steps += 1
```

**When Called**: End of each global step
**LoRA Handling**: Only sync LoRA params, backbone shared

### Metrics and Logging
#### Buffered Metrics System
```python
def buffer_metrics(metrics: MetricsT, mode: Mode = Mode.TRAIN):
    """Buffers metrics until global step increment"""
```

**Buffering Strategy**:
- Accumulate metrics within a step
- Log when global_steps increments
- Separate buffers for TRAIN/EVAL modes

**Async Metrics** (buffer_metrics_async):
- For parallel rollout/training
- Each step has explicit step number
- Logs when step completes

### Performance Tracing
**PerfTracer Integration**:
```python
with self._perf.span("rollout", mesh.devices) as span:
    outputs = self.rollout.generate(prompts, config)
    span.device_end([o.logits for o in outputs])
```

**Traced Operations**:
- `rollout`: Text generation
- `refer_inference`: Reference model forward
- `old_actor_inference`: Old policy forward
- `actor_training`: Policy updates
- `critic_training`: Value function updates
- `weight_sync`: Parameter synchronization

## 3. RLLearner - Abstract Base Class
### Base Architecture
#### Generic Type System
```python
TConfig = TypeVar("TConfig", bound=algo_config_lib.AlgorithmConfig)

class RLLearner(abc.ABC, Generic[TConfig]):
    """Base class for PPOLearner, GRPOLearner, etc."""
```

**Type Safety**: Each learner specifies its config type

#### Initialization Components
```python
def __init__(
    rl_cluster: RLCluster,
    algo_config: TConfig,
    reward_fns: RewardFn | List[RewardFn],
    metric_fns: Sequence[MetricFn] | None = None,
    data_shuffle_seed: int | None = None,
):
```

**Key Attributes**:
- `rl_cluster`: Access to all models and infrastructure
- `algo_config`: Algorithm-specific hyperparameters
- `reward_fns`: Custom reward computation functions
- `metric_fns`: User-defined metrics for logging
- `data_shuffle_seed`: For reproducible data ordering

### Abstract Methods (Algorithm-Specific)
#### 1. Generate and Compute Advantage
```python
@abstractmethod
def _generate_and_compute_advantage(
    training_input: TrainingInputT,
    mode: Mode = Mode.TRAIN,
) -> common.TrainExample:
    """
    1. Generate completions from prompts
    2. Compute rewards for completions
    3. Compute advantages based on algorithm
    Returns: TrainExample with all data for training
    """
```

**Algorithm Differences**:
- **PPO**: GAE advantages, needs critic values
- **GRPO**: Group relative advantages, no critic needed

#### 2. Compute Trajectory IDs
```python
@abstractmethod
def _compute_trajectory_ids(
    example: TrainingInputT, 
    steps: int
) -> List[str]:
    """Unique ID for each trajectory (for logging/tracking)"""
```

#### 3. Number of Iterations (μ)
```python
@abstractmethod
def _num_iterations() -> int:
    """How many epochs per batch of rollouts"""
```

#### 4. Number of Generations (G)
```python
@abstractmethod
def _num_generations() -> int:
    """How many completions per prompt (GRPO) or 1 (PPO)"""
```

### Reward Computation Pipeline
#### Compute Rewards Method
```python
def _compute_rewards(
    prompts: List[str],
    completions: List[str],
    mode: Mode,
    step: int | None = None,
    **kwargs,
) -> np.ndarray:
    """
    Combines multiple reward functions
    Returns: [batch_size] array of total rewards
    """
```

**Process**:
1. Call each reward_fn with prompts and completions
2. Collect rewards into [batch_size, num_fns] array
3. Sum across reward functions: `sum_rewards = np.nansum(rewards, axis=1)`
4. Log individual and aggregated rewards

**Reward Function Signature**:
```python
RewardFn = Callable[..., List[float]]
# Args: prompts, completions, **kwargs
# Returns: List of scalar rewards (one per prompt)
```

**Logged Metrics**:
- `rewards/sum`: Total reward per trajectory
- `rewards/min`, `rewards/max`: Range of rewards
- `rewards/{fn_name}`: Individual reward function scores
- `prompts`, `completions`: Text for inspection

### Data Preparation Pipeline
#### Process Accumulated Batches
```python
def _process_accumulated_batches(
    micro_batches: list[TrainingInputT],
    micro_batch_sizes: list[int],
    mode: Mode,
) -> list[common.TrainExample]:
    """
    Merges micro-batches, generates, computes advantages, splits back
    """
```

**Why Accumulation?**
- Rollout optimal batch size ≠ Training optimal batch size
- Accumulate to service_target_batch_size (LCM of micro sizes)
- Single large forward pass is more efficient

**Steps**:
1. Merge micro-batches into one large batch
2. Repeat samples if needed (sample_repeat = G for GRPO)
3. Call `_generate_and_compute_advantage` once
4. Split TrainExample back to original micro boundaries
5. Return list of small TrainExample chunks

#### Prepare Data Method
```python
def _prepare_data(
    iterator: Iterator[TrainingInputT],
    proceed_num_steps: int,
    sample_repeat: int,
    batch_repeat: int,
    service_target_batch_size: int,
    data_queue: AbstractDataQueue,
    async_loading: bool = False,
    mode: Mode = Mode.TRAIN,
):
    """Orchestrates the data preparation pipeline"""
```

**Pipeline Flow**:
```
Iterator → Accumulate → Merge → Repeat Samples → 
Generate & Compute Advantage → Split → Enqueue → 
Training Queue
```

**Key Parameters**:
- `proceed_num_steps`: How many micro-batches to process (-1 = all)
- `sample_repeat`: G in GRPO (repeat each prompt G times)
- `batch_repeat`: μ in algorithms (repeat for μ epochs)
- `service_target_batch_size`: LCM of rollout and inference batch sizes
- `async_loading`: Enqueue immediately vs accumulate then enqueue

**Accumulation Logic**:
```python
accumulated_samples_num += cur_batch_size
if accumulated_samples_num >= service_target_batch_size:
    produced = self._process_accumulated_batches(...)
    # Reset accumulation
    micro_batches.clear()
    accumulated_samples_num = 0
```

**Boundary Handling**:
- At `proceed_num_steps` boundary: Process tail
- At `StopIteration`: Process tail
- Tail = remaining micro-batches not yet reaching threshold

**Async vs Sync Enqueuing**:
- **Async**: Enqueue each TrainExample immediately (parallel rollout/training)
- **Sync**: Accumulate all, then enqueue with repeats (sequential)

### Training Loop (train method)
#### Batch Size Hierarchy
```python
full_batch_size = len(first_item["prompts"])  # e.g., 512
mini_batch_size = config.mini_batch_size or full_batch_size  # e.g., 64
train_micro_batch_size = config.train_micro_batch_size or mini_batch_size  # e.g., 8
rollout_micro_batch_size = rollout_micro_batch_size or train_micro_batch_size
compute_logps_micro_batch_size = compute_logps_micro_batch_size or train_micro_batch_size
```

**Relationships**:
- `full_batch_size`: Total prompts per global step
- `mini_batch_size`: Batch for one optimizer step
- `train_micro_batch_size`: Gradient accumulation micro-batch
- `rollout_micro_batch_size`: Generation batch size
- `compute_logps_micro_batch_size`: Inference batch size

**service_target_batch_size** = LCM(rollout_micro_batch_size, compute_logps_micro_batch_size)

#### Training Loop Structure
```python
while True:  # Loop over M global steps
    with perf.span_group("global_step"):
        self._run_global_step(...)
        
        if should_sync_weights:
            self.rl_cluster.sync_weights()
        else:
            self.rl_cluster.global_steps += 1
        
        self.rl_cluster.buffer_metrics(perf.export(), mode=TRAIN)
        
        if actor_trainer.train_steps >= max_steps:
            break
```

**Global Step Breakdown**:
```
Global Step = full_batch_size / mini_batch_size mini-batch steps
Mini-batch Step = gradient_accumulation_steps micro-batch steps
```

#### Run Global Step
```python
def _run_global_step(...):
    for _ in range(full_batch_size // mini_batch_size):
        self._run_mini_batch_step(...)
```

**Each mini-batch step**:
1. Run all micro-batch steps (data preparation)
2. Update actor/critic models
3. Sync iter_steps with trainer

#### Run Mini-Batch Step
```python
def _run_mini_batch_step(...):
    self._run_all_micro_batch_steps(...)
```

#### Run All Micro-Batch Steps
```python
def _run_all_micro_batch_steps(...):
    # Setup queues
    train_data_queue = SimpleDataQueue(maxsize=grad_acc_steps * num_iterations + 1)
    eval_data_queue = SimpleDataQueue(maxsize=0)
    
    # Launch async data preparation
    future = executor.submit(
        self._prepare_data,
        iterator=train_iterator,
        proceed_num_steps=grad_acc_steps,
        ...
    )
    
    # Training loop
    while True:
        curr_train_ds = train_data_queue.get(block=True)
        if curr_train_ds is None:
            break
        
        # Trigger eval if needed
        if eval_ds and should_eval:
            self._prepare_data(eval_iterator, ...)
            curr_eval_ds = eval_data_queue.get()
        
        # Update models
        self.rl_cluster.update_actor(curr_train_ds, curr_eval_ds)
        if has_critic:
            self.rl_cluster.update_critic(curr_train_ds, curr_eval_ds)
    
    future.result()  # Wait for data preparation completion
```

**Key Features**:
- **Producer-Consumer**: Data prep in thread, training in main
- **Blocking Get**: Training waits for data
- **Eval Triggering**: Every N steps, prepare eval data
- **Critic Training**: After actor update (PPO only)

#### Async Execution Benefits
When `can_enable_async_rollout = True` (disaggregated setup):
- Rollout happens in parallel with training
- Training dequeues pre-generated batches
- Logs `actor_dequeue_time` metric
- Improves throughput significantly

### Checkpoint Resume Support
#### Fast-Forward Iterator
```python
while self._iter_steps < self._last_iter_step:
    next(iterator)  # Skip micro-batches already processed
    self._iter_steps += 1
```

**Resume Logic**:
- `global_steps` restored from checkpoint metadata
- `iter_steps` synced with actor_trainer
- Iterator fast-forwarded to correct position

## 4. PPOLearner Implementation
### PPO Algorithm Overview
**Proximal Policy Optimization** (Schulman et al., 2017)
- Actor-Critic architecture
- Clipped surrogate objective
- Prevents large policy updates
- Requires separate value function (critic)

**Key Innovation**: Trust region without explicit constraints
- Clip importance ratio: `ratio ∈ [1-ε, 1+ε]`
- Pessimistic bound: `max(clipped_loss, unclipped_loss)`

### PPOConfig
```python
@dataclasses.dataclass
class PPOConfig(AlgorithmConfig):
    algo_variant: str = "ppo"
    advantage_estimator: str = "gae"
    policy_loss_fn: str = "ppo"
    num_iterations: int = 1  # μ epochs per batch
    
    # GAE parameters
    gamma: float = 1.0  # Discount factor
    gae_lambda: float = 0.95  # GAE λ
    
    # PPO loss parameters
    beta: float = 0.04  # KL penalty coefficient
    epsilon: float = 0.2  # Clipping range
    epsilon_low: float | None = None  # Lower clip (defaults to epsilon)
    epsilon_high: float | None = None  # Upper clip (defaults to epsilon)
    epsilon_c: float | None = None  # Dual-clip PPO threshold
    entropy_coef: float | None = None  # Entropy regularization
    clip_range_value: float = 0.2  # Value function clipping
    kl_method: str = "low_var_kl"  # "kl", "mse_kl", "low_var_kl"
```

**Dual-Clip PPO** (Ye et al., 2019):
- Prevents negative-advantage updates
- `epsilon_c > 1.0` required
- Additional clipping: `min(-epsilon_c * A, clipped_loss)` when A < 0

### PPOLearner Initialization
#### Model Requirements Check
```python
def __init__(...):
    # Must have exactly one of: reward_fns OR reward model
    if bool(reward_fns) == bool(rl_cluster.reward_model):
        raise ValueError("Need one of reward_fns or reward model")
    
    # Must have critic model
    if not rl_cluster.critic:
        raise ValueError("PPO requires critic model")
```

**Two Reward Options**:
1. **Reward Functions**: Python callables (flexible, custom logic)
2. **Reward Model**: Neural network (trained scorer)

#### Actor Trainer Configuration
```python
# Set policy loss function
self.rl_cluster.actor_trainer.with_loss_fn(
    policy_loss_fn=ppo_policy_loss_fn,
    has_aux=True,
)

# Configure model input generation
self.rl_cluster.actor_trainer.with_gen_model_input_fn(
    lambda x: {
        "train_example": x,
        "epsilon_low": self.algo_config.epsilon_low,
        "epsilon_high": self.algo_config.epsilon_high,
        "epsilon_c": self.algo_config.epsilon_c,
        "entropy_coef": self.algo_config.entropy_coef,
        "pad_id": self.rl_cluster.rollout.pad_id(),
        "eos_id": self.rl_cluster.rollout.eos_id(),
    }
)

# Configure metrics to log from aux
self.rl_cluster.actor_trainer.with_rl_metrics_to_log({
    "pg_clipfrac": np.mean,  # Fraction of samples clipped
    "pg_clipfrac_lower": np.mean,  # If dual-clip enabled
    "loss/entropy": np.mean,  # If entropy regularization
})
```

#### Critic Trainer Configuration
```python
self.rl_cluster.critic_trainer.with_loss_fn(
    ppo_value_loss_fn,
    has_aux=True,
)

self.rl_cluster.critic_trainer.with_gen_model_input_fn(
    lambda x: {
        "train_example": x,
        "clip_range_value": self.algo_config.clip_range_value,
        "pad_id": pad_id,
        "eos_id": eos_id,
    }
)

self.rl_cluster.critic_trainer.with_rl_metrics_to_log({
    "vpred_mean": np.mean,
    "vf_clipfrac": np.mean,
})
```

### Generate and Compute Advantage
#### Complete Pipeline
```python
def _generate_and_compute_advantage(
    training_input: TrainingInputT,
    mode: Mode = Mode.TRAIN,
) -> TrainExample:
```

**Step 1: Generation**
```python
completion_output = self.rl_cluster.generate(
    prompts=training_input["prompts"],
    micro_batch_size=self._rollout_micro_batch_size,
)
completion_ids = completion_output.tokens
prompt_ids = jnp.array(completion_output.left_padded_prompt_tokens)
```

**Step 2: Mask Creation**
```python
prompt_mask = (prompt_ids != pad_value).astype("int32")
completion_mask = common.np_make_completion_mask(
    completion_ids, eos_tok=eos_value
)
```

**Step 3: Reference Log Probabilities** (if beta ≠ 0)
```python
ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
    prompt_tokens=prompt_ids,
    completion_tokens=jax_completion_ids,
    pad_id=pad_value,
    eos_id=eos_value,
    micro_batch_size=self._compute_logps_micro_batch_size,
)
```

**Step 4: Old Policy Log Probabilities**
```python
old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
    prompt_tokens=prompt_ids,
    completion_tokens=jax_completion_ids,
    micro_batch_size=self._compute_logps_micro_batch_size,
)
```

**Step 5: Value Predictions**
```python
values = self.rl_cluster.get_values(
    prompt_tokens=prompt_ids,
    completion_tokens=jax_completion_ids,
    pad_id=pad_value,
    eos_id=eos_value,
)
# Keep only completion values
values = values[:, -logits_to_keep - 1 : -1]
values = values * jax_completion_mask
```

**Step 6: Reward Computation**
```python
if self._use_reward_model:
    scores = self.rl_cluster.get_rewards(...)
    last_token_scores = scores[jnp.arange(batch_size), eos_idx]
else:
    last_token_scores = self._compute_rewards(
        prompts=training_input["prompts"],
        completions=completion_output.text,
        mode=mode,
    )

# Initialize rewards (zeros) and add final reward
rewards = jnp.zeros_like(jax_completion_ids)
rewards = rewards.at[jnp.arange(batch_size), eos_idx].add(last_token_scores)

# Subtract KL penalty if beta ≠ 0
if self.algo_config.beta != 0.0:
    kl = common.compute_kl_divergence(
        old_per_token_logps,
        ref_per_token_logps,
        method=self.algo_config.kl_method,
    )
    rewards = rewards - self.algo_config.beta * kl
```

**Reward Structure**:
- All tokens get 0 reward except final token
- Final token gets full episode reward
- KL penalty applied per token

**Step 7: GAE Advantage Estimation**
```python
advantages, returns = advantage_estimator(
    rewards=rewards,
    values=values,
    completion_mask=jax_completion_mask,
    gamma=self.algo_config.gamma,
    gae_lambda=self.algo_config.gae_lambda,
)
```

**GAE Formula**:
```
δₜ = rₜ + γ·Vₜ₊₁ - Vₜ
Aₜ = Σᵢ₌₀ (γλ)ⁱ δₜ₊ᵢ
```

**Step 8: Metrics Logging**
```python
self.rl_cluster.buffer_metrics({
    "score/mean": (np.mean(last_token_scores), np.mean),
    "reward/mean": (np.mean(sequence_rewards), np.mean),
    "reward_kl_penalty": (kl_mean, np.mean),
    "completions/mean_length": (mean_length, np.mean),
    "advantages/mean": (advantages.mean(), np.mean),
    "returns/mean": (returns.mean(), np.mean),
    "old_values/mean": (values.mean(), np.mean),
}, mode=mode)
```

**Step 9: Return TrainExample**
```python
return TrainExample(
    prompt_ids=prompt_ids,
    prompt_mask=prompt_mask,
    completion_ids=jax_completion_ids,
    completion_mask=jax_completion_mask,
    ref_per_token_logps=ref_per_token_logps,
    advantages=advantages,
    returns=returns,
    old_per_token_logps=old_per_token_logps,
    old_values=values,
)
```

### PPO Policy Loss Function
```python
@registry.register("policy_loss_fn", "ppo")
def ppo_policy_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    epsilon_low: float,
    epsilon_high: float,
    epsilon_c: float | None,
    entropy_coef: float | None,
    pad_id: int,
    eos_id: int,
):
```

**Step 1: Compute New Log Probabilities**
```python
per_token_logps, logits = common.compute_per_token_logps(
    model,
    prompt_tokens=prompt_ids,
    completion_tokens=completion_ids,
    pad_id=pad_id,
    eos_id=eos_id,
    stop_gradient=False,
    return_logits=True,
)
```

**Step 2: Compute Importance Ratio**
```python
ratio = jnp.exp(per_token_logps - old_per_token_logps)
ratio_clipped = jnp.clip(ratio, 1 - epsilon_low, 1 + epsilon_high)
```

**Step 3: Vanilla PPO Loss**
```python
pg_losses_1 = -ratio * advantages
pg_losses_2 = -ratio_clipped * advantages
clip_pg_losses_1 = jnp.maximum(pg_losses_1, pg_losses_2)
```

**Intuition**: Take pessimistic bound (maximum loss)

**Step 4: Dual-Clip (Optional)**
```python
if epsilon_c is not None:
    pg_losses_3 = -epsilon_c * advantages
    clip_pg_losses_2 = jnp.minimum(pg_losses_3, clip_pg_losses_1)
    pg_losses = jnp.where(advantages < 0.0, clip_pg_losses_2, clip_pg_losses_1)
```

**Intuition**: When advantage is negative, limit how much we reduce probability

**Step 5: Token-Mean Normalization**
```python
policy_loss = ppo_helpers.masked_mean(pg_losses, completion_mask)
```

**Step 6: Entropy Regularization (Optional)**
```python
if entropy_coef is not None and entropy_coef > 0.0:
    token_entropy = ppo_helpers.compute_entropy_from_logits(logits)
    entropy_loss = ppo_helpers.masked_mean(token_entropy, completion_mask)
    policy_loss -= entropy_coef * entropy_loss
```

**Purpose**: Encourage exploration, prevent collapse to deterministic policy

**Return**: `(policy_loss, aux_dict)`

### PPO Value Loss Function
```python
def ppo_value_loss_fn(
    model: nnx.Module,
    train_example: TrainExample,
    clip_range_value: float | None,
    pad_id: int,
    eos_id: int,
):
```

**Step 1: Get New Value Predictions**
```python
vpreds = common.compute_score(
    model,
    prompt_ids,
    completion_ids,
    pad_id,
    eos_id,
    stop_gradient=False,
)
vpreds = vpreds[:, -logits_to_keep - 1 : -1]
```

**Step 2: Clipped Value Predictions**
```python
vpred_clipped = jnp.clip(
    vpreds,
    values - clip_range_value,
    values + clip_range_value,
)
```

**Step 3: MSE with Clipping**
```python
vf_losses1 = jnp.square(vpreds - returns)
vf_losses2 = jnp.square(vpred_clipped - returns)
clipped_vf_losses = jnp.maximum(vf_losses1, vf_losses2)
vf_loss = 0.5 * ppo_helpers.masked_mean(clipped_vf_losses, completion_mask)
```

**Clipping Purpose**: Prevent large value function updates

**Return**: `(vf_loss, aux_dict)`

### Trajectory IDs (PPO)
```python
def _compute_trajectory_ids(example: TrainingInputT, steps: int) -> List[str]:
    batch_size = len(example["prompts"])
    row_offset = steps * batch_size
    return np.arange(row_offset, row_offset + batch_size).astype(str).tolist()
```

**Format**: Simple offset (e.g., "0", "1", "2", ...)

## 5. GRPOLearner Implementation
### GRPO Algorithm Overview
**Group Relative Policy Optimization** (Shao et al., 2024)
- No critic model needed (memory efficient)
- Generates G responses per prompt
- Computes relative advantages within group
- Similar to REINFORCE with baseline

**Key Innovation**: Group normalization
- Baseline = mean reward across G generations
- Advantage = (reward - mean) / std
- Reduces variance without separate value function

### GRPOConfig
```python
@dataclasses.dataclass
class GRPOConfig(AlgorithmConfig):
    algo_variant: str = "grpo"
    advantage_estimator: str = "grpo"
    policy_loss_fn: str = "grpo"
    loss_agg_mode: str = "sequence-mean-token-mean"
    loss_algo: str = "grpo"  # "grpo" or "gspo-token"
    
    num_generations: int = 2  # G (must be > 1)
    num_iterations: int = 1  # μ epochs per batch
    beta: float = 0.04  # KL penalty coefficient
    epsilon: float = 0.2  # Clipping range
```

**GSPO-token**: Token-level importance sampling variant
- More flexible than sequence-level GRPO
- Experimental feature

### GRPOLearner Initialization
#### Model Requirements
- **No critic needed**: Main advantage over PPO
- Must have `reward_fns`: Custom reward computation required
- Reference model optional (only if beta ≠ 0)

#### Actor Trainer Configuration
```python
loss_fn = lambda model, train_example, algo_config: policy_loss_fn(
    model,
    train_example,
    algo_config=self.algo_config,
    pad_id=self.rl_cluster.rollout.pad_id(),
    eos_id=self.rl_cluster.rollout.eos_id(),
)

self.rl_cluster.actor_trainer.with_loss_fn(loss_fn, has_aux=True)

self.rl_cluster.actor_trainer.with_gen_model_input_fn(
    lambda x: {
        "train_example": x,
        "algo_config": self.algo_config,
    }
)

self.rl_cluster.actor_trainer.with_rl_metrics_to_log({"kl": np.mean})
```

### Generate and Compute Advantage (GRPO)
#### Complete Pipeline
```python
def _generate_and_compute_advantage(
    training_input: TrainingInputT,
    mode: Mode = Mode.TRAIN,
) -> TrainExample:
```

**Step 1: Generation with Repetition**
```python
rollout_output = self.rl_cluster.generate(
    prompts=training_input["prompts"],
    mode=mode,
    micro_batch_size=self._rollout_micro_batch_size * self.algo_config.num_generations,
)
```

**Key Difference from PPO**: 
- Each prompt generates G completions
- Batch size multiplied by num_generations

**Step 2: Mask Creation**
```python
prompt_mask = prompt_ids != pad_value
completion_padding_mask = np.not_equal(completion_ids, pad_value)
completion_mask = common.np_make_completion_mask(completion_ids, eos_tok=eos_value)
completion_mask = completion_mask * completion_padding_mask
```

**Step 3: Reference Log Probabilities** (if beta ≠ 0)
```python
if self.algo_config.beta != 0.0:
    with self.rl_cluster.perf.span("refer_inference", devices):
        ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
            prompt_tokens=prompt_ids,
            completion_tokens=jax_completion_ids,
            pad_id=pad_value,
            eos_id=eos_value,
            micro_batch_size=self._compute_logps_micro_batch_size * num_generations,
        )
```

**Step 4: Old Policy Log Probabilities** (if μ > 1)
```python
if self.algo_config.num_iterations > 1:
    old_per_token_logps = self.rl_cluster.get_old_per_token_logps(
        prompt_tokens=prompt_ids,
        completion_tokens=jax_completion_ids,
        micro_batch_size=self._compute_logps_micro_batch_size * num_generations,
    )
```

**Step 5: Reward Computation**
```python
rewards = self._compute_rewards(
    prompts=training_input["prompts"],
    completions=completion_text,
    mode=mode,
    **{k: v for k, v in training_input.items() if k != "prompts"},
)
```

**Rewards Shape**: [batch_size * G] - one reward per completion

**Step 6: Group Relative Advantage Estimation**
```python
advantage_estimator = function_registry.get_advantage_estimator("grpo")
advantages = advantage_estimator(
    rewards=rewards,
    num_generations=self.algo_config.num_generations,
)
```

**GRPO Advantage Computation**:
```python
@function_registry.register_advantage_estimator("grpo")
def compute_advantages(rewards: np.ndarray, num_generations: int) -> np.ndarray:
    # Reshape to [num_prompts, G]
    mean_grouped_rewards = rewards.reshape(-1, num_generations).mean(axis=-1)
    std_grouped_rewards = rewards.reshape(-1, num_generations).std(axis=-1, ddof=1)
    
    # Broadcast back to [num_prompts * G]
    mean_grouped_rewards = mean_grouped_rewards.repeat(num_generations)
    std_grouped_rewards = std_grouped_rewards.repeat(num_generations)
    
    # Normalize
    return (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
```

**Intuition**: 
- Each group has its own baseline (mean)
- Standardize within group
- Better completions get positive advantages

**Step 7: Metrics Logging**
```python
self.rl_cluster.buffer_metrics({
    "completions/mean_length": (mean_length, np.mean),
    "completions/max_length": (max_length, np.max),
    "completions/min_length": (min_length, np.min),
}, mode=mode)

# User-defined metrics
for m_fn in self.metric_fns:
    user_metrics = m_fn(
        prompts=prompts,
        completions=completions,
        advantages=advantages,
        rewards=rewards,
        ...
    )
    self.rl_cluster.buffer_metrics(user_metrics, mode=mode)
```

**Step 8: Return TrainExample**
```python
return TrainExample(
    prompt_ids=prompt_ids,
    prompt_mask=prompt_mask,
    completion_ids=jax_completion_ids,
    completion_mask=jax_completion_mask,
    ref_per_token_logps=ref_per_token_logps,
    advantages=jax.device_put(advantages),
    old_per_token_logps=old_per_token_logps,
)
```

### GRPO Policy Loss Function
```python
@function_registry.register_policy_loss_fn("grpo")
def grpo_loss_fn(
    model,
    train_example,
    algo_config,
    pad_id,
    eos_id,
):
```

**Step 1: Compute New Log Probabilities**
```python
per_token_logps = common.compute_per_token_logps(
    model,
    prompt_tokens=train_example.prompt_ids,
    completion_tokens=completion_ids,
    pad_id=pad_id,
    eos_id=eos_id,
    stop_gradient=False,
    return_logits=False,
)
```

**Step 2: Importance Ratio**
```python
if train_example.old_per_token_logps is None:
    old_per_token_logps = jax.lax.stop_gradient(per_token_logps)
else:
    old_per_token_logps = train_example.old_per_token_logps

seq_importance_ratio = per_token_logps - old_per_token_logps
```

**Step 3: GSPO-token Adjustment** (if enabled)
```python
if loss_algo == "gspo-token":
    # Sequence-level ratio
    seq_ratio = (seq_importance_ratio * completion_mask).sum(axis=-1) / \
                jnp.clip(completion_mask.sum(-1), min=1)
    # Token-level with sequence baseline
    seq_importance_ratio = (
        per_token_logps
        - jax.lax.stop_gradient(per_token_logps)
        + jnp.expand_dims(jax.lax.stop_gradient(seq_ratio), axis=-1)
    )
    seq_importance_ratio = jnp.clip(seq_importance_ratio, max=10.0)
```

**Step 4: Clipped Loss**
```python
coef_1 = jnp.exp(seq_importance_ratio)
coef_2 = jnp.clip(coef_1, 1 - epsilon, 1 + epsilon_high)

per_token_loss = -jnp.minimum(
    coef_1 * jnp.expand_dims(advantages, 1),
    coef_2 * jnp.expand_dims(advantages, 1),
)
```

**Step 5: KL Penalty** (if beta ≠ 0)
```python
if beta is not None and beta != 0.0:
    kl = common.compute_kl_divergence(
        per_token_logps,
        train_example.ref_per_token_logps,
    )
    per_token_loss = per_token_loss + beta * kl
    
    aux["kl"] = (kl * completion_mask).sum() / jnp.clip(completion_mask.sum(), min=1)
```

**Step 6: Loss Aggregation**
```python
loss = common.aggregate_loss(
    per_token_loss,
    completion_mask,
    loss_aggregation_mode,
)
```

**Loss Aggregation Modes**:
- `"sequence-mean-token-mean"`: Mean over sequences, mean over tokens
- `"sequence-sum-token-mean"`: Sum over sequences, mean over tokens
- `"token-mean"`: Mean over all tokens (flatten)

**Return**: `(loss, aux)`

### Trajectory IDs (GRPO)
```python
def _compute_trajectory_ids(example: TrainingInputT, steps: int) -> List[str]:
    batch_size = len(example["prompts"]) // self.algo_config.num_generations
    row_offset = steps * batch_size
    
    row_offsets = np.repeat(
        np.arange(row_offset, row_offset + batch_size),
        self.algo_config.num_generations,
        axis=0,
    )
    group_offsets = np.tile(
        np.arange(self.algo_config.num_generations),
        batch_size,
    )
    
    return [f"{r_off}_{g_off}" for r_off, g_off in zip(row_offsets, group_offsets)]
```

**Format**: "row_offset_group_offset" (e.g., "0_0", "0_1", "1_0", "1_1", ...)
**Purpose**: Track which generation within group

## 6. Supporting Components
### InferenceWorker
**Purpose**: Hosts critic, reference, and reward models for inference

```python
class InferenceWorker:
    def __init__(self, models: dict[str, nnx.Module]):
        # Supported keys: "critic", "reference", "reward"
        self._models = models
```

**Methods**:
- `get_rewards(prompt_tokens, completion_tokens, pad_id, eos_id)`: Reward model inference
- `get_ref_per_token_logps(...)`: Reference model log probabilities
- `get_values(...)`: Critic model value predictions
- `get_model(role)`: Access model by role
- `update_model(role, params)`: Update model parameters

**Compute Score** (common utility):
```python
def compute_score(
    model: nnx.Module,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
) -> jax.Array:
    """Forward pass returning final layer outputs (values or rewards)"""
```

### BaseRollout Interface
**Abstract base class for all rollout engines**

```python
class BaseRollout(ABC):
    @abstractmethod
    def generate(
        prompts: list[str],
        rollout_config: RolloutConfig,
        **kwargs,
    ) -> RolloutOutput:
        """Generate text completions"""
    
    @abstractmethod
    def get_per_token_logps(
        prompt_tokens: jax.Array,
        completion_tokens: jax.Array,
        completion_mask: jax.Array | None = None,
    ) -> jax.Array:
        """Compute log probabilities for given completions"""
    
    @abstractmethod
    def model() -> nnx.Module:
        """Return the underlying model"""
    
    @abstractmethod
    def update_params(params: PyTree, filter_types=None):
        """Update model parameters (for weight sync)"""
```

**RolloutConfig** (key fields):
```python
@dataclasses.dataclass
class RolloutConfig:
    max_tokens_to_generate: int = 64
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int | None = None
    seed: jax.Array | None = None
    max_prompt_length: int = 64
    kv_cache_size: int = 1024
    data_type: jnp.dtype | None = None
    eos_tokens: list[int] | None = None
    
    # vLLM specific
    rollout_vllm_model_version: str = ""
    rollout_vllm_hbm_utilization: float = 0.2
    rollout_vllm_init_with_random_weights: bool = True
    rollout_vllm_tpu_backend_type: str | None = None
    
    # SGLang JAX specific
    rollout_sglang_jax_model_version: str = ""
    rollout_sglang_jax_context_length: int | None = None
    rollout_sglang_jax_mem_fraction_static: float = 0.2
    rollout_sglang_jax_disable_radix_cache: bool = True
```

**RolloutOutput**:
```python
@dataclasses.dataclass
class RolloutOutput:
    text: list[str]  # Decoded text
    logits: jax.Array  # Per-step logits
    tokens: np.ndarray  # Generated token IDs
    left_padded_prompt_tokens: np.ndarray  # Prompt tokens
    logprobs: list[float] | None  # Log probabilities
```

### RL Trainer
**Extends PeftTrainer with RL-specific features**

```python
class Trainer(peft_trainer.PeftTrainer):
    def __init__(
        model: nnx.Module,
        optimizer: optax.GradientTransformation,
        training_config: TrainingConfig,
        custom_checkpoint_metadata_fn: Callable[[], dict],
        metrics_logger: MetricsLogger,
        perf_tracer: Tracer,
    ):
```

**Additional Features**:
- `rl_metrics_to_log`: Dict mapping metric names to aggregation functions
- `tqdm_metrics_to_display`: List of metrics for progress bar
- `custom_checkpoint_metadata_fn`: Lambda returning dict (e.g., global_step)
- `restored_global_step()`: Get global step from checkpoint

**Overrides**:
- `custom_checkpoint_metadata()`: Save extra RL state
- `_post_process_train_step(aux)`: Extract RL metrics from aux
- `_post_process_eval_step(aux)`: Extract RL metrics from aux

### Common Utilities
#### TrainExample Dataclass
```python
@flax.struct.dataclass(frozen=True)
class TrainExample:
    prompt_ids: jax.Array
    prompt_mask: jax.Array
    completion_ids: jax.Array
    completion_mask: jax.Array
    advantages: jax.Array
    ref_per_token_logps: jax.Array | None
    old_per_token_logps: jax.Array | None
```

**Subclasses**:
- `ppo.TrainExample`: Adds `returns`, `old_values`
- `grpo.TrainExample`: No additional fields

#### KL Divergence Methods
```python
def compute_kl_divergence(
    per_token_logps: jax.Array,
    ref_per_token_logps: jax.Array,
    method: str = "low_var_kl",
) -> jax.Array:
```

**Methods**:
1. **"kl"**: Forward KL = logp - ref_logp
   - Unbiased, high variance
2. **"mse_kl"**: 0.5 * (logp - ref_logp)²
   - Biased, low variance
3. **"low_var_kl"**: exp(ref_logp - logp) - (ref_logp - logp) - 1
   - Unbiased, low variance (Schulman approximation)

#### Compute Per-Token Log Probabilities
```python
def compute_per_token_logps(
    model: nnx.Module,
    prompt_tokens: jax.Array,
    completion_tokens: jax.Array,
    pad_id: int,
    eos_id: int,
    stop_gradient: bool = True,
    return_logits: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
```

**Process**:
1. Concatenate prompt and completion
2. Create causal attention mask
3. Build position IDs
4. Forward pass through model
5. Extract completion logits
6. Apply log_softmax and gather selected tokens

#### RepeatIterable
```python
class RepeatIterable(Iterable):
    def __init__(
        data: list[Any],
        repeat: int,
        mini_batch_size: int | None = None,
        shuffle: bool = False,
        key: jnp.ndarray | None = None,
    ):
```

**Purpose**: Iterate over rollout batches with:
- Shuffling within each batch
- Slicing into mini-batches
- Repeating for multiple epochs

**Use Case**: PPO/GRPO μ iterations over same rollout data

## 7. Configuration Classes
### RLTrainingConfig
```python
@dataclasses.dataclass
class RLTrainingConfig(peft_trainer.TrainingConfig):
    actor_optimizer: optax.GradientTransformation
    critic_optimizer: optax.GradientTransformation | None = None
    mini_batch_size: int | None = None
    train_micro_batch_size: int | None = None
    rollout_micro_batch_size: int | None = None
    compute_logps_micro_batch_size: int | None = None
```

**Validation**:
- All batch sizes must be positive integers
- `train_micro_batch_size` must divide `mini_batch_size`
- `gradient_accumulation_steps` auto-computed: `mini_batch_size // train_micro_batch_size`

### AlgorithmConfig
```python
@dataclasses.dataclass
class AlgorithmConfig:
    algo_variant: str = "grpo"  # "grpo", "ppo", "gspo"
    advantage_estimator: str = "grpo"  # "grpo", "gae"
    policy_loss_fn: str = "grpo"  # "grpo", "ppo"
```

**Purpose**: Base class for PPOConfig, GRPOConfig

## 8. Complete Training Example
### PPO Training Setup
```python
import jax
from flax import nnx
from tunix.rl import rl_cluster, ppo_learner
from tunix.models.automodel import AutoModel

# 1. Load models
actor = AutoModel.from_pretrained("google/gemma-2-2b")
critic = AutoModel.from_pretrained("google/gemma-2-2b")  # Separate or shared backbone
reference = actor  # Can share with actor

# 2. Setup meshes
devices = jax.devices()
actor_mesh = Mesh(devices, ("data",))
critic_mesh = actor_mesh  # Colocated
rollout_mesh = actor_mesh
reference_mesh = actor_mesh

# 3. Create cluster config
cluster_config = rl_cluster.ClusterConfig(
    role_to_mesh={
        rl_cluster.Role.ACTOR: actor_mesh,
        rl_cluster.Role.CRITIC: critic_mesh,
        rl_cluster.Role.ROLLOUT: rollout_mesh,
        rl_cluster.Role.REFERENCE: reference_mesh,
    },
    rollout_engine="vanilla",  # or "vllm", "sglang_jax"
    offload_to_cpu=False,
    training_config=rl_cluster.RLTrainingConfig(
        actor_optimizer=optax.adam(1e-6),
        critic_optimizer=optax.adam(3e-6),
        mini_batch_size=64,
        train_micro_batch_size=8,
        rollout_micro_batch_size=8,
        compute_logps_micro_batch_size=8,
        max_steps=1000,
        checkpoint_root_directory="checkpoints/",
    ),
    rollout_config=rl_cluster.RolloutConfig(
        max_tokens_to_generate=128,
        temperature=0.9,
        top_p=0.95,
        kv_cache_size=2048,
    ),
)

# 4. Initialize RL cluster
rl_cluster_instance = rl_cluster.RLCluster(
    actor=actor,
    critic=critic,
    reference=reference,
    reward=None,  # Using reward functions instead
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

# 5. Define reward function
def accuracy_reward(prompts, completions, **kwargs):
    # Custom logic: check if answer is correct
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # Parse and check correctness
        is_correct = check_answer(prompt, completion)
        rewards.append(1.0 if is_correct else 0.0)
    return rewards

# 6. Create PPO config
ppo_config = ppo_learner.PPOConfig(
    num_iterations=1,
    gamma=1.0,
    gae_lambda=0.95,
    beta=0.04,
    epsilon=0.2,
    clip_range_value=0.2,
    entropy_coef=0.01,
)

# 7. Initialize learner
learner = ppo_learner.PPOLearner(
    rl_cluster=rl_cluster_instance,
    ppo_config=ppo_config,
    reward_fns=accuracy_reward,
)

# 8. Prepare dataset
train_ds = [
    {"prompts": ["What is 2+2?", "What is 3+3?", ...]},
    {"prompts": ["What is 5+5?", "What is 7+7?", ...]},
    ...
]

# 9. Train
learner.train(train_ds=train_ds, eval_ds=eval_ds)
```

### GRPO Training Setup (Simpler)
```python
# Similar to PPO but:
# 1. No critic model needed
cluster_config = rl_cluster.ClusterConfig(
    role_to_mesh={
        rl_cluster.Role.ACTOR: actor_mesh,
        rl_cluster.Role.ROLLOUT: rollout_mesh,
        rl_cluster.Role.REFERENCE: reference_mesh,
        # No CRITIC role
    },
    ...
)

rl_cluster_instance = rl_cluster.RLCluster(
    actor=actor,
    critic=None,  # No critic
    reference=reference,
    reward=None,
    tokenizer=tokenizer,
    cluster_config=cluster_config,
)

grpo_config = grpo_learner.GRPOConfig(
    num_generations=4,  # Generate 4 completions per prompt
    num_iterations=1,
    beta=0.04,
    epsilon=0.2,
)

learner = grpo_learner.GRPOLearner(
    rl_cluster=rl_cluster_instance,
    algo_config=grpo_config,
    reward_fns=accuracy_reward,
)

learner.train(train_ds=train_ds)
```

## 9. Best Practices
### Mesh Configuration
**Colocated Setup** (Single Host):
- All roles on same mesh
- Lower latency
- Higher memory usage
- Sequential execution

**Disaggregated Setup** (Multi-Host):
- Separate meshes for actor/rollout/reference
- Parallel execution (rollout while training)
- Lower memory per host
- Higher communication overhead

### Batch Size Tuning
**Guidelines**:
1. `rollout_micro_batch_size`: As large as generation memory allows
2. `compute_logps_micro_batch_size`: As large as inference memory allows
3. `train_micro_batch_size`: As large as training memory allows
4. `mini_batch_size`: Typically 64-256
5. All must divide `full_batch_size`

### CPU Offloading
**When to Use**:
- Multiple large models (actor + critic + reference)
- Limited HBM
- Disaggregated setup with idle time

**Trade-offs**:
- Reduces HBM usage significantly
- Adds host-device transfer overhead
- Best with fast host memory (pinned)

### Rollout Engine Selection
**Vanilla**:
- Pure JAX implementation
- Easy to customize
- Good for small models/batches

**vLLM**:
- Optimized for throughput
- Paged attention
- Best for large-scale production

**SGLang-JAX**:
- RadixCache for prefix sharing
- Good for similar prompts
- TPU-optimized

### Hyperparameter Starting Points
**PPO**:
- `epsilon`: 0.1-0.3
- `gae_lambda`: 0.9-0.99
- `gamma`: 0.99-1.0
- `beta`: 0.01-0.1
- `entropy_coef`: 0.001-0.01

**GRPO**:
- `num_generations`: 4-16
- `epsilon`: 0.1-0.3
- `beta`: 0.01-0.1

### Monitoring
**Key Metrics**:
- `rewards/sum`: Total episode rewards
- `advantages/mean`: Should be near 0
- `pg_clipfrac`: Clipping frequency (10-30% ideal)
- `kl`: KL divergence (should be small)
- `completions/mean_length`: Check for collapse

**Warning Signs**:
- KL diverging: Reduce learning rate or increase beta
- All advantages positive/negative: Reward scaling issue
- Clipfrac > 50%: Increase epsilon or reduce learning rate

```
