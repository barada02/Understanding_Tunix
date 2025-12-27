---
markmap:
  initialExpandLevel: 2
---

# Phase 2.3: Distillation Module

## 1. Module Overview
### Purpose and Scope
- Knowledge distillation from large teacher to small student
- Transfer learned representations
- Compress models while maintaining performance
- Multiple distillation strategies supported

### Key Files Structure
- `distillation_trainer.py` (145 lines) - Main trainer class
- `strategies/` - Distillation strategy implementations
  - `base_strategy.py` - Abstract base class
  - `logit.py` - Logit-based distillation
  - `attention.py` - Attention transfer
  - `feature_projection.py` - Feature projection with learnable layers
  - `feature_pooling.py` - Feature pooling without projection
- `feature_extraction/` - Feature extraction utilities
  - `projection.py` - Linear projection layers
  - `pooling.py` - Average pooling for shape matching
  - `sowed_module.py` - Intermediate output capturing

### Design Philosophy
- **Strategy Pattern**: Pluggable distillation strategies
- **Flexible Feature Extraction**: Capture any layer type
- **Model Preprocessing**: Temporary modifications for distillation
- **Minimal Changes**: Models restored after training
- **Protocol-Based**: ModelForwardCallable for type safety

## 2. DistillationTrainer - Main Orchestrator
### Class Overview
```python
class DistillationTrainer(peft_trainer.PeftTrainer):
    """Extends PeftTrainer with distillation capabilities"""
```

**Inheritance**: All PeftTrainer features + distillation
- Checkpoint management
- Metrics logging
- Progress bars
- Profiling
- Gradient accumulation

### Initialization
```python
def __init__(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
    strategy: strategies.BaseStrategy,
    optimizer: optax.GradientTransformation,
    training_config: TrainingConfig,
):
```

**Key Steps**:
1. Pre-process models with strategy (add feature extraction)
2. Initialize PeftTrainer with student model
3. Store teacher model and strategy
4. Set loss functions from strategy
5. Configure model input generation

**Model Pre-processing**:
```python
student_model, teacher_model = strategy.pre_process_models(
    student_model, teacher_model
)
```
- Wraps models with feature extraction hooks
- Adds projection layers if needed
- Prepares for intermediate output capture

### Training Input Structure
```python
@flax.struct.dataclass(frozen=True)
class TrainingInput(peft_trainer.TrainingInput):
    teacher_output: Any = None  # Cached teacher outputs
```

**Purpose**: Avoids recomputing teacher outputs

### Model Input Generation
```python
self.gen_model_input_fn = lambda x: {
    "inputs": {"input_tokens": x.input_tokens, "input_mask": x.input_mask},
    "teacher_output": x.teacher_output if hasattr(x, "teacher_output") else None,
}
```

**Customization**:
```python
trainer.with_gen_model_input_fn(
    lambda x: {
        "input_tokens": x.input_tokens,
        "attention_mask": x.attention_mask,
        # Custom preprocessing
    }
)
```

### Input Preparation Pipeline
```python
def _prepare_inputs(input_data: TrainingInput) -> TrainingInput:
    """Called before each forward pass"""
```

**Training Mode**:
1. Apply gen_model_input_fn to extract inputs
2. Call strategy.get_teacher_outputs (if teacher_output not cached)
3. Store teacher_output with input_tokens and input_mask
4. Return TrainingInput

**Evaluation Mode**:
- Skip teacher output computation (not needed for eval loss)
- Only compute student logits
- Use task loss only

### Loss Functions
#### Training Loss
```python
def get_train_loss(
    model: nnx.Module,
    teacher_output: Any,
    inputs: dict[str, ArrayLike],
) -> ArrayLike:
    return self.strategy.get_train_loss(model, teacher_output, inputs)
```

**Delegated to Strategy**: Each strategy implements its own loss

#### Evaluation Loss
```python
def get_eval_loss(
    model: nnx.Module,
    teacher_output: Any,  # Not used
    inputs: dict[str, ArrayLike],
) -> ArrayLike:
    return self.strategy.get_eval_loss(model, inputs)
```

**Task Loss Only**: Standard cross-entropy without teacher

### Cleanup and Post-Processing
```python
def close():
    super().close()
    self.model, self.teacher_model = self.strategy.post_process_models(
        self.model, self.teacher_model
    )
```

**Post-processing**:
- Remove feature extraction wrappers
- Remove projection layers
- Restore original model structure
- Clean up temporary modifications

### Restrictions
```python
def with_loss_fn(...):
    raise NotImplementedError(
        "with_loss_fn is not supported for distillation. "
        "Use the strategy to define the loss."
    )
```

**Why**: Loss function is tightly coupled with strategy

## 3. BaseStrategy - Abstract Interface
### Strategy Architecture
```python
class BaseStrategy(ABC):
    """Abstract base for all distillation strategies"""
    
    def __init__(
        student_forward_fn: ModelForwardCallable,
        teacher_forward_fn: ModelForwardCallable,
        labels_fn: Callable[..., jax.Array],
    ):
```

**Forward Functions**: Protocol-typed for flexibility
```python
class ModelForwardCallable(Protocol[R]):
    def __call__(self, model: nnx.Module, *args, **kwargs) -> R:
        ...
```

**JIT Compilation**: All forward functions JIT-compiled
```python
self._student_forward_fn = nnx.jit(student_forward_fn)
self._teacher_forward_fn = nnx.jit(teacher_forward_fn)
self._labels_fn = nnx.jit(labels_fn)
```

### Abstract Methods
#### Compute Loss (Training)
```python
@abstractmethod
def compute_loss(
    student_output: Any,
    teacher_output: Any,
    labels: jax.Array,
) -> jax.Array:
    """Combines distillation loss + task loss"""
```

**Typical Pattern**: `α * distillation_loss + (1-α) * task_loss`

#### Compute Eval Loss
```python
@abstractmethod
def compute_eval_loss(
    student_output: Any,
    labels: jax.Array,
) -> jax.Array:
    """Task loss only (cross-entropy)"""
```

### Model Lifecycle Hooks
#### Pre-processing
```python
def pre_process_models(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
) -> tuple[nnx.Module, nnx.Module]:
    """Prepare models for distillation (default: no-op)"""
    return student_model, teacher_model
```

**Use Cases**:
- Wrap with feature extraction modules
- Add projection layers
- Configure intermediate output capture

#### Post-processing
```python
def post_process_models(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
) -> tuple[nnx.Module, nnx.Module]:
    """Restore models after distillation (default: no-op)"""
    return student_model, teacher_model
```

**Use Cases**:
- Remove feature extraction wrappers
- Delete projection layers
- Clean up temporary state

### Output Extraction
#### Teacher Outputs
```python
def get_teacher_outputs(
    teacher_model: nnx.Module,
    inputs: dict[str, jax.Array],
) -> Any:
    """Computes and freezes teacher outputs"""
    output = self._teacher_forward_fn(teacher_model, **inputs)
    output = jax.lax.stop_gradient(output)  # No gradients for teacher
    return output
```

**stop_gradient**: Critical for efficiency
- Teacher parameters frozen
- No backprop through teacher
- Memory savings

#### Student Outputs
```python
def get_student_outputs(
    student_model: nnx.Module,
    inputs: dict[str, jax.Array],
) -> Any:
    """Computes student outputs (with gradients)"""
    return self._student_forward_fn(student_model, **inputs)
```

### Loss Computation Pipeline
#### Training Loss
```python
def get_train_loss(
    student_model: nnx.Module,
    teacher_output: Any,
    inputs: dict[str, jax.Array],
) -> jax.Array:
    """Complete training loss computation"""
    student_output = self.get_student_outputs(student_model, inputs)
    labels = self._labels_fn(**inputs)
    return self.compute_loss(
        student_output=student_output,
        teacher_output=teacher_output,
        labels=labels,
    )
```

#### Evaluation Loss
```python
def get_eval_loss(
    student_model: nnx.Module,
    inputs: dict[str, jax.Array],
) -> jax.Array:
    """Task loss only"""
    student_output = self.get_student_outputs(student_model, inputs)
    student_output = jax.lax.stop_gradient(student_output)
    labels = self._labels_fn(**inputs)
    return self.compute_eval_loss(
        student_output=student_output,
        labels=labels,
    )
```

## 4. Logit Distillation Strategy
### Overview
**Classic Knowledge Distillation** (Hinton et al., 2015)
- Distill soft targets from teacher logits
- Temperature-scaled softmax
- KL divergence loss
- Combined with task loss

**Key Paper**: "Distilling the Knowledge in a Neural Network"

### LogitStrategy Class
```python
class LogitStrategy(BaseStrategy):
    def __init__(
        student_forward_fn: ModelForwardCallable[jax.Array],
        teacher_forward_fn: ModelForwardCallable[jax.Array],
        labels_fn: Callable[..., jax.Array],
        temperature: float = 2.0,
        alpha: float = 0.5,
    ):
```

**Parameters**:
- `temperature`: Softens probability distributions (T > 1)
- `alpha`: Balance between distillation and task loss (0 to 1)

**Validation**:
```python
if temperature <= 0:
    raise ValueError("Temperature must be positive")
if not (0.0 <= alpha <= 1.0):
    raise ValueError("Alpha must be in [0, 1]")
```

### Temperature Scaling
**Purpose**: Soften probability distributions

**Low Temperature (T → 0)**: One-hot probabilities
```
Probabilities: [0.98, 0.01, 0.01]
```

**High Temperature (T = 5)**: Soft probabilities
```
Probabilities: [0.60, 0.25, 0.15]
```

**Why Useful**: Reveals relative confidences in wrong classes

### Loss Computation
```python
def compute_loss(
    student_output: jax.Array,  # Student logits
    teacher_output: jax.Array,  # Teacher logits
    labels: jax.Array,
) -> jax.Array:
```

**Step 1: Distillation Loss (KL Divergence)**
```python
# Soften with temperature
log_student_probs_temp = jax.nn.log_softmax(
    student_output / self.temperature, axis=-1
)
teacher_probs_temp = jax.nn.softmax(
    teacher_output / self.temperature, axis=-1
)

# KL divergence
kl_loss = optax.kl_divergence(log_student_probs_temp, teacher_probs_temp)

# Temperature scaling correction (from Hinton paper)
scaled_kl_loss = kl_loss * (self.temperature ** 2)
distillation_loss = jnp.mean(scaled_kl_loss)
```

**Why T² scaling**: Gradient magnitude correction
- Higher temperature → smaller gradients
- T² factor restores original gradient scale

**Step 2: Task Loss (Cross-Entropy)**
```python
ce_loss = optax.softmax_cross_entropy(
    logits=student_output,  # Original temperature
    labels=labels
)
task_loss = jnp.mean(ce_loss)
```

**Step 3: Combined Loss**
```python
combined_loss = (alpha * distillation_loss) + ((1.0 - alpha) * task_loss)
```

**Typical alpha values**:
- `0.5`: Equal weight
- `0.7-0.9`: Favor distillation (common for large teacher)
- `0.1-0.3`: Favor task loss (when labels high quality)

### Evaluation Loss
```python
def compute_eval_loss(
    student_output: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    """Standard cross-entropy (no teacher needed)"""
    ce_loss = optax.softmax_cross_entropy(
        logits=student_output,
        labels=labels
    )
    return jnp.mean(ce_loss)
```

### Example Usage
```python
# Define forward functions
def student_forward(model, input_tokens, input_mask):
    logits, _ = model(input_tokens, attention_mask=input_mask)
    return logits

def teacher_forward(model, input_tokens, input_mask):
    logits, _ = model(input_tokens, attention_mask=input_mask)
    return logits

def labels_fn(input_tokens, input_mask):
    # Next token prediction
    return jax.nn.one_hot(input_tokens[:, 1:], vocab_size)

# Create strategy
strategy = LogitStrategy(
    student_forward_fn=student_forward,
    teacher_forward_fn=teacher_forward,
    labels_fn=labels_fn,
    temperature=2.5,
    alpha=0.7,
)

# Create trainer
trainer = DistillationTrainer(
    student_model=small_model,
    teacher_model=large_model,
    strategy=strategy,
    optimizer=optax.adam(1e-4),
    training_config=TrainingConfig(max_steps=10000),
)
```

## 5. Feature-Based Strategies
### Overview
**Motivation**: Distill intermediate representations, not just outputs
- Capture how models process information
- Transfer layer-by-layer knowledge
- Better for architecturally different student/teacher

**Challenges**:
- Feature shape mismatch (student smaller than teacher)
- Layer alignment (which student layer → which teacher layer)
- Computational overhead

### Feature Extraction Infrastructure
#### Sowed Modules
**Purpose**: Capture intermediate layer outputs without modifying forward pass

```python
# Wrap specific layer types to capture outputs
feature_extraction.wrap_model_with_sowed_modules(
    model,
    [AttentionLayer]  # Capture all AttentionLayer instances
)

# Run forward pass (outputs automatically captured)
output = model(input_tokens, attention_mask)

# Retrieve captured features
features = feature_extraction.pop_sowed_intermediate_outputs(model)
# Returns: dict mapping layer paths to outputs

# Clean up after distillation
feature_extraction.unwrap_sowed_modules(model)
```

**Key Properties**:
- Minimal overhead
- Non-invasive (model structure unchanged)
- Stackable (multiple layer types)

### FeaturePoolingStrategy
#### Overview
**No Learnable Parameters**: Pools teacher features to match student size
- Uses average pooling for shape alignment
- Cosine distance loss (default)
- Attention transfer variant

**Use Case**: When student/teacher have different hidden dimensions

#### Class Definition
```python
class FeaturePoolingStrategy(BaseStrategy):
    def __init__(
        student_forward_fn: ModelForwardCallable[jax.Array],
        teacher_forward_fn: ModelForwardCallable[jax.Array],
        labels_fn: Callable[..., jax.Array],
        feature_layer: type[nnx.Module],  # Layer type to capture
        alpha: float = 0.75,
        feature_loss_fn: Callable[[jax.Array, jax.Array], jax.Array] | None = None,
        cosine_distance_axis: int | tuple[int, ...] = -1,
    ):
```

**Parameters**:
- `feature_layer`: Type of layer to capture (e.g., `AttentionLayer`)
- `alpha`: Weight for feature loss (higher = more distillation)
- `feature_loss_fn`: Custom loss or default cosine distance
- `cosine_distance_axis`: Axis for cosine computation

#### Model Pre-processing
```python
def pre_process_models(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
) -> tuple[nnx.Module, nnx.Module]:
    """Wrap both models with sowed modules"""
    feature_extraction.wrap_model_with_sowed_modules(
        student_model, [self.feature_layer]
    )
    feature_extraction.wrap_model_with_sowed_modules(
        teacher_model, [self.feature_layer]
    )
    return student_model, teacher_model
```

#### Feature Extraction
**Teacher Outputs**:
```python
def get_teacher_outputs(
    teacher_model: nnx.Module,
    inputs: dict[str, jax.Array],
) -> jax.Array:
    _ = self._teacher_forward_fn(teacher_model, **inputs)
    teacher_features = feature_extraction.pop_sowed_intermediate_outputs(
        teacher_model
    )
    # Stack all captured layers: [num_layers, batch, seq, hidden]
    return jnp.stack(jax.tree.leaves(teacher_features))
```

**Student Outputs**:
```python
def get_student_outputs(
    student_model: nnx.Module,
    inputs: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    student_logits = self._student_forward_fn(student_model, **inputs)
    student_features = feature_extraction.pop_sowed_intermediate_outputs(
        student_model
    )
    return {
        "logits": student_logits,
        "features": jnp.stack(jax.tree.leaves(student_features))
    }
```

#### Loss Computation
```python
def compute_loss(
    student_output: dict[str, jax.Array],
    teacher_output: jax.Array,
    labels: jax.Array,
) -> jax.Array:
```

**Step 1: Pool Teacher Features**
```python
student_features = student_output["features"]
teacher_features = feature_extraction.avg_pool_array_to_target_shape(
    teacher_output,
    student_features.shape  # Target shape
)
```

**Pooling Example**:
```
Teacher: [12 layers, 32 batch, 512 seq, 768 hidden]
Student: [6 layers, 32 batch, 512 seq, 384 hidden]

After pooling:
Teacher: [6 layers, 32 batch, 512 seq, 384 hidden]
```

**Step 2: Feature Loss**
```python
feature_loss = self.feature_loss_fn(student_features, teacher_features)

# Default: Cosine distance
# feature_loss = jnp.mean(
#     optax.cosine_distance(
#         student_features,
#         teacher_features,
#         axis=cosine_distance_axis
#     )
# )
```

**Step 3: Task Loss**
```python
student_logits = student_output["logits"]
ce_loss = optax.softmax_cross_entropy(logits=student_logits, labels=labels)
task_loss = jnp.mean(ce_loss)
```

**Step 4: Combine**
```python
combined_loss = (alpha * feature_loss) + ((1.0 - alpha) * task_loss)
```

#### Average Pooling Details
```python
def avg_pool_array_to_target_shape(
    input_array: jnp.ndarray,
    target_shape: tuple[int, ...],
    padding_mode: PaddingMode = PaddingMode.VALID,
) -> jnp.ndarray:
```

**VALID Padding**:
- Calculates window and stride to achieve exact target shape
- `stride = input_dim // output_dim`
- `window = input_dim - (output_dim - 1) * stride`

**SAME Padding**:
- Uses ceil division for stride
- `stride = ceil(input_dim / output_dim)`
- `window = stride`

**Error Handling**:
- Validates rank match
- Validates target dimensions feasible
- Ensures output shape exact

### FeatureProjectionStrategy
#### Overview
**Learnable Projection**: Linear layer maps student features to teacher dimension
- More flexible than pooling
- Learns optimal alignment
- Better for large dimension gaps

**Use Case**: When student much smaller than teacher

#### Class Definition
```python
class FeatureProjectionStrategy(BaseStrategy):
    def __init__(
        student_forward_fn: ModelForwardCallable[jax.Array],
        teacher_forward_fn: ModelForwardCallable[jax.Array],
        labels_fn: Callable[..., jax.Array],
        feature_layer: type[nnx.Module],
        dummy_input: dict[str, jax.Array],  # For shape inference
        rngs: nnx.Rngs,  # For projection layer init
        alpha: float = 0.75,
        feature_loss_fn: Callable | None = None,  # Default: MSE
    ):
```

**Key Difference from Pooling**: Requires dummy input for shape inference

#### ModelWithFeatureProjection
**Wrapper Class**:
```python
class ModelWithFeatureProjection(nnx.Module):
    def __init__(
        model: nnx.Module,
        feature_shape: int | tuple[int, ...],
        feature_target_shape: int | tuple[int, ...],
        rngs: nnx.Rngs,
    ):
        self.model = model
        self.projection_layer = nnx.LinearGeneral(
            feature_shape,
            feature_target_shape,
            axis=np.arange(len(feature_shape)),
            rngs=rngs,
        )
    
    def __call__(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        features = pop_sowed_intermediate_outputs(self.model)
        features = jnp.stack(jax.tree.leaves(features))
        projected_features = self.projection_layer(features)
        return output, projected_features
```

**LinearGeneral**: Projects all feature dimensions
- Flexible axis specification
- Batched matrix multiplication
- Efficient GPU/TPU execution

#### Model Pre-processing
```python
def pre_process_models(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
) -> tuple[nnx.Module, nnx.Module]:
    return setup_models_with_feature_projection(
        student_model,
        teacher_model,
        student_layer_to_capture=self.feature_layer,
        teacher_layer_to_capture=self.feature_layer,
        dummy_student_input=self.dummy_input,
        dummy_teacher_input=self.dummy_input,
        rngs=self.rngs,
    )
```

**setup_models_with_feature_projection**:
1. Wrap both models with sowed modules
2. Run dummy forward pass to get feature shapes
3. Create projection layer matching shapes
4. Wrap student in ModelWithFeatureProjection
5. Return modified models

**Shape Inference Example**:
```python
# After dummy pass:
student_features.shape = [6, 32, 512, 256]  # [layers, batch, seq, hidden]
teacher_features.shape = [12, 32, 512, 1024]

# Create projection: [6, 32, 512, 256] → [12, 32, 512, 1024]
projection_layer = LinearGeneral(
    (6, 32, 512, 256),
    (12, 32, 512, 1024),
    axis=(0, 1, 2, 3),
)
```

#### Student Output Extraction
```python
def get_student_outputs(
    student_model: nnx.Module,  # ModelWithFeatureProjection
    inputs: dict[str, jax.Array],
) -> dict[str, jax.Array]:
    # Automatically projects features
    student_logits, student_features = self._student_forward_fn(
        student_model, **inputs
    )
    return {"logits": student_logits, "features": student_features}
```

**Note**: Features already projected to teacher shape

#### Loss Computation
```python
def compute_loss(
    student_output: dict[str, jax.Array],
    teacher_output: jax.Array,
    labels: jax.Array,
) -> jax.Array:
    
    student_features = student_output["features"]  # Already projected!
    
    # Default: MSE loss
    feature_loss = self.feature_loss_fn(student_features, teacher_output)
    # feature_loss = jnp.mean(jnp.square(student_features - teacher_output))
    
    ce_loss = optax.softmax_cross_entropy(
        logits=student_output["logits"],
        labels=labels
    )
    task_loss = jnp.mean(ce_loss)
    
    combined_loss = (alpha * feature_loss) + ((1.0 - alpha) * task_loss)
    return combined_loss
```

**MSE vs Cosine Distance**:
- MSE: Exact matching (magnitude + direction)
- Cosine: Direction matching only
- MSE typical for projection strategies

#### Post-processing
```python
def post_process_models(
    student_model: nnx.Module,
    teacher_model: nnx.Module,
) -> tuple[nnx.Module, nnx.Module]:
    if isinstance(student_model, ModelWithFeatureProjection):
        student_model = student_model.model  # Unwrap
    unwrap_sowed_modules(student_model)
    unwrap_sowed_modules(teacher_model)
    return student_model, teacher_model
```

**Critical**: Removes projection layer from final model

### Attention Transfer Strategy
#### AttentionTransferStrategy
**Specialized FeaturePoolingStrategy** for attention maps

```python
class AttentionTransferStrategy(FeaturePoolingStrategy):
    def __init__(
        student_forward_fn: ModelForwardCallable[jax.Array],
        teacher_forward_fn: ModelForwardCallable[jax.Array],
        labels_fn: Callable[..., jax.Array],
        attention_layer: type[nnx.Module],  # Attention layer type
        alpha: float = 0.75,
        attention_loss_fn: Callable | None = None,  # Default: cosine
    ):
        super().__init__(
            student_forward_fn,
            teacher_forward_fn,
            labels_fn,
            feature_layer=attention_layer,
            alpha=alpha,
            feature_loss_fn=attention_loss_fn,
        )
```

**Just a naming convenience**: Inherits all FeaturePoolingStrategy logic

#### AttentionProjectionStrategy
**Specialized FeatureProjectionStrategy** for attention maps

```python
class AttentionProjectionStrategy(FeatureProjectionStrategy):
    def __init__(
        student_forward_fn: ModelForwardCallable[jax.Array],
        teacher_forward_fn: ModelForwardCallable[jax.Array],
        labels_fn: Callable[..., jax.Array],
        attention_layer: type[nnx.Module],
        dummy_input: dict[str, jax.Array],
        rngs: nnx.Rngs,
        alpha: float = 0.75,
        attention_loss_fn: Callable | None = None,  # Default: MSE
    ):
        super().__init__(
            student_forward_fn,
            teacher_forward_fn,
            labels_fn,
            feature_layer=attention_layer,
            dummy_input=dummy_input,
            rngs=rngs,
            alpha=alpha,
            feature_loss_fn=attention_loss_fn,
        )
```

**Just a naming convenience**: Inherits all FeatureProjectionStrategy logic

**Key Papers**:
- "Paying More Attention to Attention" (Zagoruyko & Komodakis, 2016)
- Attention maps encode relationships between tokens
- Transferring attention patterns transfers reasoning

## 6. Complete Usage Examples
### Logit Distillation Example
```python
from tunix.distillation import distillation_trainer
from tunix.distillation.strategies import logit
from tunix.models.automodel import AutoModel
import optax

# Load teacher and student models
teacher_model = AutoModel.from_pretrained("google/gemma-2-9b")
student_model = AutoModel.from_pretrained("google/gemma-2-2b")

# Define forward functions
def forward_fn(model, input_tokens, input_mask):
    logits, _ = model(input_tokens, attention_mask=input_mask, cache=None)
    return logits[:, :-1, :]  # All but last token

def labels_fn(input_tokens, input_mask):
    # Next token prediction
    labels = jax.nn.one_hot(
        input_tokens[:, 1:],  # Shifted right
        num_classes=vocab_size
    )
    return labels

# Create strategy
strategy = logit.LogitStrategy(
    student_forward_fn=forward_fn,
    teacher_forward_fn=forward_fn,
    labels_fn=labels_fn,
    temperature=2.0,
    alpha=0.7,  # 70% distillation, 30% task loss
)

# Create trainer
trainer = distillation_trainer.DistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    strategy=strategy,
    optimizer=optax.adamw(learning_rate=1e-4, weight_decay=0.01),
    training_config=distillation_trainer.TrainingConfig(
        max_steps=50000,
        checkpoint_root_directory="checkpoints/distill",
        eval_every_n_steps=1000,
        save_checkpoint_every_n_steps=5000,
    ),
)

# Train
train_data = load_dataset()
eval_data = load_eval_dataset()
trainer.train(train_data, eval_data)

# Close (removes temporary modifications)
trainer.close()
```

### Feature Pooling Example
```python
from tunix.distillation.strategies import feature_pooling

# Assume we have attention layer class
from tunix.models.gemma import AttentionLayer

strategy = feature_pooling.FeaturePoolingStrategy(
    student_forward_fn=forward_fn,
    teacher_forward_fn=forward_fn,
    labels_fn=labels_fn,
    feature_layer=AttentionLayer,  # Capture all AttentionLayer outputs
    alpha=0.8,  # High weight on feature matching
    cosine_distance_axis=-1,  # Compute cosine over hidden dim
)

trainer = distillation_trainer.DistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    strategy=strategy,
    optimizer=optax.adam(1e-4),
    training_config=config,
)

trainer.train(train_data, eval_data)
trainer.close()
```

### Feature Projection Example
```python
from tunix.distillation.strategies import feature_projection
from flax import nnx

# Create dummy input for shape inference
dummy_input = {
    "input_tokens": jnp.zeros((2, 128), dtype=jnp.int32),
    "input_mask": jnp.ones((2, 128), dtype=jnp.int32),
}

strategy = feature_projection.FeatureProjectionStrategy(
    student_forward_fn=forward_fn,
    teacher_forward_fn=forward_fn,
    labels_fn=labels_fn,
    feature_layer=AttentionLayer,
    dummy_input=dummy_input,
    rngs=nnx.Rngs(0),  # For projection layer initialization
    alpha=0.75,
    feature_loss_fn=None,  # Default MSE
)

trainer = distillation_trainer.DistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    strategy=strategy,
    optimizer=optax.adam(1e-4),
    training_config=config,
)

trainer.train(train_data, eval_data)
trainer.close()
```

### Custom Loss Function Example
```python
# Custom feature loss: L1 + cosine
def custom_feature_loss(student_features, teacher_features):
    l1_loss = jnp.mean(jnp.abs(student_features - teacher_features))
    cosine_loss = jnp.mean(
        optax.cosine_distance(student_features, teacher_features, axis=-1)
    )
    return 0.5 * l1_loss + 0.5 * cosine_loss

strategy = feature_pooling.FeaturePoolingStrategy(
    student_forward_fn=forward_fn,
    teacher_forward_fn=forward_fn,
    labels_fn=labels_fn,
    feature_layer=AttentionLayer,
    alpha=0.75,
    feature_loss_fn=custom_feature_loss,
)
```

### Multi-Layer Distillation
```python
# Capture multiple layer types
from tunix.models.gemma import AttentionLayer, FeedForwardLayer

# Need custom strategy or wrap both types
def pre_process_custom(student, teacher):
    wrap_model_with_sowed_modules(student, [AttentionLayer, FeedForwardLayer])
    wrap_model_with_sowed_modules(teacher, [AttentionLayer, FeedForwardLayer])
    return student, teacher

# Custom strategy inheriting from FeaturePoolingStrategy
class MultiLayerStrategy(FeaturePoolingStrategy):
    def pre_process_models(self, student, teacher):
        return pre_process_custom(student, teacher)
```

## 7. Best Practices
### Strategy Selection
**Logit Distillation**:
- ✅ Student and teacher same architecture
- ✅ Focus on final predictions
- ✅ Fastest training
- ✅ No extra memory for features
- ❌ Doesn't transfer intermediate knowledge

**Feature Pooling**:
- ✅ Different architectures okay
- ✅ Transfers layer-wise knowledge
- ✅ No learnable parameters (simpler)
- ✅ Good for attention transfer
- ❌ Pooling may lose information
- ❌ Slower than logit-only

**Feature Projection**:
- ✅ Large dimension gaps
- ✅ Learnable alignment
- ✅ Most flexible
- ❌ Extra parameters to train
- ❌ Requires dummy input
- ❌ Slowest training

### Hyperparameter Tuning
#### Temperature (Logit Distillation)
**Low (1.0-2.0)**:
- Sharper distributions
- Focus on top predictions
- Better for similar models

**High (3.0-10.0)**:
- Softer distributions
- Reveals more knowledge
- Better for very different models

**Rule of Thumb**: Start with 2.0, increase if teacher much larger

#### Alpha (Distillation Weight)
**Low (0.1-0.3)**:
- Favor task loss
- Use when labels high quality
- Faster convergence

**Medium (0.5-0.7)**:
- Balanced approach
- Standard choice
- Good default

**High (0.8-0.95)**:
- Favor distillation
- Use when teacher very strong
- May overfit to teacher

**Rule of Thumb**: Start with 0.7, reduce if underfitting on eval

### Layer Selection (Feature-Based)
**Early Layers**:
- Low-level features
- Edges, textures
- Less useful for distillation

**Middle Layers**:
- High-level features
- Concepts, patterns
- **Best for distillation**

**Late Layers**:
- Task-specific features
- May be too specialized
- Use with caution

**Recommendation**: Distill attention layers from middle 1/3 to 2/3 of model

### Memory Optimization
**Teacher Model**:
- Keep in eval mode (no gradients)
- Use bfloat16 if possible
- Consider caching teacher outputs for small datasets

**Feature Extraction**:
- Only capture necessary layers
- Clear features after each batch
- Use stop_gradient aggressively

**Projection Layers**:
- Use smaller intermediate dimensions if possible
- Consider factorized projections (two small layers)

### Data Considerations
**Dataset Size**:
- Small (<10k): Higher alpha, cache teacher outputs
- Medium (10k-100k): Standard approach
- Large (>100k): Can use lower alpha, focus on efficiency

**Data Augmentation**:
- Apply same augmentation to student and teacher
- Consistency important for distillation

### Monitoring
**Key Metrics**:
- Task loss (student performance on labels)
- Distillation loss (student-teacher agreement)
- Combined loss
- Eval accuracy vs teacher accuracy

**Health Checks**:
- Distillation loss should decrease initially
- Task loss should not spike
- Student eval should approach teacher eval
- Gap narrowing over time indicates good distillation

**Warning Signs**:
- Distillation loss stuck: Increase learning rate or temperature
- Task loss increasing: Reduce alpha
- Student not improving: Check feature alignment

### Common Pitfalls
1. **Forgetting to call close()**: Leaves feature extraction wrappers
2. **Temperature too low**: Distributions too sharp, minimal transfer
3. **Alpha too high**: Student overfits to teacher, poor generalization
4. **Wrong layer types**: Capturing output logits as features
5. **Shape mismatches**: Not handling different sequence lengths
6. **Memory leaks**: Not clearing sowed outputs each batch

## 8. Advanced Topics
### Staged Distillation
**Progressive Compression**: Distill in stages
```python
# Stage 1: Large → Medium
trainer1 = DistillationTrainer(
    student_model=medium_model,
    teacher_model=large_model,
    ...
)
trainer1.train(data)

# Stage 2: Medium → Small
trainer2 = DistillationTrainer(
    student_model=small_model,
    teacher_model=medium_model,  # Previous student!
    ...
)
trainer2.train(data)
```

**Benefits**:
- Easier optimization
- Better final performance
- Can use different strategies per stage

### Multi-Teacher Distillation
**Ensemble Teachers**: Average multiple teacher outputs
```python
def ensemble_teacher_forward(models, **inputs):
    outputs = [model(**inputs) for model in models]
    return jnp.mean(jnp.stack(outputs), axis=0)

strategy = LogitStrategy(
    student_forward_fn=forward_fn,
    teacher_forward_fn=lambda model, **inputs: 
        ensemble_teacher_forward(teacher_models, **inputs),
    ...
)
```

### Self-Distillation
**Student as Own Teacher**: Distill from earlier checkpoint
```python
# Save checkpoint at step 10k
# Load as teacher
# Continue training with distillation

# Benefits: Smooths optimization, reduces overfitting
```

### Task-Specific Distillation
**Different Tasks**: Teacher and student on related tasks
- Teacher: Pre-trained on broad task
- Student: Fine-tuned on narrow task
- Distill during student fine-tuning

```
