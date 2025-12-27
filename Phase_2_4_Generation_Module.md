# Phase 2.4 - Generation Module

```markmap
---
markmap:
  initialExpandLevel: 2
---

# Phase 2.4: Generation Module

## Overview
### Purpose
- **Autoregressive text generation** - Implements sampling-based decoding for LLMs
- **Multiple backends** - Supports vanilla JAX, vLLM, and SGLang-JAX
- **Flexible sampling** - Greedy, top-p, top-k, and beam search strategies
- **Efficient inference** - KV cache management and compilation optimizations

### Module Location
- **Path**: `tunix/generate/`
- **Files**: 9 files total
  - `base_sampler.py` - Abstract interface
  - `sampler.py` - Vanilla JAX implementation (806 lines)
  - `vllm_sampler.py` - vLLM integration (453 lines)
  - `sglang_jax_sampler.py` - SGLang-JAX integration (259 lines)
  - `beam_search.py` - Beam search algorithm (276 lines)
  - `tokenizer_adapter.py` - Tokenizer wrapper (283 lines)
  - `utils.py` - Helper functions (859 lines)
  - `mappings.py` - Weight mapping utilities
  - `vllm_async_driver.py` - Async vLLM support

### Design Philosophy
- **Backend abstraction** - Common interface across vanilla/vLLM/SGLang
- **State separation** - Separate graphdef from state for efficient compilation
- **Memory efficiency** - Optimized KV cache allocation and management
- **Production-ready** - Support for both research (vanilla) and deployment (vLLM/SGLang)

## BaseSampler Interface
### Abstract Base Class
```python
class BaseSampler(ABC):
    @property
    @abstractmethod
    def transformer(self) -> nnx.Module:
        """Returns the transformer module used by the sampler."""
    
    @property
    @abstractmethod
    def transformer_state(self) -> statelib.State:
        """Returns the transformer state used by the sampler."""
    
    @abstractmethod
    def __call__(
        self,
        input_strings: List[str],
        max_generation_steps,
        max_prompt_length=None,
        temperature=0.0,
        top_p=None,
        top_k=None,
        beam_size=None,
        seed=None,
        multi_sampling: int = 1,
        return_logits: bool = True,
        echo: bool = False,
        pad_output: bool = False,
    ):
        """Returns a list of generated samples for the input strings."""
    
    @abstractmethod
    def tokenize(self, input_string: str) -> np.ndarray | list[int]:
        """Returns the tokenized the input string."""
```

### SamplerOutput Dataclass
```python
@dataclasses.dataclass
class SamplerOutput:
    # Decoded samples from the model
    text: list[str]
    
    # Per-step logits used during sampling
    logits: Optional[list[jax.Array] | jax.Array]
    
    # Tokens corresponding to the generated samples
    tokens: list[np.ndarray] | np.ndarray
    
    # Left padded prompt tokens
    padded_prompt_tokens: np.ndarray
    
    # Log probabilities (optional)
    logprobs: Optional[list[float]]
```

### Design Rationale
- **Protocol-based design** - Allows different backends to implement the same interface
- **Unified output format** - All samplers return `SamplerOutput` for consistency
- **Flexibility** - Parameters support both research and production use cases
- **Type hints** - Clear contracts for implementers

## Vanilla JAX Sampler
### Class Overview
- **File**: `tunix/generate/sampler.py`
- **Purpose**: Pure JAX implementation for research and experimentation
- **Key Feature**: Full control over sampling logic and compilation

### Architecture
```python
class Sampler(base_sampler.BaseSampler):
    def __init__(
        self,
        transformer: nnx.Module,
        tokenizer: Any,
        cache_config: CacheConfig,
    ):
        # Wrap tokenizer in adapter
        self.tokenizer = TokenizerAdapter(tokenizer)
        
        # Separate graphdef from state for compilation efficiency
        self._transformer_graphdef, transformer_state = nnx.split(transformer)
        
        # Flatten state for efficient passing to JIT functions
        self._flattened_transformer_state = jax.tree.leaves(transformer_state)
        
        # JIT compile decode and prefill functions
        self._compiled_decode_fn = jax.jit(self._decode_fn)
        self._compiled_prefill_fn = jax.jit(self._prefill_fn)
```

### State Management
#### SamplingState Dataclass
```python
@flax.struct.dataclass
class _SamplingState:
    # Core generation state
    decoding_step: int                    # Current position in sequence
    token_buffer: jax.Array               # [B, max_len] accumulates outputs
    positions: jax.Array                  # Position indices for attention
    cache: dict[str, dict[str, Array]]    # KV cache for all layers
    done: jax.Array                       # [B] per-sequence completion flags
    
    # Configuration
    total_sampling_steps: int             # Max length (prompt + generation)
    num_input_tokens: int                 # Prompt length
    
    # Sampling parameters
    temperature: float                     # Sampling temperature
    sampling_mode: str                     # 'greedy', 'top_p', or 'beam_search'
    sampling_parameters: dict              # {top_p, top_k} or {beam_size}
    
    # Optional features
    logits_buffer: Optional[jax.Array]     # [B, max_len, vocab] for tracking
    forbidden_token_ids: Optional[tuple]   # Constrained generation
    seed: jax.random.PRNGKey               # Random seed for sampling
    
    # Beam search state (optional)
    beam_search_sampling_state: Optional[_BeamSearchSamplingState]
```

#### CacheConfig Dataclass
```python
@dataclasses.dataclass
class CacheConfig:
    cache_size: int        # Maximum sequence length
    num_layers: int        # Number of transformer layers
    num_kv_heads: int      # Number of KV heads (for GQA)
    head_dim: int          # Dimension per head
```

### Key Methods
#### 1. Initialization
```python
def init_sample_state(
    self,
    input_ids: jax.Array,           # [B, prompt_len]
    include_logits: bool = False,
    total_sampling_steps: int,
    forbidden_token_ids: Optional[tuple] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: jax.random.PRNGKey,
    beam_size: Optional[int] = None,
) -> _SamplingState:
    """Initialize sampling state with buffers and cache."""
    
    # Create token buffer filled with pad tokens
    token_buffer = jnp.full(
        (batch_size, total_sampling_steps),
        self.tokenizer.pad_id(),
        dtype=jnp.int32
    )
    
    # Set input tokens and masks
    token_buffer = token_buffer.at[:, :num_input_tokens].set(input_ids)
    
    # Initialize KV cache (either via model.init_cache or deprecated _init_cache)
    cache = transformer.init_cache(batch_size, self.cache_config.cache_size)
    
    # Detect sampling mode from parameters
    if beam_size is not None:
        sampling_mode = 'beam_search'
        sampling_parameters = {'beam_size': beam_size}
    elif top_p is not None or top_k is not None:
        sampling_mode = 'top_p'
        sampling_parameters = {'top_p': top_p, 'top_k': top_k}
    else:
        sampling_mode = 'greedy'
        sampling_parameters = {}
    
    return _SamplingState(...)
```

#### 2. Tokenization
```python
def tokenize(self, input_string: str) -> jax.Array:
    """Encode string and handle BOS token deduplication."""
    input_ids = self.tokenizer.encode(input_string)
    
    # Add BOS if not present
    bos_tok = (
        [self.tokenizer.bos_id()]
        if input_ids[0] != self.tokenizer.bos_id()
        else []
    )
    
    return jnp.array(bos_tok + input_ids)
```

#### 3. Prefill Phase
```python
def _prefill_fn(self, params, sampler_state: _SamplingState):
    """Initial forward pass through entire prompt."""
    
    # Extract input tokens dynamically
    input_tokens = jax.lax.dynamic_slice(
        sampler_state.token_buffer,
        (0, 0),
        (batch_size, sampler_state.num_input_tokens)
    )
    
    # Create causal attention mask
    input_mask = input_tokens != self.tokenizer.pad_id()
    attention_mask = utils.make_causal_attn_mask(
        input_mask, self.cache_config.cache_size
    )
    
    # Forward pass through transformer
    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        input_tokens,
        sampler_state.positions[:, :num_input_tokens],
        sampler_state.cache,
        attention_mask,
    )
    
    # Update logits buffer if tracking
    if sampler_state.logits_buffer is not None:
        logits_buffer = sampler_state.logits_buffer.at[
            :, :num_input_tokens
        ].set(logits)
    
    # Initialize beam search state if needed
    if sampler_state.sampling_mode == 'beam_search':
        beam_state, updated_args = beam_search_lib.init_batched_beam_state(...)
        # Update sampler_state with beam-expanded buffers
    
    # Sample first token
    updated_sampler_state = self._sample(
        logits=logits[:, -1, :],
        cache=cache,
        eos=self.eos_ids,
        sampler_state=sampler_state,
    )
    
    return updated_sampler_state
```

#### 4. Decode Phase
```python
def _decode_fn(self, params, sampler_state: _SamplingState):
    """Autoregressive generation loop."""
    
    def cond_fn(state):
        # Continue if not all sequences done and steps remain
        return jnp.logical_and(
            jnp.logical_not(jnp.all(state.done)),
            state.decoding_step < state.total_sampling_steps - 1
        )
    
    def body_fn(state):
        return self._sample_step(params, state)
    
    # While loop for autoregressive generation
    final_state = jax.lax.while_loop(
        cond_fn,
        body_fn,
        sampler_state
    )
    
    return final_state
```

#### 5. Single Decode Step
```python
def _sample_step(self, params, sampler_state: _SamplingState):
    """Single autoregressive decoding step."""
    
    decoding_step = sampler_state.decoding_step
    
    # Extract last generated token
    last_token = jax.lax.dynamic_slice(
        sampler_state.token_buffer,
        (0, decoding_step),
        (batch_size, 1)
    )
    
    # Prepare positions and attention mask
    step_positions = sampler_state.positions[:, decoding_step:decoding_step+1]
    input_mask = sampler_state.token_buffer != self.tokenizer.pad_id()
    attention_mask = utils.compute_attention_masks(
        decoding_step, self.cache_config.cache_size, input_mask
    )
    
    # Forward pass (single token)
    transformer = nnx.merge(self._transformer_graphdef, params)
    logits, cache = transformer(
        last_token,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )
    
    # Sample next token
    updated_sampler_state = self._sample(
        logits=logits,
        cache=cache,
        eos=self.eos_ids,
        sampler_state=sampler_state,
    )
    
    # Update logits buffer if tracking
    if updated_sampler_state.logits_buffer is not None:
        next_logits = jnp.squeeze(logits, 1)
        logits_buffer = updated_sampler_state.logits_buffer.at[
            :, decoding_step + 1
        ].set(next_logits)
    
    return dataclasses.replace(
        updated_sampler_state,
        logits_buffer=logits_buffer,
    )
```

#### 6. Token Sampling
```python
def _sample(
    self,
    logits: jax.Array,              # [B, vocab_size]
    cache: dict,
    eos: jax.Array,
    sampler_state: _SamplingState,
) -> _SamplingState:
    """Sample next tokens based on sampling mode."""
    
    # Apply forbidden token masking
    if sampler_state.forbidden_token_ids is not None:
        logits = logits.at[:, sampler_state.forbidden_token_ids].set(-jnp.inf)
    
    # Branch on sampling mode
    if sampler_state.sampling_mode == 'beam_search':
        # Beam search step
        beam_state, updated_args = beam_search_lib.beam_search_step(
            logits=logits,
            done=sampler_state.done,
            token_buffer=sampler_state.token_buffer,
            cache=cache,
            logits_buffer=sampler_state.logits_buffer,
            state=sampler_state.beam_search_sampling_state,
            pad_token_id=self.tokenizer.pad_id(),
            decoding_step=sampler_state.decoding_step,
        )
        # Update state with beam search results
        
    elif sampler_state.sampling_mode == 'greedy':
        # Greedy sampling (argmax)
        next_token = sample_best(logits)
        
    elif sampler_state.sampling_mode == 'top_p':
        # Top-p sampling with temperature
        next_token = sample_top_p(
            logits,
            sampler_state.temperature,
            sampler_state.sampling_parameters.get('top_p'),
            sampler_state.sampling_parameters.get('top_k'),
            sampler_state.seed,
        )
    
    # Update token buffer
    token_buffer = sampler_state.token_buffer.at[
        :, sampler_state.decoding_step + 1
    ].set(next_token)
    
    # Check for EOS tokens
    done = jnp.logical_or(
        sampler_state.done,
        jnp.isin(next_token, eos)
    )
    
    return dataclasses.replace(
        sampler_state,
        decoding_step=sampler_state.decoding_step + 1,
        token_buffer=token_buffer,
        cache=cache,
        done=done,
    )
```

#### 7. Main Entry Point
```python
def __call__(
    self,
    input_strings: str | Sequence[str],
    max_generation_steps: int,
    max_prompt_length: int | None = None,
    echo: bool = False,
    return_logits: bool = False,
    eos_tokens: Sequence[int] | None = None,
    forbidden_tokens: Sequence[str] | None = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    beam_size: Optional[int] = None,
    seed: int | None = None,
    pad_output: bool = False,
) -> base_sampler.SamplerOutput:
    """Generate completions for input prompts."""
    
    # Normalize inputs
    input_strings = (
        [input_strings] if isinstance(input_strings, str) else input_strings
    )
    
    # Convert forbidden tokens to IDs
    forbidden_token_ids = None
    if forbidden_tokens is not None:
        forbidden_token_ids = []
        for token in forbidden_tokens:
            token_id = self.tokenizer.encode(token)
            if len(token_id) != 1:
                raise ValueError('Forbidden tokens must map to single token ids')
            forbidden_token_ids.extend(token_id)
        forbidden_token_ids = tuple(forbidden_token_ids)
    
    # Tokenize and pad prompts
    tokens = [self.tokenize(x) for x in input_strings]
    max_tokens_length = max(len(x) for x in tokens)
    if max_prompt_length is None or max_prompt_length < max_tokens_length:
        max_prompt_length = utils.next_power_of_2(max_tokens_length)
    
    all_input_ids = np.array([
        utils.pad_to_length(
            x,
            target_length=max_prompt_length,
            pad_value=self.tokenizer.pad_id(),
            left=True,  # Left padding for prompts
        )
        for x in tokens
    ])
    
    # Validate total length
    total_sampling_steps = max_prompt_length + max_generation_steps
    if total_sampling_steps > self.cache_config.cache_size:
        raise ValueError(
            f'Total sampling steps {total_sampling_steps} must be less than '
            f'cache size {self.cache_config.cache_size}.'
        )
    
    # Initialize state
    seed = jax.random.PRNGKey(seed if seed is not None else 0)
    sampling_state = self.init_sample_state(
        jnp.array(all_input_ids),
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        seed=seed,
        beam_size=beam_size,
    )
    
    # Run prefill and decode
    sampling_state = self._compiled_prefill_fn(
        self._flattened_transformer_state, sampling_state
    )
    sampling_state = self._compiled_decode_fn(
        self._flattened_transformer_state, sampling_state
    )
    
    # Finalize beam search if used
    token_buffers = sampling_state.token_buffer
    logits_buffers = sampling_state.logits_buffer
    if sampling_state.sampling_mode == 'beam_search':
        updated_args = beam_search_lib.finalize_beam_search_state(
            sampling_state.beam_search_sampling_state,
            sampling_state.token_buffer,
            sampling_state.logits_buffer,
        )
        token_buffers = updated_args['token_buffer']
        logits_buffers = updated_args['logits_buffer']
    
    # Extract and decode tokens
    if pad_output:
        # Padded output path
        max_len = total_sampling_steps if echo else max_generation_steps
        lengths, out_tokens, out_logits = utils.padded_fill_tokens_and_logits(
            token_buffers, logits_buffers, return_logits, echo,
            self.tokenizer.pad_id(), self.eos_ids,
            max_prompt_length, max_len,
        )
        out_tokens, lengths = jax.device_get(out_tokens), jax.device_get(lengths)
        decoded_outputs = [
            self.tokenizer.decode(tokens[:length].tolist())
            for tokens, length in zip(out_tokens, lengths)
        ]
    else:
        # Unpadded output path
        out_tokens = []
        out_logits = []
        for i, token_buffer in enumerate(token_buffers):
            start_idx = (
                utils.find_first_non_pad_idx(token_buffer, self.tokenizer.pad_id())
                if echo
                else max_prompt_length
            )
            end_idx = (
                utils.find_first_eos_idx(
                    token_buffer[max_prompt_length:], self.eos_ids
                )
                + max_prompt_length
            )
            out_tokens.append(jax.device_get(token_buffer[start_idx:end_idx]))
            if return_logits:
                out_logits.append(logits_buffers[i][start_idx:end_idx])
        
        decoded_outputs = [
            self.tokenizer.decode(tokens.tolist()) for tokens in out_tokens
        ]
    
    return base_sampler.SamplerOutput(
        text=decoded_outputs,
        logits=out_logits if return_logits else [],
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=None,
    )
```

### Design Insights
- **Graphdef/State Separation** - Reduces HLO size and compilation time
- **JIT Compilation** - Prefill and decode functions are pre-compiled
- **While Loop** - JAX's `lax.while_loop` enables efficient autoregressive generation
- **Dynamic Slicing** - Extracts tokens/positions without recompilation
- **State Immutability** - Uses `dataclasses.replace` for functional updates

### Performance Considerations
- **Memory**: KV cache dominates memory usage (batch × layers × heads × cache_size × head_dim)
- **Compilation**: State flattening reduces compilation overhead
- **Padding**: Left-padding for prompts, power-of-2 lengths minimize recompilations

## Sampling Strategies
### Greedy Sampling
```python
def sample_best(logits: jax.Array) -> jax.Array:
    """Greedy sampling - select token with highest probability."""
    return jnp.argmax(logits, axis=-1)
```
- **Use Case**: Deterministic generation, highest probability path
- **Advantages**: Fast, reproducible, no randomness
- **Disadvantages**: Can be repetitive, less diverse

### Top-p (Nucleus) Sampling
```python
def _sample_top_p(
    logits: jax.Array,
    top_p: float,
    top_k: int,
    key: jax.random.PRNGKey
) -> jax.Array:
    """Top-p sampling with optional top-k filtering."""
    
    # Apply top-k if specified
    if top_k is not None:
        top_k_logits, top_k_indices = jax.lax.top_k(logits, k=top_k)
        probs = jax.nn.softmax(top_k_logits, axis=-1)
    else:
        probs = jax.nn.softmax(logits, axis=-1)
        top_k_indices = jnp.arange(logits.shape[-1])
    
    # Cumulative probability mass
    sorted_probs = jnp.sort(probs, axis=-1)[:, ::-1]
    cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)
    
    # Mask tokens beyond threshold
    mask = cumsum_probs > top_p
    mask = jnp.concatenate([
        jnp.zeros_like(mask[:, :1]),  # Always keep at least one token
        mask[:, :-1]
    ], axis=-1)
    
    # Apply mask and renormalize
    sorted_probs = jnp.where(mask, 0.0, sorted_probs)
    sorted_probs = sorted_probs / jnp.sum(sorted_probs, axis=-1, keepdims=True)
    
    # Sample from filtered distribution
    sampled_indices = jax.random.categorical(key, jnp.log(sorted_probs))
    
    return top_k_indices[sampled_indices]

def sample_top_p(
    logits: jax.Array,
    temperature: float,
    top_p: Optional[float],
    top_k: Optional[int],
    key: jax.random.PRNGKey
) -> jax.Array:
    """Apply temperature scaling and sample."""
    # Temperature scaling
    logits = logits / jnp.maximum(temperature, 1e-5)
    
    return _sample_top_p(logits, top_p, top_k, key)
```
- **Use Case**: Controlled diversity, creative generation
- **Parameters**:
  - `top_p`: Cumulative probability threshold (e.g., 0.9)
  - `top_k`: Optional pre-filtering (e.g., 50)
  - `temperature`: Controls randomness (higher = more random)
- **Advantages**: Balanced diversity and quality
- **Disadvantages**: Slower than greedy, still requires tuning

### Beam Search
- **Purpose**: Explore multiple candidate sequences simultaneously
- **Algorithm**: Maintains top-k hypotheses based on cumulative scores
- **See**: Detailed coverage in "Beam Search" section below
- **Use Case**: Structured generation, translation, summarization

### Parameter Selection Guide
| **Scenario** | **Strategy** | **Parameters** |
|--------------|--------------|----------------|
| Deterministic output | Greedy | `temperature=0.0` |
| Creative writing | Top-p | `temperature=0.8, top_p=0.9` |
| Code generation | Top-p | `temperature=0.2, top_p=0.95` |
| Factual answers | Top-p | `temperature=0.3, top_p=0.9, top_k=40` |
| Translation | Beam search | `beam_size=4` |

## KV Cache Management
### Purpose
- **Avoid Recomputation**: Cache key-value pairs from previous tokens
- **Memory Trade-off**: O(batch × layers × cache_size × heads × head_dim)
- **Speed Gain**: Only compute attention for current token during decode

### Cache Structure
```python
# Type aliases
LayerCache = dict[str, jax.Array]  # {'k': [...], 'v': [...], 'end_index': int}
Cache = dict[str, LayerCache]      # {'layer_0': {...}, 'layer_1': {...}, ...}

# Example cache for one layer
layer_cache = {
    'k': jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim)),
    'v': jnp.zeros((batch_size, cache_size, num_kv_heads, head_dim)),
    'end_index': 0  # Tracks how many positions are filled
}
```

### Initialization
```python
def _init_cache(
    batch_size: int,
    cache_size: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: jnp.dtype = jnp.bfloat16,
) -> Cache:
    """Creates empty KV cache for all layers."""
    cache = {}
    for i in range(num_layers):
        cache[f'layer_{i}'] = {
            'k': jnp.zeros(
                (batch_size, cache_size, num_kv_heads, head_dim),
                dtype=dtype
            ),
            'v': jnp.zeros(
                (batch_size, cache_size, num_kv_heads, head_dim),
                dtype=dtype
            ),
            'end_index': 0,
        }
    return cache
```

### Update During Prefill
```python
# During prefill, entire prompt is processed at once
# Cache is populated with keys/values for all prompt tokens
logits, cache = transformer(
    input_tokens,              # [B, prompt_len]
    positions,                 # [B, prompt_len]
    cache,                     # Updated in-place by transformer
    attention_mask,            # Causal mask
)
# After prefill:
# cache['layer_0']['k'][:, :prompt_len] = computed keys
# cache['layer_0']['end_index'] = prompt_len
```

### Update During Decode
```python
# During decode, only one token is processed per step
# Cache is appended with new key/value pair
logits, cache = transformer(
    last_token,                # [B, 1]
    step_positions,            # [B, 1]
    cache,                     # Incrementally updated
    attention_mask,            # Mask attends to all previous tokens
)
# After each decode step:
# cache['layer_0']['k'][:, step] = new key
# cache['layer_0']['end_index'] += 1
```

### Cache Configuration Trade-offs
```python
@dataclasses.dataclass
class CacheConfig:
    cache_size: int        # Max sequence length (higher = more memory)
    num_layers: int        # Model architecture (fixed)
    num_kv_heads: int      # GQA reduces this (fewer heads = less memory)
    head_dim: int          # Model architecture (fixed)
```

**Memory Calculation**:
```python
# Bytes per cache (bf16 = 2 bytes)
memory_per_cache = (
    batch_size * cache_size * num_layers * num_kv_heads * head_dim * 2
)

# Example: Llama-3-8B
# batch=8, cache=4096, layers=32, kv_heads=8, head_dim=128
# = 8 * 4096 * 32 * 8 * 128 * 2 = 17.2 GB
```

### Advanced: Grouped Query Attention (GQA)
- **Standard MHA**: num_kv_heads = num_query_heads (e.g., 32)
- **GQA**: num_kv_heads < num_query_heads (e.g., 8 vs 32)
- **Memory Savings**: 4x smaller KV cache in above example
- **Performance**: Minimal quality loss, significant memory reduction

## vLLM Sampler
### Overview
- **File**: `tunix/generate/vllm_sampler.py`
- **Purpose**: Production-grade inference with optimized memory management
- **Backend**: vLLM engine (supports JAX, TorchAX, TorchXLA)

### Configuration
```python
@dataclasses.dataclass
class VllmConfig:
    model_version: str                       # HF model identifier
    max_model_len: int                       # Maximum sequence length
    mesh: jax.sharding.Mesh                  # Device mesh
    hbm_utilization: float                   # Memory fraction (e.g., 0.9)
    init_with_random_weights: bool           # Fast bootstrap for weight sync
    tpu_backend_type: str                    # 'jax', 'torchax', or 'torchxla'
    mapping_config: MappingConfig            # Weight name mappings
    
    # Advanced options
    swap_space: float = 4.0                  # CPU swap space in GiB
    lora_config: Optional[Dict] = None       # LoRA adapter config
    server_mode: bool = False                # Async request handling
    async_scheduling: bool = False           # Async scheduling
    tensor_parallel_size: int = -1           # TP size (-1 = auto)
    data_parallel_size: int = -1             # DP size (-1 = auto)
    hf_config_path: Optional[Dict] = None    # Custom HF config
    additional_config: Optional[Dict] = None # Extra vLLM args
```

### Initialization
```python
class VllmSampler(base_sampler.BaseSampler):
    def __init__(self, tokenizer: Any, config: VllmConfig):
        # Set backend type
        os.environ["TPU_BACKEND_TYPE"] = config.tpu_backend_type
        
        # Enable data parallelism if needed
        if config.data_parallel_size > 1:
            os.environ["NEW_MODEL_DESIGN"] = "1"
        
        # Fast initialization with random weights
        if config.init_with_random_weights:
            os.environ["JAX_RANDOM_WEIGHTS"] = "1"
        
        self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
        self.config = config
        self.args = self._vllm_config(config)
        
        # Initialize vLLM engine
        if config.server_mode:
            self._driver = self._create_driver()  # Async driver
        else:
            self.llm = LLM(**self.args)  # Synchronous engine
```

### Weight Synchronization
```python
def update_params(
    self,
    updated_weights: jaxtyping.PyTree,
    filter_types: Optional[Tuple[Any, ...]] = None,
):
    """Sync weights from trainer to vLLM model."""
    
    if self.to_hf_key_mappings:
        # Mapped weight sync (e.g., Vanilla -> vLLM)
        utils.transfer_state_with_mappings(
            src_state=updated_weights,
            dst_state=self.transformer_state,
            key_mappings=self.to_hf_key_mappings,
            key_mapping_hook_fns=self.to_hf_hook_fns,
            transpose_keys=self.to_hf_transpose_keys,
            reshard_fn=reshard.reshard_pytree,
        )
    else:
        # Direct weight sync (e.g., MaxText -> MaxText)
        utils.transfer_state_directly(
            src_state=updated_weights,
            dst_state=self.transformer_state,
            reshard_fn=reshard.reshard_pytree,
        )
```

### Generation Interface
```python
def __call__(
    self,
    input_strings: List[str],
    max_generation_steps: int,
    max_prompt_length: int = None,
    temperature: float = 0.0,
    top_p: float = None,
    top_k: int = None,
    beam_size: int = None,
    seed: int = None,  # Not supported by vLLM JAX backend
    multi_sampling: int = 1,
    return_logits: bool = True,
    echo: bool = False,
    pad_output: bool = False,
) -> base_sampler.SamplerOutput:
    """Generate using vLLM engine."""
    
    # Configure sampling parameters
    if beam_size is not None:
        sampling_params = BeamSearchParams(
            beam_width=beam_size,
            max_tokens=max_generation_steps,
            temperature=temperature,
        )
    else:
        sampling_params = SamplingParams(
            max_tokens=max_generation_steps,
            n=multi_sampling,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            logprobs=1,
            prompt_logprobs=1,
            stop_token_ids=[self.tokenizer.eos_id()],
        )
    
    # Tokenize prompts
    prompt_ids = [self.tokenize(x) for x in input_strings]
    prompt_objects = [
        TokensPrompt(prompt_token_ids=ids) for ids in prompt_ids
    ]
    
    # Generate
    if self._driver is not None:
        outputs = self._generate_server_mode(prompt_objects, sampling_params)
    else:
        outputs = self.llm.generate(
            prompts=prompt_objects,
            sampling_params=sampling_params,
            use_tqdm=True,
        )
    
    # Decode outputs
    decoded_outputs, out_logprobs, out_tokens = self.detokenize(
        input_strings, outputs
    )
    
    return base_sampler.SamplerOutput(
        text=decoded_outputs[0],
        logits=None,  # vLLM doesn't expose logits directly
        tokens=out_tokens,
        padded_prompt_tokens=all_input_ids,
        logprobs=out_logprobs[0],
    )
```

### Server Mode (Async)
```python
def _generate_server_mode(
    self,
    prompts: List[TokensPrompt],
    sampling_params: Union[SamplingParams, BeamSearchParams],
) -> List[RequestOutput]:
    """Async request handling for server mode."""
    
    futures = []
    for idx, prompt in enumerate(prompts):
        request_id = str(next(self._request_counter))
        params = sampling_params.clone() if idx > 0 else sampling_params
        
        # Submit async request
        future = self._driver.submit_request(
            request_id=request_id,
            prompt=prompt,
            params=params,
        )
        futures.append(future)
    
    # Wait for all completions
    outputs = [future.result() for future in futures]
    return outputs
```

### Key Advantages
- **Optimized Memory**: Dynamic KV cache allocation, paging, CPU swap
- **Batching**: Continuous batching for high throughput
- **Production-Ready**: Async serving, request queuing, fault tolerance
- **Multi-Backend**: Supports JAX, PyTorch, TorchXLA

### When to Use
- **Production deployment** with high QPS requirements
- **Memory-constrained** environments (better than vanilla)
- **Online serving** with async request handling
- **Batch inference** with varying sequence lengths

## SGLang-JAX Sampler
### Overview
- **File**: `tunix/generate/sglang_jax_sampler.py`
- **Purpose**: SGLang-JAX integration for advanced caching strategies
- **Key Feature**: Radix cache for prefix sharing across requests

### Configuration
```python
@dataclasses.dataclass
class SglangJaxConfig:
    model_version: str                       # HF model identifier
    context_length: int                      # Maximum sequence length
    mesh: jax.sharding.Mesh                  # Device mesh
    mem_fraction_static: float               # Memory fraction for cache
    init_with_random_weights: bool           # Fast bootstrap
    disable_radix_cache: bool                # Disable prefix caching
    enable_deterministic_sampling: bool      # Deterministic RNG
    mapping_config: MappingConfig            # Weight mappings
    
    # Advanced options
    use_sort_for_toppk_minp: bool = True     # Top-p/min-p implementation
    precompile_bs_paddings: Optional[List] = None    # Batch sizes to precompile
    precompile_token_paddings: Optional[List] = None # Token lengths to precompile
    chunked_prefill_size: int = -1           # Chunked prefill (memory saving)
    page_size: int = 64                      # Page size for paged attention
```

### Initialization
```python
class SglangJaxSampler(base_sampler.BaseSampler):
    def __init__(self, tokenizer: Any, config: SglangJaxConfig):
        self.tokenizer = tok_adapter.TokenizerAdapter(tokenizer)
        self.args = self._sglang_jax_config(config)
        
        # Initialize SGLang-JAX engine
        self.engine = Engine(**self.args)
        
        # Store weight mappings
        self.mappings = config.mapping_config.to_hf_mappings
        self.to_hf_transpose_keys = config.mapping_config.to_hf_transpose_keys
        self.to_hf_hook_fns = config.mapping_config.to_hf_hook_fns
```

### Weight Synchronization
```python
def update_params(
    self,
    updated_weights: jaxtyping.PyTree,
    filter_types: Optional[Tuple[Any, ...]] = None,
):
    """Sync weights from trainer to SGLang-JAX model."""
    new_state = utils.transfer_state_with_mappings(
        src_state=updated_weights,
        dst_state=self.transformer_state,
        key_mappings=self.mappings,
        transpose_keys=self.to_hf_transpose_keys,
        reshard_fn=reshard.reshard_pytree,
    )
    
    # Update model state leaves
    new_model_state_leaves, _ = jax.tree_util.tree_flatten(new_state)
    self._model_runner.model_state_leaves = new_model_state_leaves
```

### Generation Interface
```python
def __call__(
    self,
    input_strings: List[str],
    max_generation_steps: int,
    max_prompt_length: int | None = None,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    beam_size: int | None = None,
    seed: Optional[Union[List[int], int]] = None,
    multi_sampling: int = 1,
    return_logits: bool = True,
    echo: bool = False,
    pad_output: bool = False,
) -> base_sampler.SamplerOutput:
    """Generate using SGLang-JAX engine."""
    
    # Get default sampling params and configure
    self.sampling_params = self.engine.get_default_sampling_params()
    self.sampling_params.max_new_tokens = max_generation_steps
    self.sampling_params.n = multi_sampling
    self.sampling_params.temperature = temperature
    self.sampling_params.stop_token_ids = [self.tokenizer.eos_id()]
    
    if top_p is not None:
        self.sampling_params.top_p = top_p
    if top_k is not None:
        self.sampling_params.top_k = top_k
    
    # Generate using engine
    outputs = self.engine.generate(
        input_strings,
        sampling_params=self.sampling_params,
    )
    
    # Process outputs...
    return base_sampler.SamplerOutput(...)
```

### Key Features
#### 1. Radix Cache
- **Purpose**: Share KV cache for common prefixes across requests
- **Example**: System prompts, few-shot examples
- **Memory Savings**: Avoid recomputing identical prompt prefixes
- **Use Case**: Chat applications, instruction-tuned models

#### 2. Chunked Prefill
- **Purpose**: Break long prompts into chunks to reduce memory spikes
- **Parameter**: `chunked_prefill_size` (tokens per chunk)
- **Trade-off**: Slightly slower prefill, much lower peak memory

#### 3. Paged Attention
- **Purpose**: Non-contiguous KV cache storage
- **Parameter**: `page_size` (default 64)
- **Benefit**: Better memory utilization, dynamic allocation

### When to Use
- **Prefix sharing** - Many requests with common prefixes
- **Chat applications** - System prompts reused across conversations
- **Memory constraints** - Need chunked prefill for long contexts
- **Advanced caching** - Radix cache for better throughput

### Comparison: vLLM vs SGLang-JAX
| **Feature** | **vLLM** | **SGLang-JAX** |
|-------------|----------|----------------|
| KV Cache | Paged attention | Paged + Radix cache |
| Prefix Sharing | Limited | Excellent |
| Async Serving | Yes | Yes |
| Maturity | More mature | Newer |
| Use Case | General serving | Prefix-heavy workloads |

## Tokenizer Adapter
### Purpose
- **Unified Interface**: Support SentencePiece, HuggingFace, and custom tokenizers
- **Type Detection**: Automatic tokenizer type detection
- **Method Normalization**: Common API across tokenizer types

### TokenizerType Enum
```python
class TokenizerType(enum.Enum):
    SP = 'sp'      # SentencePiece tokenizer
    HF = 'hf'      # HuggingFace tokenizer
    NONE = 'none'  # Custom tokenizer with required methods
```

### Initialization and Detection
```python
class TokenizerAdapter:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer
        
        # Detect tokenizer type
        if isinstance(self._tokenizer, spm.SentencePieceProcessor):
            self._tokenizer_type = TokenizerType.SP
        elif self._is_hf_tokenizer():
            self._tokenizer_type = TokenizerType.HF
        elif not self._missing_methods():
            self._tokenizer_type = TokenizerType.NONE
        else:
            raise ValueError(
                'Tokenizer must be SentencePiece, HuggingFace, or implement: '
                '["encode", "decode", "bos_id", "eos_id", "pad_id"]'
            )
    
    def _is_hf_tokenizer(self) -> bool:
        """Check if tokenizer is HuggingFace."""
        baseclasses = inspect.getmro(type(self._tokenizer))
        baseclass_names = [
            f"{bc.__module__}.{bc.__name__}" for bc in baseclasses
        ]
        return 'transformers.tokenization_utils_base.PreTrainedTokenizerBase' in baseclass_names
    
    def _missing_methods(self) -> list[str]:
        """Check for required methods in custom tokenizers."""
        required_methods = ['encode', 'decode', 'bos_id', 'eos_id', 'pad_id']
        return [m for m in required_methods if not hasattr(self._tokenizer, m)]
```

### Core Methods
```python
def encode(self, text: str, **kwargs) -> list[int]:
    """Encode text to token IDs."""
    if self._tokenizer_type == TokenizerType.SP:
        return self._tokenizer.EncodeAsIds(text, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
        return self._tokenizer.encode(text, **kwargs)
    else:
        return self._tokenizer.encode(text, **kwargs)

def decode(self, ids: list[int], **kwargs) -> str:
    """Decode token IDs to text."""
    if self._tokenizer_type == TokenizerType.SP:
        return self._tokenizer.DecodeIds(ids, **kwargs)
    elif self._tokenizer_type == TokenizerType.HF:
        return self._tokenizer.decode(ids, **kwargs)
    else:
        return self._tokenizer.decode(ids, **kwargs)

def bos_id(self) -> int:
    """Get BOS token ID."""
    if self._tokenizer_type == TokenizerType.SP:
        return self._tokenizer.bos_id()
    elif self._tokenizer_type == TokenizerType.HF:
        return self._tokenizer.bos_token_id
    else:
        return self._tokenizer.bos_id()

def eos_id(self) -> int:
    """Get EOS token ID."""
    if self._tokenizer_type == TokenizerType.SP:
        return self._tokenizer.eos_id()
    elif self._tokenizer_type == TokenizerType.HF:
        return self._tokenizer.eos_token_id
    else:
        return self._tokenizer.eos_id()

def pad_id(self) -> int:
    """Get PAD token ID (with fallback logic)."""
    if self._tokenizer_type == TokenizerType.SP:
        ret_id = self._tokenizer.pad_id()
        if ret_id == -1:
            raise ValueError('SentencePiece tokenizer has undefined pad_id.')
        return ret_id
    elif self._tokenizer_type == TokenizerType.HF:
        # HuggingFace tokenizers may not have pad_id (e.g., Llama-3)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer.pad_token_id
    else:
        return self._tokenizer.pad_id()
```

### Chat Template Support
```python
def apply_chat_template(
    self,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
    tokenize: bool = False,
    **kwargs,
) -> str | list[int]:
    """Apply chat template for conversation formatting."""
    
    if self._tokenizer_type == TokenizerType.HF:
        return self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            **kwargs,
        )
    else:
        # Fallback: Gemma chat template
        return self._apply_gemma_chat_template(
            messages, add_generation_prompt, tokenize
        )

def _apply_gemma_chat_template(
    self,
    messages: list[dict[str, str]],
    add_generation_prompt: bool,
    tokenize: bool,
) -> str | list[int]:
    """Gemma-style chat template."""
    chat_str = ''
    for message in messages:
        role = message.get('role')
        content = message.get('content')
        if role == 'user':
            chat_str += f'<start_of_turn>user\n{content}<end_of_turn>\n'
        elif role == 'assistant':
            chat_str += f'<start_of_turn>model\n{content}<end_of_turn>\n'
    
    if add_generation_prompt:
        chat_str += '<start_of_turn>model\n'
    
    return self.encode(chat_str) if tokenize else chat_str
```

### Helper Methods
```python
def dedup_bos_ids(self, ids: list[int]) -> list[int]:
    """Remove duplicate BOS tokens at the beginning."""
    i = 0
    while i < len(ids) - 1 and ids[i] == ids[i + 1] == self.bos_id():
        i += 1
    return ids[i:]
```

### Design Benefits
- **Flexibility**: Works with any tokenizer implementing required methods
- **Consistency**: Same API regardless of underlying tokenizer
- **Robustness**: Handles edge cases (missing pad_id, chat templates)
- **Extensibility**: Easy to add new tokenizer types

## Beam Search
### Algorithm Overview
- **Purpose**: Maintain multiple candidate sequences (beams) simultaneously
- **Scoring**: Select top-k sequences by cumulative log probability
- **File**: `tunix/generate/beam_search.py`

### Beam Search State
```python
@flax.struct.dataclass
class _BeamSearchSamplingState:
    # Accumulated scores (log sum probabilities) for each beam
    # Shape: [batch_size, beam_size]
    scores: jnp.ndarray
    
    # Flag indicating if state is initialized
    initialized: bool = flax.struct.field(pytree_node=False)
```

### Initialization
```python
def init_batched_beam_state(
    logits: jax.Array,                       # [B, vocab_size]
    input_token_buffer: jax.Array,           # [B, L]
    initial_cache: dict,                     # KV cache
    done: jax.Array,                         # [B, 1]
    positions: jax.Array,                    # [B, L]
    logits_buffer: jax.Array | None,        # [B, L, V]
    beam_size: int,
) -> tuple[_BeamSearchSamplingState, dict[str, Any]]:
    """Expand batch dimension by beam_size for beam search."""
    
    batch_size = input_token_buffer.shape[0]
    
    # Expand all states by repeating along batch dimension
    caches = jax.tree.map(
        lambda x: jnp.repeat(x, beam_size, axis=0),
        initial_cache
    )
    
    return _BeamSearchSamplingState(
        scores=jnp.zeros((batch_size, beam_size), dtype=jnp.float32),
        initialized=False,
    ), {
        "logits": jnp.repeat(logits, beam_size, axis=0),
        "token_buffer": jnp.repeat(input_token_buffer, beam_size, axis=0),
        "cache": caches,
        "done": jnp.repeat(done, beam_size, axis=0),
        "positions": jnp.repeat(positions, beam_size, axis=0),
        "logits_buffer": (
            jnp.repeat(logits_buffer, beam_size, axis=0)
            if logits_buffer is not None
            else None
        ),
    }
```

### Algorithm Flow
1. **Initialization**: Expand batch by `beam_size` (replicate inputs)
2. **First Step**: Select top `beam_size` tokens from vocabulary
3. **Subsequent Steps**:
   - Generate `beam_size * vocab_size` candidates
   - Compute cumulative scores (previous score + log prob)
   - Mask finished beams
   - Select top `beam_size` candidates
   - Update token buffers and caches
4. **Finalization**: Select best beam per batch

### Complexity Analysis
- **Time**: O(steps × beam_size × vocab_size) per batch
- **Memory**: O(beam_size) increase in KV cache and buffers
- **Trade-off**: Better quality vs. slower inference

### When to Use Beam Search
- **Structured tasks**: Translation, summarization
- **Deterministic output**: Want most likely sequence
- **Quality over speed**: Willing to trade compute for better results
- **Typical beam_size**: 4-8 (diminishing returns beyond 8)

## Usage Examples
### Example 1: Vanilla JAX Sampler (Basic)
```python
from flax import nnx
from tunix.generate import sampler as vanilla_sampler
from tunix.models import Gemma2

# Initialize model and tokenizer
model = Gemma2(config)
tokenizer = load_tokenizer("google/gemma-2-2b")

# Configure cache
cache_config = vanilla_sampler.CacheConfig(
    cache_size=2048,
    num_layers=model.config.num_layers,
    num_kv_heads=model.config.num_kv_heads,
    head_dim=model.config.head_dim,
)

# Create sampler
sampler = vanilla_sampler.Sampler(
    transformer=model,
    tokenizer=tokenizer,
    cache_config=cache_config,
)

# Generate (greedy sampling)
output = sampler(
    input_strings="Once upon a time",
    max_generation_steps=128,
    temperature=0.0,  # Greedy
)

print(output.text[0])
print(f"Tokens generated: {len(output.tokens[0])}")
```

### Example 2: Top-p Sampling with Temperature
```python
# Creative generation with top-p sampling
output = sampler(
    input_strings=[
        "Write a haiku about machine learning:",
        "Explain quantum computing:",
    ],
    max_generation_steps=256,
    temperature=0.8,     # Higher temperature for creativity
    top_p=0.9,           # Nucleus sampling
    top_k=50,            # Additional filtering
    return_logits=True,  # Track logits for analysis
)

for i, text in enumerate(output.text):
    print(f"\n=== Output {i+1} ===")
    print(text)
    print(f"Logit shape: {output.logits[i].shape}")
```

### Example 3: Beam Search
```python
# Translation with beam search
output = sampler(
    input_strings="Translate to French: Hello, how are you?",
    max_generation_steps=64,
    beam_size=4,         # Maintain 4 beams
    temperature=1.0,     # Standard temperature for beam search
)

print(output.text[0])
```

### Example 4: Constrained Generation
```python
# Avoid certain words during generation
output = sampler(
    input_strings="Generate a story without using:",
    max_generation_steps=200,
    forbidden_tokens=["and", "the"],  # Forbidden words
    temperature=0.7,
    top_p=0.95,
)

print(output.text[0])
```

### Example 5: vLLM Sampler (Production)
```python
from tunix.generate import vllm_sampler
from tunix.generate.mappings import MappingConfig
import jax

# Create device mesh
mesh = jax.sharding.Mesh(
    jax.devices(),
    axis_names=('data', 'model')
)

# Configure vLLM
vllm_config = vllm_sampler.VllmConfig(
    model_version="google/gemma-2-2b",
    max_model_len=4096,
    mesh=mesh,
    hbm_utilization=0.9,           # Use 90% of HBM for cache
    init_with_random_weights=True,  # Fast bootstrap
    tpu_backend_type="jax",
    mapping_config=MappingConfig(
        to_hf_mappings={}  # Add weight mappings if needed
    ),
    tensor_parallel_size=8,         # 8-way tensor parallelism
)

# Create vLLM sampler
vllm_sampler_instance = vllm_sampler.VllmSampler(
    tokenizer=tokenizer,
    config=vllm_config,
)

# Sync weights from trainer
vllm_sampler_instance.update_params(trained_weights)

# Generate
output = vllm_sampler_instance(
    input_strings=["Question: What is JAX?\nAnswer:"] * 16,  # Batch of 16
    max_generation_steps=256,
    temperature=0.3,
    top_p=0.95,
)

for text in output.text:
    print(text)
```

### Example 6: SGLang-JAX with Radix Cache
```python
from tunix.generate import sglang_jax_sampler

# Configure SGLang-JAX
sglang_config = sglang_jax_sampler.SglangJaxConfig(
    model_version="google/gemma-2-2b",
    context_length=4096,
    mesh=mesh,
    mem_fraction_static=0.8,
    init_with_random_weights=True,
    disable_radix_cache=False,      # Enable prefix sharing
    enable_deterministic_sampling=False,
    mapping_config=MappingConfig(...),
    chunked_prefill_size=512,       # Chunk long prompts
    page_size=64,
)

# Create sampler
sglang_sampler_instance = sglang_jax_sampler.SglangJaxSampler(
    tokenizer=tokenizer,
    config=sglang_config,
)

# Chat example with shared system prompt
system_prompt = "You are a helpful AI assistant."
conversations = [
    f"{system_prompt}\nUser: {query}\nAssistant:"
    for query in [
        "What is Python?",
        "Explain machine learning.",
        "What are neural networks?",
    ]
]

# Radix cache shares the common system prompt across requests
output = sglang_sampler_instance(
    input_strings=conversations,
    max_generation_steps=200,
    temperature=0.7,
    top_p=0.9,
)

for i, text in enumerate(output.text):
    print(f"\n=== Response {i+1} ===")
    print(text)
```

### Example 7: Chat Template Usage
```python
# Format conversation using chat template
messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What is its population?"},
]

# Apply chat template
chat_input = sampler.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=False,
)

# Generate response
output = sampler(
    input_strings=chat_input,
    max_generation_steps=128,
    temperature=0.5,
    top_p=0.9,
)

print(output.text[0])
```

### Example 8: Weight Updates During Training
```python
# During RL training loop
for step in range(num_steps):
    # Training step
    updated_weights = trainer.train_step()
    
    # Sync weights to sampler for rollout collection
    vllm_sampler_instance.update_params(updated_weights)
    
    # Collect rollouts with updated policy
    rollouts = vllm_sampler_instance(
        input_strings=prompts,
        max_generation_steps=512,
        temperature=0.8,
        top_p=0.95,
    )
    
    # Continue training...
```

## Best Practices
### 1. Choosing the Right Sampler
#### Vanilla JAX Sampler
- **Use When**:
  - Research and experimentation
  - Full control over sampling logic
  - Custom modifications needed
  - Small-scale inference
- **Avoid When**:
  - Production deployment at scale
  - Need optimized memory management
  - High throughput requirements

#### vLLM Sampler
- **Use When**:
  - Production deployment
  - High throughput required
  - Memory-constrained environments
  - Serving multiple users
  - Need continuous batching
- **Avoid When**:
  - Need custom sampling logic
  - Debugging model behavior
  - Research experiments

#### SGLang-JAX Sampler
- **Use When**:
  - Many requests with shared prefixes
  - Chat applications (system prompts)
  - Few-shot prompting
  - Need radix cache benefits
- **Avoid When**:
  - Prompts have no common prefixes
  - Simple single-request scenarios

### 2. Memory Management
#### Cache Size Selection
```python
# Calculate required cache size
cache_size = max_prompt_length + max_generation_steps

# Memory estimation (bf16)
memory_gb = (
    batch_size * cache_size * num_layers * 
    num_kv_heads * head_dim * 2  # bytes
) / (1024**3)

print(f"Estimated cache memory: {memory_gb:.2f} GB")
```

#### Padding Strategies
- **Left padding for prompts**: Aligns generation starts
- **Power-of-2 lengths**: Reduces recompilations
- **Dynamic padding**: Use `max_prompt_length=None` for auto-padding

#### Memory-Saving Techniques
1. **Reduce batch size**: Most direct impact
2. **Reduce cache_size**: Shorter sequences
3. **Use GQA models**: Fewer KV heads
4. **Enable CPU swap** (vLLM): `swap_space=4.0`
5. **Chunked prefill** (SGLang): `chunked_prefill_size=512`

### 3. Sampling Parameter Tuning
#### Temperature Guidelines
| **Temperature** | **Effect** | **Use Case** |
|-----------------|------------|--------------|
| 0.0 | Deterministic (greedy) | Factual QA, code generation |
| 0.1 - 0.3 | Low diversity | Technical writing, documentation |
| 0.5 - 0.7 | Balanced | General conversation |
| 0.8 - 1.0 | High diversity | Creative writing, brainstorming |
| 1.5+ | Very random | Experimental |

#### Top-p Guidelines
- **0.9**: Good default for most tasks
- **0.95**: More conservative, higher quality
- **0.85**: More focused, less repetition
- **Combine with top-k**: `top_p=0.9, top_k=50` for better control

#### Beam Search Guidelines
- **beam_size=1**: Equivalent to greedy
- **beam_size=4-6**: Good trade-off
- **beam_size=10+**: Diminishing returns, slower
- **Use for**: Translation, summarization, structured outputs

### 4. Performance Optimization
#### Compilation Optimization
```python
# Avoid recompilations
# Bad: Different prompt lengths trigger recompilation
for prompt in varying_length_prompts:
    output = sampler(prompt, max_generation_steps=128)

# Good: Pad to common length
max_len = max(len(tokenizer.encode(p)) for p in prompts)
max_len = next_power_of_2(max_len)  # Power of 2
outputs = sampler(
    prompts,
    max_generation_steps=128,
    max_prompt_length=max_len,
)
```

#### Batching
```python
# Process multiple requests together
batch_prompts = [
    "Prompt 1",
    "Prompt 2",
    "Prompt 3",
    # ...
]

# Single batched call (much faster than loop)
outputs = sampler(
    input_strings=batch_prompts,
    max_generation_steps=256,
    temperature=0.7,
)
```

#### JIT Warmup
```python
# Warmup sampler to avoid first-call compilation
dummy_output = sampler(
    input_strings="Warmup",
    max_generation_steps=8,
)
# Actual generation will be faster
```

### 5. Debugging and Monitoring
#### Track Logits
```python
output = sampler(
    input_strings=prompt,
    max_generation_steps=128,
    return_logits=True,  # Enable logit tracking
)

# Analyze logits
logits = output.logits[0]  # [num_tokens, vocab_size]
print(f"Logit shape: {logits.shape}")
print(f"Max logit: {logits.max()}")
print(f"Min logit: {logits.min()}")

# Check for degenerate distributions
probs = jax.nn.softmax(logits, axis=-1)
entropy = -jnp.sum(probs * jnp.log(probs + 1e-10), axis=-1)
print(f"Mean entropy: {entropy.mean()}")
```

#### Detect Repetition
```python
def detect_repetition(tokens, n=4):
    """Detect n-gram repetition."""
    for i in range(len(tokens) - 2*n):
        ngram = tuple(tokens[i:i+n])
        if ngram == tuple(tokens[i+n:i+2*n]):
            return True
    return False

if detect_repetition(output.tokens[0]):
    print("Warning: Repetition detected")
    # Consider adjusting temperature or top_p
```

#### Monitor Generation Speed
```python
import time

start = time.time()
output = sampler(
    input_strings=prompt,
    max_generation_steps=256,
)
elapsed = time.time() - start

tokens_generated = len(output.tokens[0])
throughput = tokens_generated / elapsed
print(f"Throughput: {throughput:.2f} tokens/sec")
```

### 6. Common Pitfalls
#### ❌ Don't: Exceed cache size
```python
# BAD: total_steps > cache_size
sampler(
    input_strings=long_prompt,  # 1500 tokens
    max_generation_steps=1000,  # 1500 + 1000 = 2500
)  # Error if cache_size = 2048
```

#### ✅ Do: Validate lengths
```python
prompt_length = len(tokenizer.encode(prompt))
total_steps = prompt_length + max_generation_steps
assert total_steps <= cache_config.cache_size, "Exceeds cache"
```

#### ❌ Don't: Use temperature=0 with top_p
```python
# BAD: Greedy + top_p is contradictory
sampler(prompt, temperature=0.0, top_p=0.9)
```

#### ✅ Do: Use consistent sampling parameters
```python
# Greedy
sampler(prompt, temperature=0.0)

# Top-p sampling
sampler(prompt, temperature=0.7, top_p=0.9)
```

#### ❌ Don't: Forget BOS/EOS tokens
```python
# BAD: Tokenizer might not add BOS automatically
tokens = tokenizer.encode(prompt)  # Missing BOS
```

#### ✅ Do: Use tokenizer adapter methods
```python
# GOOD: Adapter handles BOS/EOS correctly
tokens = sampler.tokenize(prompt)
```

### 7. Production Checklist
- [ ] Use vLLM or SGLang-JAX sampler
- [ ] Configure appropriate `hbm_utilization`
- [ ] Set up weight synchronization from trainer
- [ ] Enable async mode for serving
- [ ] Monitor memory usage and throughput
- [ ] Implement request batching
- [ ] Add error handling for OOM
- [ ] Set up logging and metrics
- [ ] Test with representative workloads
- [ ] Optimize padding and batch sizes

```
