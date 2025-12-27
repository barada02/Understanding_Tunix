---
markmap:
  initialExpandLevel: 2
---

# Phase 2.5: Models Module

## Overview
### Purpose
- **Model Factory** - Unified interface for loading pretrained models
- **Multi-Architecture Support** - Gemma, Llama, Qwen model families
- **Flexible Loading** - Support for Kaggle, HuggingFace, GCS, internal sources
- **Weight Management** - SafeTensors loading, checkpoint conversion, dummy initialization

### Module Location
- **Path**: `tunix/models/`
- **Core Files**:
  - `automodel.py` - AutoModel factory interface (439 lines)
  - `naming.py` - Model naming utilities (200 lines)
  - `safetensors_loader.py` - SafeTensors weight loading (234 lines)
  - `safetensors_saver.py` - SafeTensors weight saving
  - `dummy_model_creator.py` - Random weight initialization (100 lines)
- **Model Families**: `gemma/`, `gemma3/`, `llama3/`, `qwen2/`, `qwen3/`

### Design Philosophy
- **Dynamic Import** - Model modules loaded dynamically based on model name
- **Naming Convention** - Standardized naming system for all models
- **Backend Agnostic** - Works with different weight formats and sources
- **Sharding Support** - First-class support for distributed model sharding

### Supported Models
| **Family** | **Versions** | **Sizes** | **Features** |
|------------|--------------|-----------|--------------|
| Gemma | 1, 1.1, 2 | 2B, 7B, 9B, 27B | Sliding window attention (Gemma2) |
| Gemma3 | 3 | Various | Latest generation |
| Llama | 3, 3.1, 3.2 | 1B, 3B, 8B, 70B | GQA, weight tying (3.2) |
| Qwen | 2.5, 3 | 0.5B, 1.5B, 3B, 7B | Tied embeddings, MoE support |
| DeepSeek-R1-Distill | Qwen variant | 1.5B | Distilled reasoning model |

## AutoModel Interface
### Main Entry Point
```python
class AutoModel:
    """Factory class for instantiating Tunix models from pretrained checkpoints."""
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        mesh: jax.sharding.Mesh,
        *,
        model_source: ModelSource = ModelSource.HUGGINGFACE,
        model_path: str | None = None,
        model_download_path: str | None = None,
        **kwargs,
    ) -> tuple[nnx.Module, str | None]:
        """Load a pretrained model from a given identifier."""
```

### Model Sources
```python
class ModelSource(enum.Enum):
    KAGGLE = 'kaggle'        # Download from Kaggle (requires NNX conversion)
    GCS = 'gcs'              # Load from Google Cloud Storage
    HUGGINGFACE = 'huggingface'  # Load from HuggingFace Hub
    INTERNAL = 'internal'     # Load from internal storage
```

### Usage Pattern
```python
from tunix.models.automodel import AutoModel, ModelSource
import jax

# Create device mesh
mesh = jax.sharding.Mesh(
    jax.devices(),
    axis_names=('fsdp', 'tp')
)

# Load model from HuggingFace
model, model_path = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3.1-8B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
)

# Model is ready for training/inference
print(f"Loaded model from: {model_path}")
print(f"Model type: {type(model)}")
```

### Loading Flow
1. **Model Name Extraction**: Extract model name from model_id
2. **Download**: Download weights from specified source
3. **Config Selection**: Select appropriate ModelConfig based on model name
4. **Weight Loading**: Load weights via SafeTensors or custom loader
5. **Model Creation**: Instantiate NNX model with loaded weights
6. **Sharding**: Apply sharding based on mesh configuration

### Special Cases
#### Gemma Models from Kaggle
```python
# Gemma from Kaggle requires NNX conversion (slow)
model, path = AutoModel.from_pretrained(
    model_id="google/gemma-2b",
    mesh=mesh,
    model_source=ModelSource.KAGGLE,
    intermediate_ckpt_dir="/tmp/gemma_nnx_checkpoint",  # Cache converted checkpoint
    rng_seed=42,
)
```

#### Gemma3 Models from GCS
```python
# Gemma3 uses checkpoint-based loading
model, path = AutoModel.from_pretrained(
    model_id="google/gemma-3-2b",
    mesh=mesh,
    model_source=ModelSource.GCS,
    model_path="gs://my-bucket/gemma3-checkpoints",
)
```

#### Other Models from HuggingFace (Standard Path)
```python
# Most models use SafeTensors loading
model, path = AutoModel.from_pretrained(
    model_id="Qwen/Qwen2.5-0.5B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
    model_download_path="/tmp/model_cache",  # Optional cache directory
)
```

### Internal Helper Functions
#### Dynamic Module Import
```python
def get_model_module(model_name: str, module_type: ModelModule) -> Any:
    """Dynamically imports a model module."""
    model_config_category = naming.get_model_config_category(model_name)
    module_path = f'{_BASE_MODULE_PATH}.{model_config_category}.{module_type.value}'
    
    try:
        model_lib_module = importlib.import_module(module_path)
        return model_lib_module
    except ImportError:
        raise ImportError(f'Could not import module for {model_config_category}')
```

#### Config Calling
```python
def call_model_config(model_name: str) -> Any:
    """Dynamically calls a configuration function based on model_name."""
    config_id = naming.get_model_config_id(model_name)  # e.g., "llama3p1_8b"
    model_lib_module = get_model_module(model_name, ModelModule.MODEL)
    target_obj = model_lib_module.ModelConfig
    
    method_to_call = getattr(target_obj, config_id)  # e.g., ModelConfig.llama3p1_8b()
    return method_to_call()
```

## Model Naming System
### Naming Hierarchy
```
model_id (HuggingFace):  "meta-llama/Llama-3.1-8B"
                         ↓
model_name (internal):   "llama-3.1-8b" (lowercase, normalized)
                         ↓
┌────────────────────┬──────────────────────┐
model_family:        model_version:
"llama3p1"          "8b"
(standardized)      (standardized)
                         ↓
model_config_category:  "llama3" (module path)
model_config_id:        "llama3p1_8b" (method name)
```

### Naming Utilities
#### File: `naming.py`
```python
def get_model_name_from_model_id(model_id: str) -> str:
    """Extract model name from HuggingFace ID.
    
    Examples:
        "meta-llama/Llama-3.1-8B" → "llama-3.1-8b"
        "Qwen/Qwen2.5-0.5B" → "qwen2.5-0.5b"
        "google/gemma-2b" → "gemma-2b"
    """
    if '/' in model_id:
        model_name = model_id.split('/')[-1].lower()
        if model_name.startswith('meta-llama-'):
            return model_name.replace('meta-llama-', 'llama-', 1)
        return model_name
    else:
        raise ValueError(f'Invalid model ID format: {model_id}')

def split(model_name: str) -> tuple[str, str]:
    """Split model name into family and version.
    
    Examples:
        "llama3.1-8b" → ("llama3.1-", "8b")
        "gemma2-2b-it" → ("gemma2-", "2b-it")
        "qwen2.5-0.5b" → ("qwen2.5-", "0.5b")
    """
    model_name = model_name.lower()
    matched_family = ''
    
    # Find longest matching prefix from _MODEL_FAMILY_INFO_MAPPING
    for family in _MODEL_FAMILY_INFO_MAPPING:
        if model_name.startswith(family) and len(family) > len(matched_family):
            matched_family = family
    
    if matched_family:
        return matched_family, model_name[len(matched_family):].lstrip('-')
    else:
        raise ValueError(f'Unknown model family: {model_name}')

def get_model_family_and_version(model_name: str) -> tuple[str, str]:
    """Get standardized family and version.
    
    Standardization:
        - Lowercase
        - Replace '-' with '_'
        - Replace '.' with 'p'
    
    Examples:
        "llama-3.1-8b" → ("llama3p1", "8b")
        "qwen2.5-0.5b" → ("qwen2p5", "0p5b")
        "gemma2-2b-it" → ("gemma2", "2b_it")
    """
    raw_model_family, raw_model_version = split(model_name)
    model_family = _MODEL_FAMILY_INFO_MAPPING[raw_model_family].family
    model_version = _standardize_model_version(raw_model_version)
    return model_family, model_version

def get_model_config_category(model_name: str) -> str:
    """Get the model config category (module path).
    
    Examples:
        "gemma-2b" → "gemma"
        "gemma2-9b" → "gemma"  # Both use gemma/ directory
        "llama-3.1-8b" → "llama3"
        "qwen2.5-0.5b" → "qwen2"
    """
    raw_model_family, _ = split(model_name)
    return _MODEL_FAMILY_INFO_MAPPING[raw_model_family].config_category

def get_model_config_id(model_name: str) -> str:
    """Get the model config ID (method name).
    
    Examples:
        "llama-3.1-8b" → "llama3p1_8b"
        "qwen2.5-0.5b" → "qwen2p5_0p5b"
        "gemma2-2b-it" → "gemma2_2b_it"
    """
    model_family, model_version = get_model_family_and_version(model_name)
    config_id = f'{model_family}_{model_version}'
    config_id = config_id.replace('.', 'p').replace('-', '_')
    return config_id
```

### Model Family Mapping
```python
_MODEL_FAMILY_INFO_MAPPING = immutabledict.immutabledict({
    'gemma-': _ModelFamilyInfo(family='gemma', config_category='gemma'),
    'gemma1.1-': _ModelFamilyInfo(family='gemma1p1', config_category='gemma'),
    'gemma-1.1-': _ModelFamilyInfo(family='gemma1p1', config_category='gemma'),
    'gemma2-': _ModelFamilyInfo(family='gemma2', config_category='gemma'),
    'gemma-2-': _ModelFamilyInfo(family='gemma2', config_category='gemma'),
    'gemma3-': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'gemma-3-': _ModelFamilyInfo(family='gemma3', config_category='gemma3'),
    'llama3-': _ModelFamilyInfo(family='llama3', config_category='llama3'),
    'llama-3-': _ModelFamilyInfo(family='llama3', config_category='llama3'),
    'llama3.1-': _ModelFamilyInfo(family='llama3p1', config_category='llama3'),
    'llama-3.1-': _ModelFamilyInfo(family='llama3p1', config_category='llama3'),
    'llama3.2-': _ModelFamilyInfo(family='llama3p2', config_category='llama3'),
    'llama-3.2-': _ModelFamilyInfo(family='llama3p2', config_category='llama3'),
    'qwen2.5-': _ModelFamilyInfo(family='qwen2p5', config_category='qwen2'),
    'qwen3-': _ModelFamilyInfo(family='qwen3', config_category='qwen3'),
    'deepseek-r1-distill-qwen-': _ModelFamilyInfo(
        family='deepseek_r1_distill_qwen', config_category='qwen2'
    ),
})
```

## Model Configurations
### ModelConfig Pattern
All model families follow a common pattern:

```python
@dataclasses.dataclass
class ModelConfig:
    """Configuration for the model."""
    
    # Architecture parameters
    num_layers: int
    vocab_size: int  # or num_embed for Gemma
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int  # For Grouped Query Attention
    
    # Model-specific parameters
    norm_eps: float
    rope_theta: int  # RoPE base frequency
    
    # Optional features
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
    remat_config: RematConfig = RematConfig.NONE
    
    @classmethod
    def <model_family>_<version>(cls):
        """Factory method for specific model variant."""
        return cls(...)
```

### Example: Llama3 Configurations
```python
# tunix/models/llama3/model.py

@dataclasses.dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
    weight_tying: bool = False
    remat_config: RematConfig = RematConfig.NONE
    
    @classmethod
    def llama3p2_1b(cls):
        return cls(
            num_layers=16,
            vocab_size=128256,
            embed_dim=2048,
            hidden_dim=8192,
            num_heads=32,
            head_dim=64,
            num_kv_heads=8,
            norm_eps=1e-05,
            rope_theta=500_000,
            weight_tying=True,
        )
    
    @classmethod
    def llama3p2_3b(cls):
        return cls(
            num_layers=28,
            vocab_size=128256,
            embed_dim=3072,
            hidden_dim=8192,
            num_heads=24,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-05,
            rope_theta=500_000,
            weight_tying=True,
        )
    
    @classmethod
    def llama3p1_8b(cls):
        return cls(
            num_layers=32,
            vocab_size=128256,
            embed_dim=4096,
            hidden_dim=14336,
            num_heads=32,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-05,
            rope_theta=500_000,
            weight_tying=False,
        )
    
    @classmethod
    def llama3_70b(cls):
        return cls(
            num_layers=80,
            vocab_size=128256,
            embed_dim=8192,
            hidden_dim=28672,
            num_heads=64,
            head_dim=128,
            num_kv_heads=8,
            norm_eps=1e-05,
            rope_theta=500_000,
            weight_tying=False,
        )
```

### Example: Gemma2 Configurations
```python
# tunix/models/gemma/model.py

@dataclasses.dataclass(slots=True)
class ModelConfig:
    num_layers: int
    num_embed: int  # Vocabulary size
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    final_logit_softcap: float | None
    use_post_attn_norm: bool
    use_post_ffw_norm: bool
    attention_types: Iterable[AttentionType]
    attn_logits_soft_cap: float | None = None
    sliding_window_size: int | None = None
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
    remat_config: RematConfig = RematConfig.NONE
    
    @classmethod
    def gemma2_2b(cls):
        num_layers = 26
        return cls(
            num_layers=num_layers,
            num_embed=256128,
            embed_dim=2304,
            hidden_dim=9216,
            num_heads=8,
            head_dim=256,
            num_kv_heads=4,
            final_logit_softcap=30.0,
            attention_types=(
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            ) * int(num_layers / 2),  # Alternating pattern
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            attn_logits_soft_cap=50.0,
            sliding_window_size=4096,
        )
    
    @classmethod
    def gemma2_9b(cls):
        num_layers = 42
        return cls(
            num_layers=num_layers,
            num_embed=256128,
            embed_dim=3584,
            hidden_dim=28672,
            num_heads=16,
            head_dim=256,
            num_kv_heads=8,
            final_logit_softcap=30.0,
            attention_types=(
                AttentionType.LOCAL_SLIDING,
                AttentionType.GLOBAL,
            ) * int(num_layers / 2),
            use_post_attn_norm=True,
            use_post_ffw_norm=True,
            attn_logits_soft_cap=50.0,
            sliding_window_size=4096,
        )
```

### Example: Qwen2.5 Configurations
```python
# tunix/models/qwen2/model.py

@dataclasses.dataclass(slots=True)
class ModelConfig:
    num_layers: int
    vocab_size: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int
    rope_theta: int
    norm_eps: float
    use_tied_embedding: bool = False
    shd_config: ShardingConfig = ShardingConfig.get_default_sharding()
    remat_config: RematConfig = RematConfig.NONE
    
    @classmethod
    def qwen2p5_0p5b(cls):
        return cls(
            num_layers=24,
            vocab_size=151936,
            embed_dim=896,
            hidden_dim=4864,
            num_heads=14,
            head_dim=64,
            num_kv_heads=2,
            norm_eps=1e-06,
            rope_theta=1_000_000,
            use_tied_embedding=True,
        )
    
    @classmethod
    def deepseek_r1_distill_qwen_1p5b(cls):
        return cls(
            num_layers=28,
            vocab_size=151936,
            embed_dim=1536,
            hidden_dim=8960,
            num_heads=12,
            head_dim=128,
            num_kv_heads=2,
            norm_eps=1e-06,
            rope_theta=10000,  # Different from standard Qwen
            use_tied_embedding=False,
        )
```

### Sharding Configuration
```python
@dataclasses.dataclass(slots=True, frozen=True)
class ShardingConfig:
    """Sharding configuration for distributed training."""
    
    emb_vd: Tuple[str | None, ...]          # Embedding table
    q_weight_ndh: Tuple[str | None, ...]    # Query projection
    kv_weight_dnh: Tuple[str | None, ...]   # Key/Value projections
    o_weight_nhd: Tuple[str | None, ...]    # Output projection
    ffw_weight_df: Tuple[str | None, ...]   # FFN up projection
    ffw_weight_fd: Tuple[str | None, ...]   # FFN down projection
    rms_norm_weight: Tuple[str | None, ...]  # Layer norms
    act_btd: Tuple[str | None, ...]         # Activations (batch, time, dim)
    act_btf: Tuple[str | None, ...]         # Activations (batch, time, ffn)
    act_btnh: Tuple[str | None, ...]        # Activations (batch, time, heads, head_dim)
    
    @staticmethod
    def get_default_sharding(is_sampling: bool = False):
        fsdp = 'fsdp' if not is_sampling else None
        
        return ShardingConfig(
            emb_vd=('tp', fsdp),
            q_weight_ndh=(fsdp, 'tp', None),
            kv_weight_dnh=(fsdp, 'tp', None),
            o_weight_nhd=('tp', None, fsdp),
            ffw_weight_df=(fsdp, 'tp'),
            ffw_weight_fd=('tp', fsdp),
            rms_norm_weight=('tp',),
            act_btd=('fsdp', None, None if is_sampling else 'tp'),
            act_btf=('fsdp', None, 'tp'),
            act_btnh=('fsdp', None, 'tp', None),
        )
```

## SafeTensors Loader
### Purpose
- **Efficient Loading**: Memory-mapped file access for large models
- **Format Conversion**: PyTorch (SafeTensors) → JAX/Flax NNX
- **Weight Mapping**: Automatic key translation between frameworks
- **Sharding Support**: Load directly into sharded arrays

### File: `safetensors_loader.py`

#### Core Function
```python
def load_and_create_model(
    file_dir: str,
    model_class,
    config,
    key_mapping,
    mesh=None,
    preprocess_fn=None,
    dtype: jnp.dtype | None = None,
):
    """Loads safetensors files and creates an NNX model.
    
    Args:
        file_dir: Directory containing .safetensors files
        model_class: NNX model class to instantiate
        config: Model configuration object
        key_mapping: Function returning (torch_key → jax_key, transform) mapping
        mesh: Optional JAX device mesh for sharding
        preprocess_fn: Optional function to preprocess loaded state
        dtype: Optional dtype to cast loaded tensors to
    
    Returns:
        NNX model instance with loaded weights
    """
```

#### Loading Process
1. **Find SafeTensors Files**
```python
files = list(epath.Path(file_dir).expanduser().glob('*.safetensors'))
if not files:
    raise ValueError(f'No safetensors found in {file_dir}')
```

2. **Create Model Structure**
```python
with mesh:
    model = nnx.eval_shape(lambda: model_class(config, rngs=nnx.Rngs(params=0)))
graph_def, abs_state = nnx.split(model)
state_dict = abs_state.to_pure_dict()
```

3. **Get Sharding Info**
```python
if mesh is not None:
    sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
else:
    device = jax.devices()[0]
    sharding_dict = jax.tree.map(lambda _: device, state_dict)
```

4. **Load with Memory Mapping**
```python
def load_safetensors_with_offsets(filepath):
    """Load safetensors with memory-mapped access."""
    with open(filepath, 'rb') as f:
        # Read header
        header_size_bytes = f.read(8)
        header_size = struct.unpack('<Q', header_size_bytes)[0]
        header_bytes = f.read(header_size)
        header = json.loads(header_bytes.decode('utf-8'))
    
    # Memory map the data block
    f = open(filepath, 'rb')
    mm = mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ)
    contiguous_array = np.frombuffer(
        mm,
        dtype=to_np_dtype(common_dtype),
        count=total_elements,
        offset=data_block_start_offset_bytes,
    )
    
    return contiguous_array, tensor_metadata, mm, f
```

5. **Map and Transform Weights**
```python
key_map = key_mapping(config)

for array, metadata_list in arrays:
    for metadata in metadata_list:
        jax_key_mapped, transform = torch_key_to_jax_key(
            key_map, metadata['name']
        )
        
        # Extract tensor from memory-mapped array
        offset = metadata['offset_elements']
        size = metadata['size_elements']
        tensor = array[offset:offset + size].reshape(metadata['shape'])
        
        # Apply transformations (transpose, reshape)
        if transform:
            permute_rule, reshape_rule = transform
            if permute_rule:
                tensor = np.transpose(tensor, permute_rule)
            if reshape_rule:
                tensor = np.reshape(tensor, reshape_rule)
        
        # Shard and add to state_dict
        state_dict[jax_key_mapped] = jax.device_put(
            tensor, sharding_dict[jax_key_mapped]
        )
```

6. **Merge and Return**
```python
if preprocess_fn:
    state_dict = preprocess_fn(state_dict)

return nnx.merge(graph_def, state_dict)
```

### Key Mapping Pattern
Each model family defines a key mapping function:

```python
def _get_key_and_transform_mapping(cfg: ModelConfig):
    """Map PyTorch keys to JAX keys with optional transformations.
    
    Returns:
        dict mapping regex patterns to (jax_key, (permute, reshape))
    """
    return {
        # Pattern → (JAX key, (permute_axes, reshape_dims))
        
        r"model\.embed_tokens\.weight": (
            "embedder.input_embedding",
            None  # No transformation
        ),
        
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.attn.q_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim))
            # Transpose (D, H*D_h) → (H*D_h, D)
            # Then reshape → (D, H, D_h)
        ),
        
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.attn.o_proj.w",
            ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim))
            # Transpose and reshape for output projection
        ),
        
        # ... more mappings ...
    }
```

### Example: Llama3 Key Mapping
```python
# From tunix/models/llama3/params.py

def _get_key_and_transform_mapping(cfg: model_lib.ModelConfig):
    return {
        # Embeddings
        r"model\.embed_tokens\.weight": ("embedder.input_embedding", None),
        
        # Q, K, V projections
        r"model\.layers\.([0-9]+)\.self_attn\.q_proj\.weight": (
            r"layers.\1.attn.q_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.k_proj\.weight": (
            r"layers.\1.attn.k_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        r"model\.layers\.([0-9]+)\.self_attn\.v_proj\.weight": (
            r"layers.\1.attn.v_proj.w",
            ((1, 0), (cfg.embed_dim, cfg.num_kv_heads, cfg.head_dim)),
        ),
        
        # Output projection
        r"model\.layers\.([0-9]+)\.self_attn\.o_proj\.weight": (
            r"layers.\1.attn.o_proj.w",
            ((1, 0), (cfg.num_heads, cfg.head_dim, cfg.embed_dim)),
        ),
        
        # MLP
        r"model\.layers\.([0-9]+)\.mlp\.gate_proj\.weight": (
            r"layers.\1.mlp.gate_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.up_proj\.weight": (
            r"layers.\1.mlp.up_proj.kernel",
            ((1, 0), None),
        ),
        r"model\.layers\.([0-9]+)\.mlp\.down_proj\.weight": (
            r"layers.\1.mlp.down_proj.kernel",
            ((1, 0), None),
        ),
        
        # Norms
        r"model\.norm\.weight": ("final_norm.w", None),
        r"model\.layers\.([0-9]+)\.input_layernorm\.weight": (
            r"layers.\1.input_layernorm.w",
            None,
        ),
        r"model\.layers\.([0-9]+)\.post_attention_layernorm\.weight": (
            r"layers.\1.post_attention_layernorm.w",
            None,
        ),
        
        # LM head
        r"lm_head\.weight": ("lm_head.w", ((1, 0), None)),
    }
```

### Preprocess Functions
Some models require additional preprocessing after loading:

```python
# Gemma2 example: Stack Q, K, V into QKV tensor
def _make_preprocess_fn(cfg: model_lib.ModelConfig):
    """Creates a preprocess function to stack Q, K, V tensors."""
    q_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.q$")
    k_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.k$")
    v_pat = re.compile(r"tmp\.layers\.([0-9]+)\.attn\.v$")
    
    fused_qkv = cfg.num_heads == cfg.num_kv_heads
    pending = {}
    
    def preprocess(state_dict):
        output = {}
        
        for key, value in state_dict.items():
            q_m = q_pat.match(key)
            k_m = k_pat.match(key)
            v_m = v_pat.match(key)
            
            if q_m:
                layer_idx = q_m.group(1)
                pending.setdefault(layer_idx, {})['q'] = value
            elif k_m:
                layer_idx = k_m.group(1)
                pending.setdefault(layer_idx, {})['k'] = value
            elif v_m:
                layer_idx = v_m.group(1)
                pending.setdefault(layer_idx, {})['v'] = value
            else:
                output[key] = value
        
        # Stack Q, K, V into QKV tensor
        for layer_idx, qkv_dict in pending.items():
            if len(qkv_dict) == 3:
                q, k, v = qkv_dict['q'], qkv_dict['k'], qkv_dict['v']
                # Stack along appropriate dimension
                if fused_qkv:
                    qkv = jnp.stack([q, k, v], axis=0)  # [3, N, D, H]
                else:
                    qkv = jnp.concatenate([q, k, v], axis=1)  # [N_q+N_k+N_v, D, H]
                output[f'layers.{layer_idx}.attn.qkv'] = qkv
        
        return output
    
    return preprocess
```

### Model-Specific Loaders
```python
# tunix/models/llama3/params.py
def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Llama3:
    """Load Llama3 model from safetensors."""
    return safetensors_loader.load_and_create_model(
        file_dir=file_dir,
        model_class=model_lib.Llama3,
        config=config,
        key_mapping=_get_key_and_transform_mapping,
        mesh=mesh,
        preprocess_fn=None,  # Llama3 doesn't need preprocessing
        dtype=dtype,
    )

# tunix/models/gemma/params_safetensors.py
def create_model_from_safe_tensors(
    file_dir: str,
    config: model_lib.ModelConfig,
    mesh: jax.sharding.Mesh | None = None,
    dtype: jnp.dtype | None = None,
) -> model_lib.Gemma:
    """Load Gemma model from safetensors."""
    return safetensors_loader.load_and_create_model(
        file_dir=file_dir,
        model_class=model_lib.Gemma,
        config=config,
        key_mapping=_get_key_and_transform_mapping,
        mesh=mesh,
        preprocess_fn=_make_preprocess_fn(config),  # Gemma needs QKV stacking
        dtype=dtype,
    )
```

## Model Architectures
### Overview
Tunix supports four major model families:

1. **Gemma** (gemma/, gemma3/)
2. **Llama** (llama3/)
3. **Qwen** (qwen2/, qwen3/)
4. **DeepSeek-R1-Distill** (uses qwen2/)

### Common Architecture Components
All models implement these core components:

#### 1. Transformer Model
```python
class Transformer(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        # Embeddings
        self.embedder = Embedder(config, rngs=rngs)
        
        # Transformer blocks
        self.layers = [
            TransformerBlock(config, layer_id=i, rngs=rngs)
            for i in range(config.num_layers)
        ]
        
        # Final norm and LM head
        self.final_norm = RMSNorm(config.embed_dim, eps=config.norm_eps, rngs=rngs)
        self.lm_head = nnx.Linear(config.embed_dim, config.vocab_size, rngs=rngs)
    
    def __call__(self, tokens, positions, cache, attention_mask):
        x = self.embedder(tokens)
        
        for layer in self.layers:
            x, cache = layer(x, positions, cache, attention_mask)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits, cache
```

#### 2. Transformer Block
```python
class TransformerBlock(nnx.Module):
    def __init__(self, config: ModelConfig, layer_id: int, *, rngs: nnx.Rngs):
        # Pre-attention norm
        self.input_layernorm = RMSNorm(config.embed_dim, rngs=rngs)
        
        # Multi-head attention
        self.attn = Attention(config, rngs=rngs)
        
        # Post-attention norm (if applicable)
        if config.use_post_attn_norm:
            self.post_attn_norm = RMSNorm(config.embed_dim, rngs=rngs)
        
        # Pre-FFN norm
        self.post_attention_layernorm = RMSNorm(config.embed_dim, rngs=rngs)
        
        # Feed-forward network
        self.mlp = FeedForward(config, rngs=rngs)
        
        # Post-FFN norm (if applicable)
        if config.use_post_ffw_norm:
            self.post_ffw_norm = RMSNorm(config.embed_dim, rngs=rngs)
    
    def __call__(self, x, positions, cache, attention_mask):
        # Attention with residual
        attn_input = self.input_layernorm(x)
        attn_output, cache = self.attn(attn_input, positions, cache, attention_mask)
        
        if hasattr(self, 'post_attn_norm'):
            attn_output = self.post_attn_norm(attn_output)
        
        x = x + attn_output  # Residual connection
        
        # FFN with residual
        ffn_input = self.post_attention_layernorm(x)
        ffn_output = self.mlp(ffn_input)
        
        if hasattr(self, 'post_ffw_norm'):
            ffn_output = self.post_ffw_norm(ffn_output)
        
        x = x + ffn_output  # Residual connection
        
        return x, cache
```

#### 3. Grouped Query Attention (GQA)
```python
class Attention(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        
        # Q, K, V projections
        self.q_proj = nnx.Linear(
            config.embed_dim,
            config.num_heads * config.head_dim,
            use_bias=False,
            rngs=rngs
        )
        self.k_proj = nnx.Linear(
            config.embed_dim,
            config.num_kv_heads * config.head_dim,
            use_bias=False,
            rngs=rngs
        )
        self.v_proj = nnx.Linear(
            config.embed_dim,
            config.num_kv_heads * config.head_dim,
            use_bias=False,
            rngs=rngs
        )
        
        # Output projection
        self.o_proj = nnx.Linear(
            config.num_heads * config.head_dim,
            config.embed_dim,
            use_bias=False,
            rngs=rngs
        )
        
        # RoPE
        self.rope = RoPE(config.head_dim, config.rope_theta)
    
    def __call__(self, x, positions, cache, attention_mask):
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE
        q = self.rope(q, positions)
        k = self.rope(k, positions)
        
        # Update cache
        cache = self.update_cache(cache, k, v, positions)
        k_cached, v_cached = self.get_cached_kv(cache)
        
        # Grouped query attention
        if self.num_heads != self.num_kv_heads:
            # Repeat K, V to match number of query heads
            k_cached = jnp.repeat(k_cached, self.num_heads // self.num_kv_heads, axis=2)
            v_cached = jnp.repeat(v_cached, self.num_heads // self.num_kv_heads, axis=2)
        
        # Compute attention
        scores = jnp.einsum('bqhd,bkhd->bhqk', q, k_cached)
        scores = scores / jnp.sqrt(self.head_dim)
        scores = jnp.where(attention_mask, scores, -1e9)
        attn_weights = jax.nn.softmax(scores, axis=-1)
        
        # Apply attention to values
        attn_output = jnp.einsum('bhqk,bkhd->bqhd', attn_weights, v_cached)
        attn_output = attn_output.reshape(batch_size, seq_len, -1)
        
        # Output projection
        output = self.o_proj(attn_output)
        
        return output, cache
```

#### 4. Feed-Forward Network (SwiGLU)
```python
class FeedForward(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        # SwiGLU: two projections for gating
        self.gate_proj = nnx.Linear(
            config.embed_dim,
            config.hidden_dim,
            use_bias=False,
            rngs=rngs
        )
        self.up_proj = nnx.Linear(
            config.embed_dim,
            config.hidden_dim,
            use_bias=False,
            rngs=rngs
        )
        self.down_proj = nnx.Linear(
            config.hidden_dim,
            config.embed_dim,
            use_bias=False,
            rngs=rngs
        )
    
    def __call__(self, x):
        # SwiGLU activation: silu(gate) * up
        gate = jax.nn.silu(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        
        # Down projection
        output = self.down_proj(hidden)
        
        return output
```

### Model-Specific Features
#### Gemma2 - Sliding Window Attention
```python
class Gemma(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        # Layers alternate between LOCAL_SLIDING and GLOBAL attention
        self.layers = []
        for i in range(config.num_layers):
            attn_type = config.attention_types[i]
            layer = TransformerBlock(
                config,
                layer_id=i,
                attention_type=attn_type,
                rngs=rngs
            )
            self.layers.append(layer)
    
    # Attention applies sliding window mask for LOCAL_SLIDING layers
    def apply_sliding_window_mask(self, attention_mask, layer_id):
        if self.config.attention_types[layer_id] == AttentionType.LOCAL_SLIDING:
            # Restrict attention to window_size
            window_size = self.config.sliding_window_size
            # ... mask logic ...
        return attention_mask
```

#### Llama3.2 - Weight Tying
```python
class Llama3(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.embedder = Embedder(config, rngs=rngs)
        # ...
        
        if config.weight_tying:
            # Share weights between input embedding and output LM head
            self.lm_head = self.embedder.input_embedding
        else:
            self.lm_head = nnx.Linear(
                config.embed_dim,
                config.vocab_size,
                use_bias=False,
                rngs=rngs
            )
```

#### Qwen - Tied Embeddings
```python
class Qwen2(nnx.Module):
    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.embedder = Embedder(config, rngs=rngs)
        # ...
        
        if config.use_tied_embedding:
            # Tie input and output embeddings
            self.lm_head.kernel = self.embedder.input_embedding.embedding
```

## Weight Management
### Dummy Model Creation
For testing or initialization without pretrained weights:

```python
# tunix/models/dummy_model_creator.py

def create_dummy_model(
    model_class,
    config,
    mesh=None,
    dtype: jnp.dtype | None = None,
    random_seed: int = 0,
    scale: float = 0.02,
):
    """Create a model with random-initialized parameters.
    
    Args:
        model_class: Model class to instantiate
        config: Model configuration
        mesh: Optional JAX mesh for sharding
        dtype: Optional dtype for parameter initialization
        random_seed: RNG seed for initialization
        scale: Scaling factor applied to random normal values
    
    Returns:
        Model instance with randomly initialized weights
    """
    context_manager = mesh if mesh is not None else contextlib.nullcontext()
    
    with context_manager:
        # Build abstract model
        abs_model = nnx.eval_shape(
            lambda: model_class(config, rngs=nnx.Rngs(params=0))
        )
    
    graph_def, abs_state = nnx.split(abs_model)
    state_dict = abs_state.to_pure_dict()
    
    if mesh is not None:
        sharding_dict = nnx.get_named_sharding(abs_state, mesh).to_pure_dict()
    else:
        sharding_dict = None
    
    rngs = nnx.Rngs(random_seed)
    
    @partial(nnx.jit, static_argnums=(2, 3,))
    def make_param(rngs, scale, shape, dt):
        return scale * rngs.params.normal(shape, dt)
    
    def make_random_tensor(path, param, shard=None):
        shape = param.shape
        dt = dtype or getattr(param, "dtype", None) or jnp.float32
        
        if shard is None:
            return make_param(rngs, scale, shape, dt)
        else:
            # Create sharded array with callback
            shard_shape = shard.shard_shape(shape)
            
            def _callback(index):
                return make_param(rngs, scale, shard_shape, dt)
            
            return jax.make_array_from_callback(shape, shard, _callback)
    
    if sharding_dict is not None:
        state_dict = jax.tree.map_with_path(
            make_random_tensor, state_dict, sharding_dict
        )
    else:
        state_dict = jax.tree.map_with_path(make_random_tensor, state_dict)
    
    return nnx.merge(graph_def, state_dict)
```

### Usage
```python
from tunix.models import dummy_model_creator
from tunix.models.llama3.model import Llama3, ModelConfig
import jax

# Create config
config = ModelConfig.llama3p1_8b()

# Create mesh
mesh = jax.sharding.Mesh(
    jax.devices(),
    axis_names=('fsdp', 'tp')
)

# Create random model
model = dummy_model_creator.create_dummy_model(
    model_class=Llama3,
    config=config,
    mesh=mesh,
    dtype=jnp.bfloat16,
    random_seed=42,
    scale=0.02,  # Standard deviation of initialization
)

print(f"Model created with {sum(x.size for x in jax.tree.leaves(nnx.state(model)))} parameters")
```

## Usage Examples
### Example 1: Load Model from HuggingFace
```python
from tunix.models.automodel import AutoModel, ModelSource
import jax

# Setup mesh
devices = jax.devices()
mesh = jax.sharding.Mesh(
    devices.reshape(2, 4),  # 2 FSDP, 4 TP
    axis_names=('fsdp', 'tp')
)

# Load Llama-3.1-8B from HuggingFace
model, model_path = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3.1-8B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
    model_download_path="/tmp/hf_cache",
)

print(f"Model loaded from: {model_path}")
print(f"Model type: {type(model)}")
print(f"Number of layers: {model.config.num_layers}")
```

### Example 2: Load Multiple Model Sizes
```python
# Load different Qwen2.5 sizes
models = {}
for size in ['0.5b', '1.5b', '3b']:
    model, _ = AutoModel.from_pretrained(
        model_id=f"Qwen/Qwen2.5-{size.upper()}",
        mesh=mesh,
        model_source=ModelSource.HUGGINGFACE,
    )
    models[size] = model

# Compare parameter counts
for size, model in models.items():
    param_count = sum(x.size for x in jax.tree.leaves(nnx.state(model)))
    print(f"Qwen2.5-{size}: {param_count / 1e9:.2f}B parameters")
```

### Example 3: Load Model with Custom Config
```python
from tunix.models.llama3.model import Llama3, ModelConfig
from tunix.models import safetensors_loader
from tunix.models.llama3 import params

# Create custom config (modify existing)
base_config = ModelConfig.llama3p1_8b()
custom_config = dataclasses.replace(
    base_config,
    num_layers=16,  # Reduce layers for testing
)

# Load with custom config
model = params.create_model_from_safe_tensors(
    file_dir="/path/to/safetensors",
    config=custom_config,
    mesh=mesh,
)

print(f"Custom model with {custom_config.num_layers} layers")
```

### Example 4: Create Dummy Model for Testing
```python
from tunix.models import dummy_model_creator
from tunix.models.gemma.model import Gemma, ModelConfig

# Create config
config = ModelConfig.gemma2_2b()

# Create random model
dummy_model = dummy_model_creator.create_dummy_model(
    model_class=Gemma,
    config=config,
    mesh=mesh,
    dtype=jnp.bfloat16,
    random_seed=42,
    scale=0.02,
)

# Use for testing training loop without downloading weights
# ... test training code ...
```

### Example 5: Load with Specific Sharding
```python
from tunix.models.llama3.model import Llama3, ModelConfig, ShardingConfig

# Custom sharding configuration
custom_sharding = ShardingConfig(
    emb_vd=('tp', None),           # Full TP for embeddings
    q_weight_dnh=(None, 'tp', None),  # TP only on head dimension
    kv_weight_dnh=(None, 'tp', None),
    o_weight_nhd=('tp', None, None),
    ffw_weight_df=(None, 'tp'),
    ffw_weight_fd=('tp', None),
    rms_norm_weight=(None,),       # Replicate norms
    act_btd=(None, None, None),    # Replicate activations
    act_btf=(None, None, 'tp'),
    act_btnh=(None, None, 'tp', None),
)

# Create config with custom sharding
config = ModelConfig.llama3p1_8b()
config = dataclasses.replace(config, shd_config=custom_sharding)

# Load model
model, _ = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3.1-8B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
)
```

### Example 6: DeepSeek-R1-Distill Model
```python
# Load DeepSeek-R1-Distill-Qwen model
model, path = AutoModel.from_pretrained(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
)

# Check config
print(f"Model family: {model.config.model_family}")
print(f"RoPE theta: {model.config.rope_theta}")  # Different from standard Qwen
print(f"Tied embeddings: {model.config.use_tied_embedding}")
```

### Example 7: Model Inference
```python
# Prepare inputs
import jax.numpy as jnp

batch_size = 4
seq_len = 128
tokens = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
positions = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

# Initialize cache
cache = model.init_cache(batch_size, cache_size=2048)

# Create attention mask
attention_mask = jnp.ones((batch_size, 1, seq_len), dtype=jnp.bool_)

# Forward pass
logits, updated_cache = model(tokens, positions, cache, attention_mask)

print(f"Logits shape: {logits.shape}")  # [batch_size, seq_len, vocab_size]
print(f"Cache updated: {len(updated_cache)} layers")
```

### Example 8: Save Model to SafeTensors
```python
from tunix.models import safetensors_saver

# Get model state
_, state = nnx.split(model)
state_dict = state.to_pure_dict()

# Save to SafeTensors
save_dir = "/path/to/save/model"
safetensors_saver.save_model_to_safetensors(
    state_dict=state_dict,
    save_dir=save_dir,
    max_shard_size="5GB",  # Split into 5GB shards
)

print(f"Model saved to: {save_dir}")
```

### Example 9: Transfer Between Model Families
```python
# Load teacher model (Gemma2-9B)
teacher_model, _ = AutoModel.from_pretrained(
    model_id="google/gemma-2-9b",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
)

# Load student model (Gemma2-2B)
student_model, _ = AutoModel.from_pretrained(
    model_id="google/gemma-2-2b",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
)

# Use for distillation
from tunix.distillation import DistillationTrainer
# ... distillation setup ...
```

### Example 10: Inspect Model Structure
```python
from tunix.models.automodel import AutoModel, call_model_config, get_model_module, ModelModule

# Get config without loading weights
model_name = "llama-3.1-8b"
config = call_model_config(model_name)

print(f"Config for {model_name}:")
print(f"  Layers: {config.num_layers}")
print(f"  Embed dim: {config.embed_dim}")
print(f"  Hidden dim: {config.hidden_dim}")
print(f"  Heads: {config.num_heads}")
print(f"  KV heads: {config.num_kv_heads}")
print(f"  Head dim: {config.head_dim}")
print(f"  RoPE theta: {config.rope_theta}")

# Get model module
model_module = get_model_module(model_name, ModelModule.MODEL)
print(f"\nModel class: {model_module.Llama3}")

# List all available configs
print(f"\nAvailable configs:")
for attr in dir(model_module.ModelConfig):
    if not attr.startswith('_') and callable(getattr(model_module.ModelConfig, attr)):
        print(f"  - {attr}")
```

## Best Practices
### 1. Choosing the Right Model Source
#### HuggingFace (Recommended)
- **Use When**:
  - Standard model loading
  - Production deployment
  - Most model families (Llama, Qwen, Gemma from HF)
- **Advantages**:
  - Direct SafeTensors loading (fast)
  - No conversion needed
  - Standard format
- **Example Models**:
  - `meta-llama/Llama-3.1-8B`
  - `Qwen/Qwen2.5-0.5B`
  - `google/gemma-2-2b` (if available on HF)

#### Kaggle
- **Use When**:
  - Loading Gemma models from Kaggle
  - No HuggingFace access
- **Disadvantages**:
  - Requires NNX conversion (slow, ~10-20 minutes)
  - Large memory overhead during conversion
- **Best Practice**: Cache converted checkpoint
```python
model, _ = AutoModel.from_pretrained(
    model_id="google/gemma-2b",
    mesh=mesh,
    model_source=ModelSource.KAGGLE,
    intermediate_ckpt_dir="/persistent/cache/gemma_nnx",  # Reuse on next run
)
```

#### GCS
- **Use When**:
  - Internal Google Cloud deployment
  - Custom checkpoints
  - Gemma3 models (currently GCS-only)
- **Requirements**: Provide `model_path` parameter

#### INTERNAL
- **Use When**: Internal Google infrastructure
- **Note**: Not available in OSS

### 2. Memory Management
#### Model Size Estimation
```python
def estimate_model_memory(config):
    """Estimate model memory requirements."""
    # Embedding table
    emb_params = config.vocab_size * config.embed_dim
    
    # Attention per layer
    attn_params_per_layer = (
        config.embed_dim * config.num_heads * config.head_dim +  # Q
        config.embed_dim * config.num_kv_heads * config.head_dim * 2 +  # K, V
        config.num_heads * config.head_dim * config.embed_dim  # O
    )
    
    # FFN per layer
    ffn_params_per_layer = (
        config.embed_dim * config.hidden_dim * 2 +  # gate, up
        config.hidden_dim * config.embed_dim  # down
    )
    
    # Total
    total_params = (
        emb_params +
        config.num_layers * (attn_params_per_layer + ffn_params_per_layer)
    )
    
    # Memory in GB (bf16 = 2 bytes)
    memory_gb = total_params * 2 / (1024**3)
    
    return total_params, memory_gb

# Example
config = ModelConfig.llama3p1_8b()
params, mem = estimate_model_memory(config)
print(f"Llama-3.1-8B: {params/1e9:.2f}B params, ~{mem:.2f}GB (bf16)")
```

#### Sharding for Large Models
```python
# For 70B model, use more aggressive sharding
devices = jax.devices()
mesh = jax.sharding.Mesh(
    devices.reshape(8, 8),  # 8 FSDP, 8 TP
    axis_names=('fsdp', 'tp')
)

# Load with automatic sharding
model, _ = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3-70B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
)
```

### 3. Model Naming Best Practices
#### Always Use Standard Names
```python
# ✅ GOOD: Use standard lowercase naming
model_name = "llama-3.1-8b"
config = call_model_config(model_name)

# ❌ BAD: Non-standard casing
model_name = "Llama-3.1-8B"  # Will fail
```

#### Extract Name from Model ID
```python
# ✅ GOOD: Use naming utility
from tunix.models.naming import get_model_name_from_model_id

model_id = "meta-llama/Llama-3.1-8B"
model_name = get_model_name_from_model_id(model_id)  # "llama-3.1-8b"

# ❌ BAD: Manual parsing
model_name = model_id.split('/')[-1].lower()  # May not handle edge cases
```

#### Validate Model Names
```python
from tunix.models.naming import get_model_config_category, split

try:
    model_family, model_version = split(model_name)
    config_category = get_model_config_category(model_name)
    print(f"Valid model: {model_name}")
    print(f"  Family: {model_family}, Version: {model_version}")
    print(f"  Config category: {config_category}")
except ValueError as e:
    print(f"Invalid model name: {e}")
```

### 4. Configuration Customization
#### Modify Existing Configs
```python
import dataclasses

# Start with base config
base_config = ModelConfig.gemma2_2b()

# Customize for your use case
custom_config = dataclasses.replace(
    base_config,
    num_layers=18,  # Reduce layers
    sliding_window_size=2048,  # Smaller window
    remat_config=RematConfig.BLOCK,  # Enable remat
)

# Use custom config
model = dummy_model_creator.create_dummy_model(
    model_class=Gemma,
    config=custom_config,
    mesh=mesh,
)
```

#### Add New Model Variants
```python
# In tunix/models/llama3/model.py

class ModelConfig:
    # ... existing configs ...
    
    @classmethod
    def llama3_custom_4b(cls):
        """Custom 4B variant."""
        return cls(
            num_layers=24,
            vocab_size=128256,
            embed_dim=2816,
            hidden_dim=11264,
            num_heads=22,
            head_dim=128,
            num_kv_heads=4,
            norm_eps=1e-05,
            rope_theta=500_000,
            weight_tying=False,
        )
```

### 5. Weight Loading Optimization
#### Parallel Loading
```python
# Load multiple models in parallel
import concurrent.futures

model_ids = [
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen2.5-3B",
]

def load_model(model_id):
    return AutoModel.from_pretrained(
        model_id=model_id,
        mesh=mesh,
        model_source=ModelSource.HUGGINGFACE,
    )

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    models = list(executor.map(load_model, model_ids))
```

#### Cache Downloaded Models
```python
# Set persistent cache directory
cache_dir = "/persistent/hf_cache"

# All models will be cached here
model, path = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3.1-8B",
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
    model_download_path=cache_dir,
)

# Subsequent loads are instant
model2, path2 = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3.1-8B",  # Loads from cache
    mesh=mesh,
    model_source=ModelSource.HUGGINGFACE,
    model_download_path=cache_dir,
)
```

### 6. Testing and Debugging
#### Use Dummy Models for Testing
```python
# Fast testing without downloading
def test_training_loop():
    config = ModelConfig.llama3p1_8b()
    model = dummy_model_creator.create_dummy_model(
        model_class=Llama3,
        config=config,
        mesh=mesh,
        random_seed=42,
    )
    
    # Test training logic
    # ... training code ...
    
    print("Training loop test passed!")

test_training_loop()
```

#### Validate Weight Loading
```python
# Check if weights loaded correctly
def validate_weights(model):
    _, state = nnx.split(model)
    state_dict = state.to_pure_dict()
    
    # Check for NaN/Inf
    has_nan = any(jnp.isnan(v).any() for v in jax.tree.leaves(state_dict))
    has_inf = any(jnp.isinf(v).any() for v in jax.tree.leaves(state_dict))
    
    if has_nan or has_inf:
        print("WARNING: Model has NaN or Inf values!")
        return False
    
    # Check parameter statistics
    all_params = jax.tree.leaves(state_dict)
    mean_val = jnp.mean(jnp.array([jnp.mean(jnp.abs(p)) for p in all_params]))
    print(f"Mean absolute parameter value: {mean_val:.6f}")
    
    return True

# Validate loaded model
model, _ = AutoModel.from_pretrained(...)
validate_weights(model)
```

### 7. Production Deployment
#### Checklist
- [ ] Use HuggingFace source for standard models
- [ ] Set up persistent cache directory
- [ ] Configure appropriate mesh for model size
- [ ] Test with dummy model first
- [ ] Validate loaded weights
- [ ] Monitor memory usage
- [ ] Set up error handling for download failures
- [ ] Document model version and config
- [ ] Test inference pipeline end-to-end
- [ ] Benchmark loading time

#### Error Handling
```python
from tunix.models.automodel import AutoModel, ModelSource
import logging

def load_model_with_retry(model_id, mesh, max_retries=3):
    """Load model with retry logic."""
    for attempt in range(max_retries):
        try:
            model, path = AutoModel.from_pretrained(
                model_id=model_id,
                mesh=mesh,
                model_source=ModelSource.HUGGINGFACE,
            )
            logging.info(f"Model loaded successfully: {path}")
            return model, path
        except Exception as e:
            logging.warning(f"Attempt {attempt+1} failed: {e}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to load model after {max_retries} attempts")
                raise
    
    return None, None
```

### 8. Common Pitfalls
#### ❌ Don't: Mix Model Families
```python
# BAD: Using Llama config with Gemma weights
config = ModelConfig.llama3p1_8b()
model, _ = AutoModel.from_pretrained(
    model_id="google/gemma-2-2b",  # Wrong!
    mesh=mesh,
)
```

#### ✅ Do: Let AutoModel Handle Config Selection
```python
# GOOD: AutoModel selects correct config automatically
model, _ = AutoModel.from_pretrained(
    model_id="google/gemma-2-2b",
    mesh=mesh,
)
```

#### ❌ Don't: Forget to Set Mesh
```python
# BAD: No mesh for distributed training
model, _ = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3-70B",  # 70B model
    mesh=None,  # Will OOM!
)
```

#### ✅ Do: Always Provide Mesh for Large Models
```python
# GOOD: Proper mesh for 70B model
mesh = jax.sharding.Mesh(
    jax.devices().reshape(8, 8),
    axis_names=('fsdp', 'tp')
)
model, _ = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3-70B",
    mesh=mesh,
)
```

#### ❌ Don't: Hardcode Model Paths
```python
# BAD: Hardcoded path
model_path = "/home/user/models/llama-3.1-8b"
```

#### ✅ Do: Use Environment Variables or Config Files
```python
# GOOD: Configurable path
import os
cache_dir = os.getenv('MODEL_CACHE_DIR', '/tmp/model_cache')
model, _ = AutoModel.from_pretrained(
    model_id="meta-llama/Llama-3.1-8B",
    mesh=mesh,
    model_download_path=cache_dir,
)
```

```

```
