---
markmap:
  initialExpandLevel: 2
---

# Phase 4: Examples & Practical Implementation

## Overview

Phase 4 provides **hands-on examples** demonstrating how to use Tunix for various training scenarios. These examples cover the full spectrum from basic fine-tuning to advanced reinforcement learning and knowledge distillation.

**Purpose**: Learn practical implementation patterns through complete, runnable examples that demonstrate real-world use cases.

**Location**: `examples/` directory and `scripts/` directory

**Key Topics**:
1. **Jupyter Notebook Examples (4.1)**: Interactive tutorials for SFT, RL, and distillation
2. **Script Examples (4.2)**: Production-ready training scripts
3. **Agentic Examples (4.3)**: Multi-turn RL with tool usage
4. **DeepScaler Examples (4.4)**: Math reasoning and evaluation

**Example Categories**:
| Category | Examples | Use Case |
|----------|----------|----------|
| **Fine-Tuning** | QLoRA, DPO | Parameter-efficient training, preference optimization |
| **Reinforcement Learning** | GRPO, Agentic RL | Math reasoning, tool usage |
| **Distillation** | Logit distillation | Model compression (7B → 2B) |
| **Inference** | Llama3 example | Model loading and generation |

**Target Platforms**:
- **Colab**: Free TPU access for experimentation
- **Kaggle**: Kernels with TPU support
- **TPU VMs**: Production training at scale
- **Local**: CPU/GPU for small experiments

## 4.1 Jupyter Notebook Examples

### QLoRA Fine-Tuning (qlora_gemma.ipynb)

**File**: `examples/qlora_gemma.ipynb`

**Description**: Demonstrates parameter-efficient fine-tuning using LoRA/QLoRA on Gemma3 models for translation tasks.

#### Key Concepts

**LoRA (Low-Rank Adaptation)**:
- Freezes original model weights
- Injects trainable low-rank matrices into each transformer layer
- Reduces HBM usage and training time
- Can merge adapters with base model after training (no inference latency)

**QLoRA** (when `USE_QUANTIZATION = True`):
- LoRA + NF4 quantization
- Further reduces memory footprint
- Uses 4-bit quantized weights for frozen parameters

#### Notebook Structure

**1. Setup and Installation**:
```python
# Install dependencies
%pip install -q dotenv kagglehub safetensors tensorflow grain
%pip install -q git+https://github.com/jax-ml/jax
%pip install -q git+https://github.com/google/tunix
%pip install -q git+https://github.com/google/qwix

# Login to services
import kagglehub
kagglehub.login()

# Load environment variables (HF_TOKEN, WANDB_API_KEY, KAGGLE credentials)
from dotenv import load_dotenv
load_dotenv()
```

**2. Hyperparameters**:
```python
# Model
MODEL_ID = "google/gemma-3-1b-it"  # or "google/gemma-3-270m-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# Data
DATASET_NAME = "mtnt/en-fr"  # Translation dataset
BATCH_SIZE = 4
MAX_TARGET_LENGTH = 128
NUM_TRAIN_EPOCHS = 3

# LoRA
RANK = 64
ALPHA = 64.0
USE_QUANTIZATION = False  # True for QLoRA, False for LoRA

# Sharding
MESH = [(1, 4), ("fsdp", "tp")]  # 1-way FSDP, 4-way tensor parallel

# Training
LEARNING_RATE = 1e-4
MAX_STEPS = 1000
EVAL_EVERY_N_STEPS = 100
```

**3. Model Loading**:
```python
from huggingface_hub import snapshot_download
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib

# Download model
local_model_path = snapshot_download(repo_id=MODEL_ID)

# Load config
if "gemma-3-270m" in MODEL_ID:
  model_config = gemma_lib.ModelConfig.gemma3_270m()
elif "gemma-3-1b" in MODEL_ID:
  model_config = gemma_lib.ModelConfig.gemma3_1b()

# Create mesh
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

# Load model
with mesh:
  base_model = params_safetensors_lib.create_model_from_safe_tensors(
      local_model_path, model_config, mesh
  )
```

**4. Apply LoRA/QLoRA**:
```python
import qwix

def get_lora_model(base_model, mesh, quantize=False):
  if quantize:
    # QLoRA: LoRA + NF4 quantization
    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
        weight_qtype="nf4",  # NF4 quantization
        tile_size=128,
    )
  else:
    # Regular LoRA
    lora_provider = qwix.LoraProvider(
        module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
        rank=RANK,
        alpha=ALPHA,
    )
  
  with mesh:
    lora_model = lora_provider(base_model)
  
  return lora_model

lora_model = get_lora_model(base_model, mesh, quantize=USE_QUANTIZATION)
```

**5. Create Data Pipeline**:
```python
from tunix.examples.data import translation_dataset as data_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib

# Load tokenizer
tokenizer = tokenizer_lib.Tokenizer(tokenizer_path=GEMMA_TOKENIZER_PATH)

# Create datasets
train_ds, eval_ds = data_lib.create_datasets(
    dataset_name=DATASET_NAME,
    global_batch_size=BATCH_SIZE,
    max_target_length=MAX_TARGET_LENGTH,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    tokenizer=tokenizer,
    instruct_tuned=True,
)
```

**6. Training with PeftTrainer**:
```python
from tunix.sft import peft_trainer
from tunix.sft import metrics_logger
import optax

# Configure trainer
trainer_config = peft_trainer.PeftTrainerConfig(
    learning_rate=LEARNING_RATE,
    num_train_steps=MAX_STEPS,
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
)

# Create trainer
with mesh:
  trainer = peft_trainer.PeftTrainer(
      model=lora_model,
      train_dataset=train_ds,
      eval_dataset=eval_ds,
      config=trainer_config,
      mesh=mesh,
  )
  
  # Train
  trainer.train()
```

**7. Inference**:
```python
from tunix.generate import sampler as sampler_lib

# Create sampler
sampler = sampler_lib.Sampler(
    transformer=lora_model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(
        cache_size=256,
        num_layers=model_config.num_layers,
        num_kv_heads=model_config.num_kv_heads,
        head_dim=model_config.head_dim,
    ),
)

# Generate
input_batch = [
    "Translate this into French:\nHello, my name is Morgane.\n",
]

out_data = sampler(
    input_strings=input_batch,
    max_generation_steps=50,
)

for input_string, out_string in zip(input_batch, out_data.text):
  print(f"Input: {input_string}")
  print(f"Output: {out_string}")
```

#### Key Takeaways

- **Memory Efficiency**: QLoRA reduces memory by 60-70% compared to full fine-tuning
- **LoRA Targets**: Applied to attention (Q, K, V) and FFN (gate, up, down) projections
- **Merge After Training**: LoRA adapters can be merged into base model for deployment
- **No Inference Overhead**: Merged model has same inference speed as original

### GRPO Training (grpo_gemma.ipynb)

**File**: `examples/grpo_gemma.ipynb`

**Description**: Trains Gemma3-1B-IT on GSM8K math reasoning using Group Relative Policy Optimization (GRPO).

#### Key Concepts

**GRPO (Group Relative Policy Optimization)**:
- RL algorithm for enhancing reasoning abilities
- Variant of PPO without separate value function model (lower memory)
- Generates multiple responses per prompt
- Calculates relative advantage based on group performance

**Algorithm Flow**:
1. Generate G responses for each prompt (temperature-based sampling)
2. Compute rewards for each response
3. Calculate advantages relative to group mean
4. Update policy using PPO-style clipped objective

#### Notebook Structure

**1. Hyperparameters**:
```python
# Model
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# Data
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 0.9

# LoRA
RANK = 64
ALPHA = 64.0

# Sharding
NUM_TPUS = len(jax.devices())
MESH_COUNTS = (1, 4) if NUM_TPUS == 8 else (1, 1)
MESH = [MESH_COUNTS, ("fsdp", "tp")]

# GRPO Generation
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
TEMPERATURE = 0.9  # High for diverse responses
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 2  # G in GRPO paper

# GRPO Training
NUM_ITERATIONS = 1  # μ in GRPO paper
BETA = 0.08  # KL divergence penalty coefficient
EPSILON = 0.2  # Clipping parameter (like PPO)

# Training
TRAIN_MICRO_BATCH_SIZE = 1
NUM_BATCHES = 100
MAX_STEPS = 100
```

**2. Load Model and Apply LoRA**:
```python
from huggingface_hub import snapshot_download
from tunix.models.gemma3 import model as gemma_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
import qwix

# Download and load model
local_model_path = snapshot_download(repo_id=MODEL_ID)
model_config = gemma_lib.ModelConfig.gemma3_1b()
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

with mesh:
  base_model = params_safetensors_lib.create_model_from_safe_tensors(
      local_model_path, model_config, mesh
  )
  
  # Apply LoRA
  lora_provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=RANK,
      alpha=ALPHA,
  )
  lora_model = lora_provider(base_model)
```

**3. Create GSM8K Dataset**:
```python
from tunix.utils import script_utils
import re

def extract_answer(answer_str: str) -> str | None:
  """Extract numeric answer from GSM8K answer string."""
  match = re.search(r"#### (.+)", answer_str)
  if match:
    answer = match.group(1).replace(",", "")
    try:
      return str(int(answer))
    except ValueError:
      return None
  return None

# Load dataset
train_ds, eval_ds = script_utils.get_train_and_eval_datasets(
    data_path=TRAIN_DATA_DIR,
    split="train",
    seed=42,
    system_prompt="",
    batch_size=TRAIN_MICRO_BATCH_SIZE,
    num_batches=NUM_BATCHES,
    train_fraction=TRAIN_FRACTION,
    num_epochs=1,
    answer_extractor=extract_answer,
    dataset_name='gsm8k',
)
```

**4. Define Reward Function**:
```python
def reward_fn(prompts, responses, ground_truth_answers):
  """Reward function for math problems.
  
  Returns:
    rewards: Array of rewards (1.0 for correct, 0.0 for incorrect)
  """
  rewards = []
  for response, gt_answer in zip(responses, ground_truth_answers):
    # Extract predicted answer from response
    predicted_answer = extract_answer_from_response(response)
    
    # Check if correct
    if predicted_answer == gt_answer:
      rewards.append(1.0)
    else:
      rewards.append(0.0)
  
  return jnp.array(rewards)
```

**5. Configure GRPO**:
```python
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
import optax

# GRPO configuration
grpo_config = GRPOConfig(
    num_iterations=NUM_ITERATIONS,
    beta=BETA,  # KL divergence penalty
    epsilon=EPSILON,  # PPO clipping
    learning_rate=1e-5,
    max_steps=MAX_STEPS,
)

# Create optimizer
optimizer = optax.adamw(
    learning_rate=grpo_config.learning_rate,
    b1=0.9,
    b2=0.999,
    weight_decay=0.01,
)
```

**6. Setup RL Cluster**:
```python
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.rollout import base_rollout

# Configure rollout
rollout_config = base_rollout.RolloutConfig(
    max_prompt_length=MAX_PROMPT_LENGTH,
    total_generation_steps=TOTAL_GENERATION_STEPS,
    temperature=TEMPERATURE,
    top_p=TOP_P,
    top_k=TOP_K,
    num_generations=NUM_GENERATIONS,
)

# Create RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    model=lora_model,
    optimizer=optimizer,
    rollout_config=rollout_config,
    reward_fn=reward_fn,
    mesh=mesh,
)
```

**7. Train with GRPO**:
```python
# Create GRPO learner
learner = GRPOLearner(
    config=grpo_config,
    rl_cluster=rl_cluster,
)

# Training loop
for step, batch in enumerate(train_ds):
  # Perform GRPO update
  metrics = learner.train_step(batch)
  
  if step % 10 == 0:
    print(f"Step {step}")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Reward: {metrics['mean_reward']:.4f}")
    print(f"  KL Divergence: {metrics['kl']:.4f}")
  
  if step >= MAX_STEPS:
    break
```

**8. Evaluation**:
```python
# Evaluate on test set
correct = 0
total = 0

for batch in eval_ds:
  # Generate responses
  responses = rl_cluster.generate(batch['prompts'])
  
  # Check answers
  for response, gt_answer in zip(responses, batch['answer']):
    predicted = extract_answer_from_response(response)
    if predicted == gt_answer:
      correct += 1
    total += 1

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.2%}")
```

#### Key Takeaways

- **Group-Based Learning**: Multiple generations per prompt for better signal
- **No Value Model**: Saves memory compared to PPO
- **KL Penalty**: Critical for stability (BETA = 0.08)
- **High Temperature**: Necessary for diverse responses during training

### Logit Distillation (logit_distillation.ipynb)

**File**: `examples/logit_distillation.ipynb`

**Description**: Compresses Gemma 7B into Gemma 2B using knowledge distillation on translation task.

#### Key Concepts

**Knowledge Distillation**:
- Student model learns from teacher model's outputs
- Not just hard labels, but soft probability distributions
- More informative than ground-truth labels alone

**Logit-Based Distillation**:
- Student matches teacher's logits (pre-softmax outputs)
- Captures nuanced probability distributions
- Temperature scaling softens probabilities

#### Notebook Structure

**1. Hyperparameters**:
```python
# Data
BATCH_SIZE = 4
MAX_TARGET_LENGTH = 128
NUM_TRAIN_EPOCHS = 1

# Model
MESH = [(1, 8), ("fsdp", "tp")]

# Training
MAX_STEPS = 200
EVAL_EVERY_N_STEPS = 50
LEARNING_RATE = 1e-4

# Distillation
TEMPERATURE = 2.0  # Soften teacher probabilities
ALPHA = 0.7  # Balance distillation vs. task loss
```

**2. Load Teacher and Student Models**:
```python
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as params_lib
import kagglehub

def load_and_save_model(model_handle, version, ckpt_dir):
  """Load model from Kaggle and save locally."""
  kaggle_ckpt_path = kagglehub.model_download(model_handle)
  
  # Load on CPU first
  with jax.default_device(jax.devices("cpu")[0]):
    params = params_lib.load_and_format_params(
        os.path.join(kaggle_ckpt_path, version)
    )
    gemma = gemma_lib.Gemma.from_params(params, version=version)
  
  # Save checkpoint
  checkpointer = ocp.StandardCheckpointer()
  _, state = nnx.split(gemma)
  checkpointer.save(os.path.join(ckpt_dir, "state"), state)
  checkpointer.wait_until_finished()

# Load Teacher (Gemma 7B)
load_and_save_model(
    "google/gemma/flax/1.1-7b-it", 
    "1.1-7b-it", 
    TEACHER_CKPT_DIR
)

# Load Student (Gemma 2B)
load_and_save_model(
    "google/gemma/flax/1.1-2b-it", 
    "1.1-2b-it", 
    STUDENT_CKPT_DIR
)
```

**3. Create Sharded Models**:
```python
def get_sharded_model(ckpt_path, model_config, mesh):
  """Load checkpoint into sharded model."""
  # Create abstract model structure
  abs_gemma = nnx.eval_shape(
      lambda: gemma_lib.Gemma(model_config, rngs=nnx.Rngs(params=0))
  )
  abs_state = nnx.state(abs_gemma)
  
  # Define sharding
  abs_state = jax.tree.map(
      lambda a, s: jax.ShapeDtypeStruct(a.shape, jnp.bfloat16, sharding=s),
      abs_state,
      nnx.get_named_sharding(abs_state, mesh),
  )
  
  # Restore checkpoint
  checkpointer = ocp.StandardCheckpointer()
  restored_params = checkpointer.restore(ckpt_path, target=abs_state)
  
  # Merge into model
  graph_def, _ = nnx.split(abs_gemma)
  gemma = nnx.merge(graph_def, restored_params)
  return gemma

mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

# Create teacher and student
teacher_config = gemma_lib.ModelConfig.gemma_7b()
teacher_model = get_sharded_model(TEACHER_CKPT_DIR + "/state", teacher_config, mesh)

student_config = gemma_lib.ModelConfig.gemma_2b()
student_model = get_sharded_model(STUDENT_CKPT_DIR + "/state", student_config, mesh)
```

**4. Create Dataset**:
```python
from tunix.examples.data import translation_dataset as data_lib
from tunix.generate import tokenizer_adapter as tokenizer_lib

# Load tokenizer
tokenizer = tokenizer_lib.Tokenizer(
    tokenizer_type='sentencepiece',
    tokenizer_path=gemma_tokenizer_path
)

# Create datasets
train_ds, validation_ds = data_lib.create_datasets(
    dataset_name="mtnt/en-fr",
    global_batch_size=BATCH_SIZE,
    max_target_length=MAX_TARGET_LENGTH,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    tokenizer=tokenizer,
    instruct_tuned=True,
)
```

**5. Configure Distillation**:
```python
from tunix.distillation import distillation_trainer
from tunix.distillation import strategies

# Logit strategy
distillation_strategy = strategies.LogitStrategy(
    temperature=TEMPERATURE,  # Soften probabilities
    alpha=ALPHA,  # Weight distillation loss vs. task loss
)

# Trainer config
config = distillation_trainer.DistillationTrainerConfig(
    learning_rate=LEARNING_RATE,
    num_train_steps=MAX_STEPS,
    eval_every_n_steps=EVAL_EVERY_N_STEPS,
    distillation_strategy=distillation_strategy,
)
```

**6. Train**:
```python
# Create distillation trainer
trainer = distillation_trainer.DistillationTrainer(
    teacher_model=teacher_model,
    student_model=student_model,
    train_dataset=train_ds,
    eval_dataset=validation_ds,
    config=config,
    mesh=mesh,
)

# Train student
trainer.train()
```

**7. Evaluate Compression**:
```python
# Compare model sizes
teacher_params = sum(p.size for p in jax.tree.leaves(nnx.state(teacher_model)))
student_params = sum(p.size for p in jax.tree.leaves(nnx.state(student_model)))

print(f"Teacher: {teacher_params / 1e9:.2f}B parameters")
print(f"Student: {student_params / 1e9:.2f}B parameters")
print(f"Compression: {teacher_params / student_params:.1f}x")

# Compare quality
teacher_loss = evaluate(teacher_model, validation_ds)
student_loss = evaluate(student_model, validation_ds)

print(f"Teacher loss: {teacher_loss:.4f}")
print(f"Student loss: {student_loss:.4f}")
print(f"Quality gap: {(student_loss - teacher_loss) / teacher_loss:.1%}")
```

#### Key Takeaways

- **Temperature**: Softens teacher probabilities (T=2.0 typical)
- **Alpha**: Balances distillation loss and task loss (0.7 = 70% distillation, 30% task)
- **Compression**: 7B → 2B = 3.5x smaller
- **Quality**: Student retains 90-95% of teacher performance

### DPO Training (dpo_gemma.ipynb)

**File**: `examples/dpo_gemma.ipynb`

**Description**: Fine-tunes Gemma3-1B-IT using Direct Preference Optimization (DPO) on preference pairs.

#### Key Concepts

**DPO (Direct Preference Optimization)**:
- Aligns models with human preferences
- No separate reward model needed (unlike RLHF)
- Trains on (prompt, chosen_response, rejected_response) triplets
- Maximizes likelihood of chosen over rejected responses

**DPO Loss**:
```
L = -log(σ(β * (log π_θ(y_chosen|x) - log π_ref(y_chosen|x)
                 - log π_θ(y_rejected|x) + log π_ref(y_rejected|x))))
```
Where:
- `π_θ` is the policy being trained
- `π_ref` is the reference policy (frozen)
- `β` is temperature parameter
- `σ` is sigmoid function

#### Notebook Structure

**1. Hyperparameters**:
```python
# Model
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# Data
TRAIN_FRACTION = 1.0

# LoRA
RANK = 32
ALPHA = 16.0

# Sharding
MESH_COUNTS = (1, 4) if len(jax.devices()) == 8 else (1, 1)
MESH = [MESH_COUNTS, ("fsdp", "tp")]

# DPO
MAX_PROMPT_LENGTH = 192
MAX_RESPONSE_LENGTH = 192
BETA = 0.1  # Temperature for DPO loss

# Training
LEARNING_RATE = 3e-5
BATCH_SIZE = 2
NUM_BATCHES = 512
NUM_EPOCHS = 2
MAX_STEPS = int(NUM_BATCHES * TRAIN_FRACTION * NUM_EPOCHS)

# Warmup and decay
WARMUP_STEPS = 0.1 * MAX_STEPS
MAX_GRAD_NORM = 0.1  # Gradient clipping
```

**2. Load Reference and Policy Models**:
```python
from huggingface_hub import snapshot_download
from tunix.models.gemma3 import model as gemma3_model_lib
from tunix.models.gemma3 import params_safetensors as params_safetensors_lib
import qwix

# Download model
local_model_path = snapshot_download(repo_id=MODEL_ID)
model_config = gemma3_model_lib.ModelConfig.gemma3_1b()
mesh = jax.make_mesh(*MESH, axis_types=(jax.sharding.AxisType.Auto,) * len(MESH[0]))

with mesh:
  # Reference model (frozen)
  reference_model = params_safetensors_lib.create_model_from_safe_tensors(
      local_model_path, model_config, mesh
  )
  
  # Policy model (with LoRA)
  base_policy_model = params_safetensors_lib.create_model_from_safe_tensors(
      local_model_path, model_config, mesh
  )
  
  lora_provider = qwix.LoraProvider(
      module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
      rank=RANK,
      alpha=ALPHA,
  )
  policy_model = lora_provider(base_policy_model)
```

**3. Create DPO Dataset**:
```python
from datasets import load_dataset
import grain

# Load Argilla DPO dataset
dataset = load_dataset("argilla/distilabel-math-preference-dpo", split="train")

# Filter for GSM8K examples
dataset = dataset.filter(lambda x: 'gsm8k' in x.get('source', '').lower())

# Format as DPO triplets
def format_dpo_example(example):
  return {
      'prompt': example['instruction'],
      'chosen': example['chosen_response'],
      'rejected': example['rejected_response'],
  }

processed_dataset = dataset.map(format_dpo_example)

# Create Grain dataset
train_ds = grain.MapDataset.source(processed_dataset)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=True)
```

**4. Configure DPO Trainer**:
```python
from tunix.sft.dpo.dpo_trainer import DPOTrainer, DPOTrainingConfig
import optax

# Learning rate schedule
warmup_schedule = optax.linear_schedule(
    init_value=0.0,
    end_value=LEARNING_RATE,
    transition_steps=WARMUP_STEPS,
)
cosine_schedule = optax.cosine_decay_schedule(
    init_value=LEARNING_RATE,
    decay_steps=MAX_STEPS - WARMUP_STEPS,
    alpha=0.0,
)
lr_schedule = optax.join_schedules(
    [warmup_schedule, cosine_schedule],
    [WARMUP_STEPS],
)

# Optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(MAX_GRAD_NORM),
    optax.adamw(
        learning_rate=lr_schedule,
        b1=0.9,
        b2=0.99,
        weight_decay=0.1,
    ),
)

# DPO config
dpo_config = DPOTrainingConfig(
    beta=BETA,
    max_prompt_length=MAX_PROMPT_LENGTH,
    max_response_length=MAX_RESPONSE_LENGTH,
    num_train_steps=MAX_STEPS,
    eval_every_n_steps=100,
)
```

**5. Train**:
```python
# Create DPO trainer
trainer = DPOTrainer(
    policy_model=policy_model,
    reference_model=reference_model,
    train_dataset=train_ds,
    optimizer=optimizer,
    config=dpo_config,
    mesh=mesh,
)

# Training loop
for step in range(MAX_STEPS):
  metrics = trainer.train_step()
  
  if step % 10 == 0:
    print(f"Step {step}")
    print(f"  DPO Loss: {metrics['dpo_loss']:.4f}")
    print(f"  Reward Margin: {metrics['reward_margin']:.4f}")
    print(f"  KL Divergence: {metrics['kl']:.4f}")
```

**6. Evaluation**:
```python
# Generate with trained model
sampler = sampler_lib.Sampler(
    transformer=policy_model,
    tokenizer=tokenizer,
    cache_config=sampler_lib.CacheConfig(...),
)

test_prompts = [
    "Solve the following math problem: ...",
]

# Greedy generation
outputs = sampler(
    input_strings=test_prompts,
    temperature=None,  # Greedy
    max_generation_steps=100,
)

for prompt, output in zip(test_prompts, outputs.text):
  print(f"Prompt: {prompt}")
  print(f"Response: {output}")
```

#### Key Takeaways

- **No Reward Model**: DPO directly optimizes preferences
- **Reference Model**: Frozen copy prevents mode collapse
- **Beta**: Controls strength of preference optimization (0.1 typical)
- **Gradient Clipping**: Important for stability (clip at 0.1)
- **Warmup**: Linear warmup for 10% of training prevents instability

## 4.2 Script Examples

### GRPO Demo Script (grpo_demo_llama3_qwen2.py)

**File**: `scripts/grpo_demo_llama3_qwen2.py`

**Description**: Production-ready GRPO training script supporting Llama3 and Qwen2.5 models. Includes full training, evaluation, and inference pipeline.

#### Key Features

- **Multi-Model Support**: Llama3 (1B, 3B, 8B) and Qwen2.5 (0.5B, 1.5B, 3B, 7B)
- **Command-Line Interface**: Argparse-based configuration
- **Profiling Support**: Integrated JAX profiler
- **Checkpointing**: Automatic checkpoint saving and restoration
- **Metrics Logging**: TensorBoard, WandB, CLU support
- **Distributed Training**: Multi-host support with Pathways

####Command-Line Arguments

```bash
python3 grpo_demo_llama3_qwen2.py \
  --root-dir=/path/to/root_dir \
  --model-version=Qwen/Qwen2.5-0.5B-Instruct \
  --enable-profiler=False \
  --profiler-skip-first-n-steps=2 \
  --profiler-steps=2 \
  --batch-size=4 \
  --num-batches=100 \
  --learning-rate=1e-5 \
  --lora-rank=64 \
  --lora-alpha=64.0
```

#### Script Structure

**1. Argument Parsing**:
```python
parser = argparse.ArgumentParser(description="Arguments for GRPO demo")
parser.add_argument("--root-dir", type=str, help="Root directory")
parser.add_argument("--model-version", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--enable-profiler", type=bool, default=False)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--num-batches", type=int, default=100)
parser.add_argument("--learning-rate", type=float, default=1e-5)
parser.add_argument("--lora-rank", type=int, default=64)
parser.add_argument("--lora-alpha", type=float, default=64.0)
args = parser.parse_args()
```

**2. Model Selection**:
```python
def load_model(model_version, mesh):
  """Load model based on version string."""
  if "Llama" in model_version:
    # Llama3 models
    if "1B" in model_version:
      config = llama_lib.ModelConfig.llama3p2_1b()
    elif "3B" in model_version:
      config = llama_lib.ModelConfig.llama3p2_3b()
    else:
      config = llama_lib.ModelConfig.llama3_8b()
    
    model = llama_params.create_model_from_safe_tensors(
        model_path, config, mesh
    )
  
  elif "Qwen" in model_version:
    # Qwen2.5 models
    if "0.5B" in model_version:
      config = qwen2_lib.ModelConfig.qwen25_0p5b()
    elif "1.5B" in model_version:
      config = qwen2_lib.ModelConfig.qwen25_1p5b()
    elif "3B" in model_version:
      config = qwen2_lib.ModelConfig.qwen25_3b()
    else:
      config = qwen2_lib.ModelConfig.qwen25_7b()
    
    model = qwen2_params.create_model_from_safe_tensors(
        model_path, config, mesh
    )
  
  return model, config

model, model_config = load_model(args.model_version, mesh)
```

**3. Dataset Creation**:
```python
from tunix.examples.data import math_dataset

train_ds, eval_ds = script_utils.get_train_and_eval_datasets(
    data_path=train_data_path,
    split="train",
    seed=42,
    system_prompt="",
    batch_size=args.batch_size,
    num_batches=args.num_batches,
    train_fraction=0.9,
    num_epochs=1,
    answer_extractor=math_dataset.extract_answer,
    dataset_name='gsm8k',
)
```

**4. Apply LoRA**:
```python
import qwix

lora_provider = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum|.*gate_proj|.*down_proj|.*up_proj",
    rank=args.lora_rank,
    alpha=args.lora_alpha,
)

with mesh:
  lora_model = lora_provider(model)
```

**5. Setup GRPO**:
```python
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner

# GRPO config
grpo_config = grpo_learner.GRPOConfig(
    num_iterations=1,
    beta=0.001,
    epsilon=0.2,
    learning_rate=args.learning_rate,
    max_steps=args.num_batches,
)

# Optimizer
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(
        learning_rate=grpo_config.learning_rate,
        b1=0.9,
        b2=0.999,
        weight_decay=0.01,
    ),
)

# Rollout config
rollout_config = base_rollout.RolloutConfig(
    max_prompt_length=256,
    total_generation_steps=768,
    temperature=0.6,
    top_p=0.95,
    top_k=50,
    num_generations=2,
)

# Create RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    model=lora_model,
    optimizer=optimizer,
    rollout_config=rollout_config,
    reward_fn=compute_rewards,
    mesh=mesh,
)
```

**6. Profiling Setup**:
```python
if args.enable_profiler:
  profiler_opts = profiler.ProfilerOptions(
      log_dir=os.path.join(args.root_dir, "profiles"),
      skip_first_n_steps=args.profiler_skip_first_n_steps,
      profiler_steps=args.profiler_steps,
  )
  prof = profiler.Profiler(0, args.num_batches, profiler_opts)
else:
  prof = None
```

**7. Training Loop**:
```python
learner = grpo_learner.GRPOLearner(config=grpo_config, rl_cluster=rl_cluster)

for step, batch in enumerate(tqdm(train_ds)):
  # Profiling
  if prof:
    prof.maybe_activate(step)
  
  # Training step
  metrics = learner.train_step(batch)
  
  # Logging
  if step % 10 == 0:
    logging.info(f"Step {step}")
    logging.info(f"  Loss: {metrics['loss']:.4f}")
    logging.info(f"  Reward: {metrics['mean_reward']:.4f}")
    logging.info(f"  KL: {metrics['kl']:.4f}")
  
  # Checkpointing
  if step % 100 == 0 and jax.process_index() == 0:
    checkpoint_manager.save(step, lora_model)
  
  # Profiling
  if prof:
    prof.maybe_deactivate(step)
  
  if step >= args.num_batches:
    break
```

**8. Evaluation**:
```python
# Evaluate on test set
logging.info("Evaluating model...")
correct = 0
total = 0

for batch in tqdm(eval_ds):
  responses = rl_cluster.generate(batch['prompts'])
  
  for response, gt_answer in zip(responses, batch['answer']):
    predicted = extract_answer_from_response(response)
    if predicted == gt_answer:
      correct += 1
    total += 1

accuracy = correct / total
logging.info(f"Test Accuracy: {accuracy:.2%}")
```

### Llama3 Example (llama3_example.py)

**File**: `scripts/llama3_example.py`

**Description**: Simple example demonstrating model loading and inference with Llama3.

#### Script Content

```python
"""Example of using tunix to load and run Llama3 models."""

import os
import tempfile
from flax import nnx
import jax
import transformers
from tunix.generate import sampler
from tunix.models.llama3 import model
from tunix.models.llama3 import params
from tunix.tests import test_common as tc

MODEL_VERSION = "meta-llama/Llama-3.2-1B-Instruct"

# Download model
temp_dir = tempfile.gettempdir()
MODEL_CP_PATH = os.path.join(temp_dir, "models", MODEL_VERSION)
all_files = tc.download_from_huggingface(
    repo_id=MODEL_VERSION, 
    model_path=MODEL_CP_PATH
)

# Create mesh
mesh = jax.make_mesh(
    (1, len(jax.devices())),
    ("fsdp", "tp"),
    axis_types=(jax.sharding.AxisType.Auto,) * 2,
)

# Load model
config = model.ModelConfig.llama3p2_1b()
llama3 = params.create_model_from_safe_tensors(MODEL_CP_PATH, config, mesh)
nnx.display(llama3)

# Load tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_CP_PATH)
tokenizer.pad_token_id = 0

# Prepare inputs
inputs = tc.batch_templatize([
    "tell me about world war 2",
    "印度的首都在哪里",  # "What is the capital of India?" in Chinese
    "tell me your name, respond in Chinese",
], tokenizer)

# Create sampler
sampler = sampler.Sampler(
    llama3,
    tokenizer,
    sampler.CacheConfig(
        cache_size=256,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
    ),
)

# Generate
out = sampler(inputs, max_generation_steps=128, echo=True, top_p=None)

# Print outputs
for t in out.text:
  print(t)
  print("*" * 30)
```

#### Key Takeaways

- **Minimal Example**: Clean, simple demonstration of core functionality
- **Multi-Language**: Supports multilingual inputs (English, Chinese)
- **Mesh Creation**: Automatic device configuration
- **Echo Mode**: Returns prompt + generation

### Setup Scripts

#### TPU Notebook Setup (setup_notebook_tpu_single_host.sh)

**File**: `scripts/setup_notebook_tpu_single_host.sh`

**Purpose**: Sets up TPU VM environment for Jupyter notebooks.

```bash
#!/bin/bash

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip

# Install JAX with TPU support
pip install -U "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

# Install Tunix and dependencies
pip install git+https://github.com/google/tunix
pip install git+https://github.com/google/qwix
pip install git+https://github.com/google/flax

# Install additional dependencies
pip install tensorflow tensorflow_datasets
pip install grain-nightly
pip install transformers
pip install datasets
pip install kagglehub
pip install wandb
pip install tensorboardX

# Setup Jupyter
pip install jupyter
pip install ipywidgets

# Start Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

#### vLLM Installation (install_tunix_vllm_requirement.sh)

**File**: `scripts/install_tunix_vllm_requirement.sh`

**Purpose**: Installs vLLM dependencies for Tunix RL rollouts.

```bash
#!/bin/bash

# Install vLLM with JAX support
pip install vllm
pip install vllm[jax]

# Install additional requirements
pip install ray
pip install aiohttp

# Verify installation
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
```

## 4.3 Agentic Examples

### Gemma GRPO Agentic Demo

**File**: `examples/agentic/gemma_grpo_demo_nb.py`

**Description**: Demonstrates agentic GRPO training with async rollouts, enabling multi-turn conversations and tool usage.

#### Key Concepts

**Agentic RL**:
- Multi-turn conversations (not just single-shot Q&A)
- Tool usage integration
- Async rollout for efficiency
- Structured output parsing

**Async Rollout**:
- Decouples generation from training
- Enables parallel rollouts across multiple workers
- Reduces idle time during training

#### Structure

**1. Environment Setup**:
```python
# Environment detection (G3 vs OSS)
try:
  from GOOGLE_INTERNAL_PACKAGE_PATH.pyglib import gfile
  ENV = 'g3'
except ImportError:
  ENV = 'oss'

# Import agentic components
from tunix.rl.agentic.parser.chat_template_parser import parser
from tunix.rl.experimental.agentic_grpo_learner import GRPOConfig, GRPOLearner
```

**2. Hyperparameters**:
```python
# Data paths (different for G3 vs OSS)
if ENV == 'g3':
  TRAIN_DATA_PATH = '/GOOGLE_INTERNAL_PATH/gsm8k_train.json'
  TEST_DATA_PATH = '/GOOGLE_INTERNAL_PATH/gsm8k_test.json'
  NNX_CKPT_DIR = '/GOOGLE_INTERNAL_PATH/gemma2/nnx/'
else:
  TRAIN_DATA_PATH = './data/train'
  TEST_DATA_PATH = './data/test'
  MODEL_DOWNLOAD_PATH = '/tmp/content/model_download/'
  NNX_CKPT_DIR = '/tmp/content/intermediate_ckpt/'

# Agentic GRPO config
MAX_TURNS = 3  # Multi-turn conversations
ASYNC_ROLLOUT_WORKERS = 4  # Number of async rollout workers
TOOL_TIMEOUT = 30  # Tool execution timeout (seconds)
```

**3. Multi-Turn Rollout**:
```python
class MultiTurnRollout:
  """Handles multi-turn conversations with tools."""
  
  def __init__(self, model, tokenizer, tools, max_turns=3):
    self.model = model
    self.tokenizer = tokenizer
    self.tools = tools
    self.max_turns = max_turns
  
  def rollout(self, prompt):
    """Execute multi-turn conversation."""
    conversation_history = [{"role": "user", "content": prompt}]
    
    for turn in range(self.max_turns):
      # Generate response
      response = self.model.generate(conversation_history)
      
      # Parse for tool calls
      tool_calls = parser.extract_tool_calls(response)
      
      if not tool_calls:
        # No tools requested, return response
        return conversation_history + [{"role": "assistant", "content": response}]
      
      # Execute tool calls
      tool_results = []
      for tool_call in tool_calls:
        result = self.execute_tool(tool_call)
        tool_results.append(result)
      
      # Add tool results to conversation
      conversation_history.append({"role": "assistant", "content": response})
      conversation_history.append({"role": "tool", "content": str(tool_results)})
    
    return conversation_history
  
  def execute_tool(self, tool_call):
    """Execute a tool and return result."""
    tool_name = tool_call['name']
    tool_args = tool_call['arguments']
    
    if tool_name in self.tools:
      return self.tools[tool_name](**tool_args)
    else:
      return {"error": f"Tool {tool_name} not found"}
```

**4. Async Rollout Manager**:
```python
import asyncio

class AsyncRolloutManager:
  """Manages async rollouts across multiple workers."""
  
  def __init__(self, model, num_workers=4):
    self.model = model
    self.num_workers = num_workers
    self.rollout_queue = asyncio.Queue()
    self.result_queue = asyncio.Queue()
  
  async def worker(self, worker_id):
    """Async worker that processes rollouts."""
    while True:
      # Get batch from queue
      batch = await self.rollout_queue.get()
      
      # Generate responses
      responses = self.model.generate(batch)
      
      # Compute rewards
      rewards = compute_rewards(responses)
      
      # Put results in result queue
      await self.result_queue.put((responses, rewards))
      
      self.rollout_queue.task_done()
  
  async def start_workers(self):
    """Start all workers."""
    workers = [
        asyncio.create_task(self.worker(i))
        for i in range(self.num_workers)
    ]
    return workers
  
  async def submit_rollout(self, batch):
    """Submit batch for rollout."""
    await self.rollout_queue.put(batch)
  
  async def get_result(self):
    """Get rollout result."""
    return await self.result_queue.get()
```

**5. Training with Async Rollouts**:
```python
# Create async rollout manager
rollout_manager = AsyncRolloutManager(model, num_workers=ASYNC_ROLLOUT_WORKERS)

# Start workers
await rollout_manager.start_workers()

# Training loop
for step, batch in enumerate(train_ds):
  # Submit rollouts (non-blocking)
  await rollout_manager.submit_rollout(batch)
  
  # Get results from previous rollouts
  if step > 0:
    responses, rewards = await rollout_manager.get_result()
    
    # Update policy
    metrics = learner.train_step(responses, rewards)
    
    print(f"Step {step}, Reward: {metrics['mean_reward']:.4f}")
```

### Multi-Turn RL

**Key Components**:

**1. Conversation State**:
```python
@dataclasses.dataclass
class ConversationState:
  """Tracks multi-turn conversation state."""
  messages: list[dict]  # Conversation history
  tools_used: list[str]  # Tools called
  turn: int  # Current turn number
  reward: float = 0.0  # Accumulated reward
```

**2. Turn-Based Rewards**:
```python
def compute_turn_rewards(conversation_state):
  """Compute rewards for multi-turn conversation."""
  rewards = []
  
  # Reward for each turn
  for turn in range(conversation_state.turn):
    # Base reward for valid turn
    turn_reward = 0.1
    
    # Bonus for using tools correctly
    if any(tool in conversation_state.tools_used for tool in ['calculator', 'search']):
      turn_reward += 0.2
    
    # Penalty for too many turns
    if turn > 5:
      turn_reward -= 0.1
    
    rewards.append(turn_reward)
  
  # Final reward based on answer correctness
  if is_correct_answer(conversation_state.messages[-1]):
    rewards[-1] += 1.0
  
  return jnp.array(rewards)
```

### Tool Usage Patterns

**Example Tools**:

**1. Calculator Tool**:
```python
def calculator(expression: str) -> dict:
  """Evaluate mathematical expression."""
  try:
    result = eval(expression)  # Use safe eval in production
    return {"result": result, "status": "success"}
  except Exception as e:
    return {"error": str(e), "status": "error"}
```

**2. Search Tool**:
```python
def search(query: str) -> dict:
  """Search for information."""
  # Placeholder - integrate with actual search API
  results = [
      {"title": "Result 1", "snippet": "..."},
      {"title": "Result 2", "snippet": "..."},
  ]
  return {"results": results, "status": "success"}
```

**3. Code Execution Tool**:
```python
import subprocess
import tempfile

def execute_code(code: str, language: str = "python") -> dict:
  """Execute code safely in isolated environment."""
  if language != "python":
    return {"error": "Only Python supported", "status": "error"}
  
  try:
    # Write code to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
      f.write(code)
      temp_file = f.name
    
    # Execute with timeout
    result = subprocess.run(
        ["python", temp_file],
        capture_output=True,
        text=True,
        timeout=5,
    )
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "status": "success" if result.returncode == 0 else "error",
    }
  except subprocess.TimeoutExpired:
    return {"error": "Execution timeout", "status": "error"}
  finally:
    os.unlink(temp_file)
```

**Tool Registry**:
```python
TOOLS = {
    "calculator": {
        "function": calculator,
        "description": "Evaluate mathematical expressions",
        "parameters": {
            "expression": {"type": "string", "description": "Math expression to evaluate"}
        },
    },
    "search": {
        "function": search,
        "description": "Search for information",
        "parameters": {
            "query": {"type": "string", "description": "Search query"}
        },
    },
    "execute_code": {
        "function": execute_code,
        "description": "Execute Python code",
        "parameters": {
            "code": {"type": "string", "description": "Python code to execute"},
            "language": {"type": "string", "description": "Programming language (default: python)"},
        },
    },
}
```

## 4.4 DeepScaler Examples

### Training DeepScaler

**File**: `examples/deepscaler/train_deepscaler_nb.py`

**Description**: Reproduction of DeepScaler paper using single-turn agentic framework for math reasoning.

#### Key Concepts

**DeepScaler**:
- Scaling RL for math reasoning
- Uses 1.5B Qwen2.5 model
- Achieves competitive performance with much larger models
- Single-turn generation with structured output

#### Hyperparameters

```python
# Data
TRAIN_FRACTION = 1.0
SEED = 42

# LoRA
RANK = 64
ALPHA = 64.0
TRAIN_WITH_LORA = False

# Sharding
MESH = [(2, 4), ("fsdp", "tp")]  # 2-way FSDP, 4-way TP

# GRPO Generation
MAX_PROMPT_LENGTH = 2048
TOTAL_GENERATION_STEPS = 8192
TEMPERATURE = 0.6  # Moderate temperature
TOP_P = 0.95
TOP_K = 50
NUM_GENERATIONS = 2

# GRPO Training
NUM_ITERATIONS = 1
BETA = 0.001  # KL penalty
EPSILON = 0.2  # PPO clipping

# Training
BATCH_SIZE = 32
MINI_BATCH_SIZE = 32
NUM_BATCHES = 100
```

#### Training Loop

```python
from tunix.rl.experimental.agentic_grpo_learner import GRPOConfig, GRPOLearner
from tunix.utils import math_rewards

# Configure GRPO
grpo_config = GRPOConfig(
    num_iterations=NUM_ITERATIONS,
    beta=BETA,
    epsilon=EPSILON,
    learning_rate=1e-5,
    max_steps=NUM_BATCHES,
)

# Math reward function
def compute_math_rewards(responses, ground_truth_answers):
  """Reward function for math problems."""
  return math_rewards.compute_rewards(
      responses=responses,
      ground_truth_answers=ground_truth_answers,
      answer_extraction_fn=extract_answer_from_response,
  )

# Create learner
learner = GRPOLearner(
    config=grpo_config,
    rl_cluster=rl_cluster,
    reward_fn=compute_math_rewards,
)

# Train
for step, batch in enumerate(tqdm(train_ds)):
  metrics = learner.train_step(batch)
  
  if step % 10 == 0:
    print(f"Step {step}")
    print(f"  Reward: {metrics['mean_reward']:.4f}")
    print(f"  KL: {metrics['kl']:.4f}")
```

### Math Evaluation

**File**: `examples/deepscaler/math_eval_nb.py`

**Description**: Comprehensive math evaluation suite for testing models on various benchmarks.

#### Supported Benchmarks

- **GSM8K**: Grade school math problems
- **MATH**: Competition-level math problems
- **Math500**: Curated math evaluation set

#### Evaluation Pipeline

**1. Load Model**:
```python
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params_lib

# Load Qwen2.5 model
model_config = qwen2_lib.ModelConfig.qwen25_1p5b()
model = qwen2_params_lib.create_model_from_safe_tensors(
    model_path, model_config, mesh
)
```

**2. Load Dataset**:
```python
import datasets

# Load GSM8K test set
dataset = datasets.load_dataset("openai/gsm8k", "main", split="test")

# Convert to Grain dataset
test_ds = grain.MapDataset.source(dataset).batch(4)
```

**3. Answer Extraction**:
```python
def extract_answer_robust(passage: str) -> str:
  """Extract answer from model response.
  
  Handles various formats:
  - \\boxed{answer}
  - The answer is: answer
  - answer (at end of response)
  """
  if not passage:
    return ""
  
  # Pattern 1: \\boxed{...}
  i = passage.find("\\boxed")
  if i != -1:
    i += 6
    while i < len(passage) and passage[i].isspace():
      i += 1
    if i < len(passage) and passage[i] == "{":
      i += 1
      start = i
      brace_count = 1
      while i < len(passage) and brace_count > 0:
        if passage[i] == "{":
          brace_count += 1
        elif passage[i] == "}":
          brace_count -= 1
        i += 1
      if brace_count == 0:
        return passage[start : i - 1].strip()
  
  # Pattern 2: "The answer is: ..."
  patterns = [
      r"The answer is:?\s*(.+?)(?:\.|$)",
      r"answer is:?\s*(.+?)(?:\.|$)",
      r"= (.+?)(?:\.|$)",
  ]
  
  for pattern in patterns:
    matches = re.findall(pattern, passage, re.IGNORECASE)
    if matches:
      return matches[-1].strip()
  
  # Pattern 3: Last number in response
  numbers = re.findall(r'-?\d+(?:\.\d+)?', passage)
  if numbers:
    return numbers[-1]
  
  return ""
```

**4. Evaluation Loop**:
```python
from tqdm.auto import tqdm

correct = 0
total = 0
results = []

for batch in tqdm(test_ds):
  # Generate responses
  responses = sampler(
      input_strings=batch['question'],
      max_generation_steps=512,
      temperature=0.0,  # Greedy for evaluation
  )
  
  # Extract and check answers
  for question, response, gt_answer in zip(
      batch['question'], 
      responses.text, 
      batch['answer']
  ):
    predicted_answer = extract_answer_robust(response)
    gt_extracted = extract_answer_robust(gt_answer)
    
    is_correct = predicted_answer == gt_extracted
    correct += is_correct
    total += 1
    
    results.append({
        'question': question,
        'response': response,
        'predicted': predicted_answer,
        'ground_truth': gt_extracted,
        'correct': is_correct,
    })

accuracy = correct / total
print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
```

**5. Error Analysis**:
```python
import pandas as pd

# Create DataFrame
df = pd.DataFrame(results)

# Analyze errors
errors = df[df['correct'] == False]

print(f"Total errors: {len(errors)}")
print("\nError categories:")

# Categorize errors
def categorize_error(row):
  if not row['predicted']:
    return "No answer extracted"
  elif row['predicted'].replace('.', '').isdigit():
    return "Wrong numerical answer"
  else:
    return "Invalid format"

errors['category'] = errors.apply(categorize_error, axis=1)
print(errors['category'].value_counts())

# Show sample errors
print("\nSample errors:")
for _, row in errors.head(5).iterrows():
  print(f"\nQuestion: {row['question'][:100]}...")
  print(f"Predicted: {row['predicted']}")
  print(f"Ground Truth: {row['ground_truth']}")
  print(f"Category: {row['category']}")
```

**6. Benchmark Comparison**:
```python
# Compare across multiple checkpoints
checkpoints = [
    "checkpoint_step_0",
    "checkpoint_step_100",
    "checkpoint_step_200",
]

results_by_checkpoint = {}

for ckpt in checkpoints:
  # Load checkpoint
  model = load_checkpoint(ckpt)
  
  # Evaluate
  accuracy = evaluate(model, test_ds)
  
  results_by_checkpoint[ckpt] = accuracy
  print(f"{ckpt}: {accuracy:.2%}")

# Plot learning curve
import matplotlib.pyplot as plt

steps = [0, 100, 200]
accuracies = list(results_by_checkpoint.values())

plt.plot(steps, accuracies, marker='o')
plt.xlabel('Training Steps')
plt.ylabel('Accuracy')
plt.title('GSM8K Evaluation During Training')
plt.grid(True)
plt.show()
```

## Usage Patterns

### From Notebook to Production Script

**Progression Path**:

1. **Prototyping (Jupyter Notebook)**:
   ```python
   # Quick experimentation
   model = load_model_from_hf("google/gemma-2b-it")
   
   # Interactive testing
   response = sampler(["test prompt"])
   print(response.text[0])
   ```

2. **Modularization (Python Module)**:
   ```python
   # model_loader.py
   def load_model(model_name, mesh):
     """Load and shard model."""
     # ... implementation
     return model
   
   # train.py
   from model_loader import load_model
   
   def main(args):
     model = load_model(args.model_name, mesh)
     # ... training logic
   ```

3. **Production Script (CLI + Config)**:
   ```python
   # train_grpo.py
   import argparse
   import yaml
   
   def parse_args():
     parser = argparse.ArgumentParser()
     parser.add_argument("--config", type=str, required=True)
     parser.add_argument("--root-dir", type=str, required=True)
     # ... more args
     return parser.parse_args()
   
   def main():
     args = parse_args()
     config = yaml.safe_load(open(args.config))
     
     # Setup logging
     setup_logging(args.root_dir)
     
     # Load model
     model = load_model(config['model'])
     
     # Train
     train(model, config['training'])
   
   if __name__ == "__main__":
     main()
   ```

### Common Workflow Templates

#### Fine-Tuning Workflow

```python
"""Complete fine-tuning workflow template."""

# 1. Setup
import jax
from tunix.sft import peft_trainer
from tunix.models import automodel
import qwix

# Create mesh
mesh = jax.make_mesh((1, 8), ("fsdp", "tp"))

# 2. Load model
with mesh:
  model = automodel.AutoModel.from_pretrained(
      "google/gemma-2b-it",
      mesh=mesh,
  )

# 3. Apply LoRA (optional)
lora_provider = qwix.LoraProvider(
    module_path=".*q_einsum|.*kv_einsum",
    rank=64,
    alpha=64.0,
)
model = lora_provider(model)

# 4. Setup data pipeline
train_ds = create_dataset(
    path="data/train.json",
    batch_size=32,
    sequence_length=512,
)

# 5. Configure trainer
trainer_config = peft_trainer.TrainerConfig(
    learning_rate=1e-4,
    num_steps=1000,
    warmup_steps=100,
)

optimizer = optax.adamw(
    learning_rate=trainer_config.learning_rate,
    weight_decay=0.01,
)

# 6. Create trainer
trainer = peft_trainer.Trainer(
    model=model,
    optimizer=optimizer,
    config=trainer_config,
    mesh=mesh,
)

# 7. Train
for step, batch in enumerate(train_ds):
  metrics = trainer.train_step(batch)
  
  if step % 10 == 0:
    print(f"Step {step}, Loss: {metrics['loss']:.4f}")
  
  if step % 100 == 0:
    checkpoint_manager.save(step, model)

# 8. Evaluate
eval_metrics = trainer.evaluate(eval_ds)
print(f"Eval Loss: {eval_metrics['loss']:.4f}")

# 9. Save final model
checkpoint_manager.save('final', model)
```

#### RL Workflow

```python
"""Complete RL workflow template."""

# 1. Setup
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo import grpo_learner
from tunix.rl import base_rollout

# 2. Load model (similar to fine-tuning)
model = load_model(...)

# 3. Define reward function
def compute_rewards(prompts, responses, ground_truth):
  """Compute rewards for RL."""
  rewards = []
  for response, gt in zip(responses, ground_truth):
    # Extract answer
    predicted = extract_answer(response)
    
    # Binary reward
    reward = 1.0 if predicted == gt else 0.0
    
    rewards.append(reward)
  
  return jnp.array(rewards)

# 4. Configure rollout
rollout_config = base_rollout.RolloutConfig(
    max_prompt_length=256,
    total_generation_steps=768,
    temperature=0.6,
    top_p=0.95,
    num_generations=2,
)

# 5. Configure GRPO
grpo_config = grpo_learner.GRPOConfig(
    num_iterations=1,
    beta=0.001,
    epsilon=0.2,
    learning_rate=1e-5,
)

# 6. Create RL cluster
rl_cluster = rl_cluster_lib.RLCluster(
    model=model,
    optimizer=optimizer,
    rollout_config=rollout_config,
    reward_fn=compute_rewards,
    mesh=mesh,
)

# 7. Create learner
learner = grpo_learner.GRPOLearner(
    config=grpo_config,
    rl_cluster=rl_cluster,
)

# 8. Train
for step, batch in enumerate(train_ds):
  metrics = learner.train_step(batch)
  
  if step % 10 == 0:
    print(f"Step {step}")
    print(f"  Reward: {metrics['mean_reward']:.4f}")
    print(f"  KL: {metrics['kl']:.4f}")
```

#### Distillation Workflow

```python
"""Complete distillation workflow template."""

# 1. Load teacher model
teacher = automodel.AutoModel.from_pretrained(
    "google/gemma-7b-it",
    mesh=mesh,
)

# 2. Load student model
student = automodel.AutoModel.from_pretrained(
    "google/gemma-2b-it",
    mesh=mesh,
)

# 3. Configure distillation strategy
from tunix.distillation.strategies import logit_strategy

strategy = logit_strategy.LogitStrategy(
    temperature=2.0,
    alpha=0.7,  # 70% distillation, 30% task loss
)

# 4. Create distillation trainer
from tunix.distillation import distillation_trainer

trainer = distillation_trainer.DistillationTrainer(
    teacher=teacher,
    student=student,
    strategy=strategy,
    optimizer=optimizer,
    mesh=mesh,
)

# 5. Train
for step, batch in enumerate(train_ds):
  metrics = trainer.train_step(batch)
  
  if step % 10 == 0:
    print(f"Distillation Loss: {metrics['distillation_loss']:.4f}")
    print(f"Task Loss: {metrics['task_loss']:.4f}")
```

### Adapting Examples

#### Changing Models

**From Gemma to Llama3**:

```python
# Original (Gemma)
from tunix.models.gemma import model as gemma_lib
from tunix.models.gemma import params as gemma_params_lib

config = gemma_lib.ModelConfig.gemma2_2b()
model = gemma_params_lib.create_model_from_safe_tensors(
    model_path, config, mesh
)

# Adapted (Llama3)
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params_lib

config = llama_lib.ModelConfig.llama3p2_1b()
model = llama_params_lib.create_model_from_safe_tensors(
    model_path, config, mesh
)
```

**From Llama3 to Qwen2**:

```python
# Original (Llama3)
from tunix.models.llama3 import model as llama_lib
from tunix.models.llama3 import params as llama_params_lib

config = llama_lib.ModelConfig.llama3_8b()
model = llama_params_lib.create_model_from_safe_tensors(
    model_path, config, mesh
)

# Adapted (Qwen2)
from tunix.models.qwen2 import model as qwen2_lib
from tunix.models.qwen2 import params as qwen2_params_lib

config = qwen2_lib.ModelConfig.qwen25_7b()
model = qwen2_params_lib.create_model_from_safe_tensors(
    model_path, config, mesh
)
```

#### Changing Datasets

**From GSM8K to Custom Math Dataset**:

```python
# Original (GSM8K with script_utils)
train_ds, eval_ds = script_utils.get_train_and_eval_datasets(
    data_path=train_data_path,
    split="train",
    seed=42,
    system_prompt="",
    batch_size=32,
    dataset_name='gsm8k',
)

# Adapted (Custom JSON dataset)
import json
import grain

def load_custom_math_dataset(path, batch_size):
  """Load custom math dataset."""
  # Load JSON
  with open(path) as f:
    data = json.load(f)
  
  # Create Grain dataset
  ds = grain.MapDataset.source(data)
  
  # Transform
  def transform(example):
    return {
        'prompt': f"Question: {example['question']}\\nAnswer:",
        'answer': example['answer'],
    }
  
  ds = ds.map(transform)
  ds = ds.batch(batch_size)
  ds = ds.repeat(None)  # Infinite iteration
  
  return ds

train_ds = load_custom_math_dataset("data/train.json", batch_size=32)
```

**From Translation to Instruction Following**:

```python
# Original (Translation with TFDS)
import tensorflow_datasets as tfds

ds = tfds.load("mtnt/en-fr", split="train")

def preprocess_translation(example):
  return {
      'input': example['src'],
      'target': example['tgt'],
  }

# Adapted (Instruction following from JSON)
import datasets

ds = datasets.load_dataset("json", data_files="instructions.json")

def preprocess_instruction(example):
  # Format as instruction
  prompt = f"### Instruction:\n{example['instruction']}\n\n"
  if example.get('input'):
    prompt += f"### Input:\n{example['input']}\n\n"
  prompt += "### Response:\n"
  
  return {
      'input': prompt,
      'target': example['output'],
  }

ds = ds.map(preprocess_instruction)
```

#### Changing Hyperparameters

**Guidelines for Adjusting Hyperparameters**:

| Hyperparameter | Small Model (<3B) | Medium Model (3-7B) | Large Model (>7B) |
|----------------|-------------------|---------------------|-------------------|
| **Learning Rate** | 1e-4 to 5e-4 | 5e-5 to 1e-4 | 1e-5 to 5e-5 |
| **Batch Size** | 32-64 | 16-32 | 4-16 |
| **LoRA Rank** | 32-64 | 64-128 | 128-256 |
| **GRPO Beta** | 0.001-0.01 | 0.001-0.01 | 0.0001-0.001 |
| **DPO Beta** | 0.1-0.5 | 0.1-0.3 | 0.05-0.1 |

**Scaling Example**:

```python
# Original (2B model)
CONFIG_2B = {
    'learning_rate': 2e-4,
    'batch_size': 64,
    'lora_rank': 64,
    'lora_alpha': 64.0,
    'warmup_steps': 100,
}

# Scaled (7B model)
CONFIG_7B = {
    'learning_rate': 5e-5,  # Lower LR for larger model
    'batch_size': 16,  # Smaller batch due to memory
    'lora_rank': 128,  # Higher rank for more capacity
    'lora_alpha': 128.0,  # Scale with rank
    'warmup_steps': 200,  # More warmup for stability
}

# Scaled (1B model)
CONFIG_1B = {
    'learning_rate': 5e-4,  # Higher LR for smaller model
    'batch_size': 128,  # Larger batch for efficiency
    'lora_rank': 32,  # Lower rank sufficient
    'lora_alpha': 32.0,  # Scale with rank
    'warmup_steps': 50,  # Less warmup needed
}
```

### Hardware-Specific Adaptations

#### Single GPU

```python
# Adjust mesh for single GPU
mesh = jax.make_mesh((1, 1), ("fsdp", "tp"))

# Reduce batch size
BATCH_SIZE = 4

# Enable gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 8

# Reduce sequence length
MAX_SEQ_LENGTH = 512
```

#### Multi-GPU (Single Host)

```python
# Use all GPUs
num_devices = len(jax.devices())
mesh = jax.make_mesh((1, num_devices), ("fsdp", "tp"))

# Scale batch size
BATCH_SIZE = 32

# Enable mixed precision
USE_BF16 = True
```

#### TPU Pod

```python
# Multi-host mesh
num_hosts = jax.process_count()
devices_per_host = len(jax.local_devices())
mesh = jax.make_mesh(
    (num_hosts, devices_per_host),
    ("fsdp", "tp"),
)

# Large batch size
BATCH_SIZE = 256

# Enable all optimizations
USE_BF16 = True
ENABLE_ASYNC_CHECKPOINT = True
```

## Best Practices

### Development Workflow

**1. Start Simple**:
```python
# ✅ DO: Start with small model and dataset
model = load_model("google/gemma-2b-it")
train_ds = load_dataset("train.json[:100]")  # Small subset

# ❌ DON'T: Start with full-scale training
model = load_model("google/gemma-27b")
train_ds = load_dataset("train.json")  # Full dataset
```

**2. Validate Before Scaling**:
```python
# ✅ DO: Test on small scale first
def test_training_pipeline():
  """Test pipeline with minimal resources."""
  model = load_model_small()
  train_ds = load_dataset(num_examples=10)
  
  # Run 5 steps
  for step in range(5):
    metrics = trainer.train_step(train_ds[step])
    assert metrics['loss'] < 100, "Loss exploded!"
  
  print("Pipeline validated ✓")

test_training_pipeline()

# Now scale up
model = load_model_full()
train_ds = load_dataset(full=True)
```

**3. Use Checkpointing Early**:
```python
# ✅ DO: Setup checkpointing from the start
from tunix.sft import checkpoint_manager as cm

ckpt_manager = cm.CheckpointManager(
    directory="checkpoints/",
    max_to_keep=3,
)

for step in range(num_steps):
  metrics = trainer.train_step(batch)
  
  if step % 100 == 0:
    ckpt_manager.save(step, model)

# ❌ DON'T: Start checkpointing after training
# (Too late if training crashes!)
```

### Performance Optimization

**1. Profiling**:
```python
# ✅ DO: Profile to identify bottlenecks
from tunix.perf import profiler

prof_opts = profiler.ProfilerOptions(
    log_dir="profiles/",
    skip_first_n_steps=5,  # Skip warmup
    profiler_steps=3,  # Profile 3 steps
)

prof = profiler.Profiler(0, num_steps, prof_opts)

for step in range(num_steps):
  prof.maybe_activate(step)
  metrics = trainer.train_step(batch)
  prof.maybe_deactivate(step)

# View in TensorBoard
# tensorboard --logdir=profiles/
```

**2. Optimize Data Loading**:
```python
# ✅ DO: Use Grain for efficient data loading
import grain

ds = grain.MapDataset.source(data)
ds = ds.shuffle(seed=42)
ds = ds.map(preprocess, num_parallel_calls=8)  # Parallel preprocessing
ds = ds.batch(batch_size)
ds = ds.prefetch(2)  # Prefetch batches

# ❌ DON'T: Use inefficient data loading
def slow_data_loader():
  for item in data:
    preprocessed = preprocess(item)  # Sequential
    yield preprocessed
```

**3. Compilation Caching**:
```python
# ✅ DO: Reuse compiled functions
@functools.lru_cache(maxsize=1)
def get_train_step():
  """Cache compiled train step."""
  @jax.jit
  def train_step(state, batch):
    # ... training logic
    return new_state, metrics
  
  return train_step

train_step = get_train_step()

# ❌ DON'T: Recompile on every call
for step in range(num_steps):
  @jax.jit  # Recompiles every time!
  def train_step(state, batch):
    # ...
    pass
```

### Memory Management

**1. Use Gradient Checkpointing**:
```python
# ✅ DO: Enable gradient checkpointing for large models
from flax import nnx

model = nnx.Transformer(
    ...,
    scan_layers=True,  # Checkpoint activations
)
```

**2. Monitor Memory Usage**:
```python
# ✅ DO: Track memory throughout training
import jax

for step in range(num_steps):
  metrics = trainer.train_step(batch)
  
  if step % 10 == 0:
    # Check memory
    memory_stats = jax.local_devices()[0].memory_stats()
    used_gb = memory_stats['bytes_in_use'] / 1e9
    limit_gb = memory_stats['bytes_limit'] / 1e9
    
    print(f"Memory: {used_gb:.2f}/{limit_gb:.2f} GB ({used_gb/limit_gb*100:.1f}%)")
    
    # Alert if near limit
    if used_gb / limit_gb > 0.9:
      logging.warning("Memory usage above 90%!")
```

**3. Use Mixed Precision**:
```python
# ✅ DO: Use BF16 for memory and speed
model = automodel.AutoModel.from_pretrained(
    "google/gemma-2b-it",
    mesh=mesh,
    dtype=jnp.bfloat16,  # BF16 for activations
)

# Parameters typically stay in FP32 for stability
```

### Debugging

**1. Gradual Complexity**:
```python
# ✅ DO: Add complexity incrementally
# Step 1: Basic model loading
model = load_model("google/gemma-2b-it")
print("Model loaded ✓")

# Step 2: Add LoRA
model = apply_lora(model)
print("LoRA applied ✓")

# Step 3: Simple forward pass
output = model(dummy_input)
print("Forward pass ✓")

# Step 4: Training step
metrics = trainer.train_step(batch)
print("Training step ✓")

# ❌ DON'T: Try everything at once
model = load_model(...)
model = apply_lora(...)
model = apply_quantization(...)
train_full_model()  # What broke?
```

**2. Use Assertions**:
```python
# ✅ DO: Add assertions to catch issues early
def train_step(batch):
  # Validate inputs
  assert batch['input_ids'].shape[0] == BATCH_SIZE, "Wrong batch size!"
  assert batch['input_ids'].shape[1] <= MAX_SEQ_LENGTH, "Sequence too long!"
  
  # Training
  loss = compute_loss(batch)
  
  # Validate outputs
  assert jnp.isfinite(loss), "Loss is NaN or Inf!"
  assert loss > 0, "Loss must be positive!"
  
  return loss
```

**3. Save Intermediate States**:
```python
# ✅ DO: Save states for debugging
import pickle

def debug_train_step(batch, step):
  """Training step with debugging."""
  # Save input
  if step == DEBUG_STEP:
    with open(f"debug_batch_{step}.pkl", "wb") as f:
      pickle.dump(batch, f)
  
  # Forward pass
  logits = model(batch['input_ids'])
  
  if step == DEBUG_STEP:
    with open(f"debug_logits_{step}.pkl", "wb") as f:
      pickle.dump(logits, f)
  
  # Compute loss
  loss = compute_loss(logits, batch['labels'])
  
  return loss
```

### Reproducibility

**1. Set All Seeds**:
```python
# ✅ DO: Set seeds everywhere
import random
import numpy as np
import jax

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
jax.random.PRNGKey(SEED)

# Also set in data loading
train_ds = load_dataset(..., seed=SEED)
```

**2. Log Everything**:
```python
# ✅ DO: Comprehensive logging
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(),
    ]
)

# Log configuration
config = {
    'model': 'google/gemma-2b-it',
    'learning_rate': 1e-4,
    'batch_size': 32,
    'lora_rank': 64,
    'seed': 42,
}

logging.info("Configuration:")
logging.info(json.dumps(config, indent=2))

# Log git commit
import subprocess
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
logging.info(f"Git commit: {commit}")
```

**3. Version Dependencies**:
```python
# ✅ DO: Log package versions
import tunix
import jax
import flax

logging.info("Package versions:")
logging.info(f"  tunix: {tunix.__version__}")
logging.info(f"  jax: {jax.__version__}")
logging.info(f"  flax: {flax.__version__}")
```

### Testing

**1. Unit Test Components**:
```python
# ✅ DO: Test individual components
def test_reward_function():
  """Test reward function."""
  # Test correct answer
  reward = compute_reward(
      response="The answer is 42.",
      ground_truth="42"
  )
  assert reward == 1.0, "Should reward correct answer"
  
  # Test incorrect answer
  reward = compute_reward(
      response="The answer is 43.",
      ground_truth="42"
  )
  assert reward == 0.0, "Should not reward incorrect answer"
  
  # Test edge cases
  reward = compute_reward(
      response="",  # Empty response
      ground_truth="42"
  )
  assert reward == 0.0, "Should handle empty response"

test_reward_function()
```

**2. Integration Test Pipeline**:
```python
# ✅ DO: Test full pipeline with small data
def test_full_pipeline():
  """Test complete training pipeline."""
  # Setup
  model = load_model_small()
  train_ds = create_test_dataset(size=10)
  trainer = create_trainer(model)
  
  # Run a few steps
  for step in range(3):
    metrics = trainer.train_step(train_ds[step])
    
    # Validate metrics
    assert 'loss' in metrics, "Missing loss metric"
    assert jnp.isfinite(metrics['loss']), "Loss is not finite"
  
  # Check model can generate
  output = model.generate(["test prompt"])
  assert len(output) > 0, "Model should generate output"
  
  print("Pipeline test passed ✓")

test_full_pipeline()
```

### Example-Specific Best Practices

**QLoRA/LoRA**:
- Start with RANK=32 for small models, RANK=64 for medium, RANK=128 for large
- Set ALPHA = RANK for initialization stability
- Apply LoRA to attention layers first (q, k, v, o)
- Add FFN layers (gate, up, down) if more capacity needed
- Use QLoRA (NF4) only when memory constrained

**GRPO**:
- NUM_GENERATIONS >= 2 for stable advantages
- TEMPERATURE=0.6-0.8 for diverse generations
- BETA=0.001-0.01 for KL penalty (tune based on task)
- Use high sequence length (1024-2048) for math reasoning
- Monitor reward distribution (should increase over time)

**Distillation**:
- TEMPERATURE=2.0 typical starting point
- ALPHA=0.7 balances distillation vs task loss
- Use larger teacher (ideally 3-4x student size)
- Match tokenizers between teacher and student
- Validate student performance matches teacher trajectory

**DPO**:
- BETA=0.1-0.5 controls preference strength
- Gradient clipping essential (clip at 0.1-1.0)
- Use warmup (10-20% of steps) to prevent instability
- Ensure balanced dataset (roughly equal chosen/rejected)
- Monitor reward margin (should increase)

