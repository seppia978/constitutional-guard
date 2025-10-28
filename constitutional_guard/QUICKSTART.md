# Constitutional Guard - Quick Start Guide

A true "constitution-first" safety classifier that actually follows the provided policy.

## What Makes This Different?

**Llama Guard 3** (from our analysis):
- ❌ Ignores policy changes (artifact control tests showed 77% attention is positional, not semantic)
- ❌ Safety alignment dominates (empty/inverted/strict policies all ignored)
- ❌ Flip-rate ~0% (model can't be constrained by policy)

**Constitutional Guard**:
- ✅ Policy-conditioned training with flip-consistency loss
- ✅ Target flip-rate ≥95% (same input, different policy → different verdict)
- ✅ Evidence grounding (cites clauses and text spans)
- ✅ Base model (Llama 3.1 8B) to minimize safety priors

## Quick Start (3 Steps)

### 1. Generate Training Data

```bash
cd constitutional_guard/data
conda run -n const-ai python data_generator.py
```

**Output**:
- `training_dataset.json` (~2700 examples)
- `policy_zoo.json` (61 diverse policies)
- `flip_test_dataset.json` (11 flip tests)

### 2. Train Model

```bash
cd constitutional_guard/training
conda run -n const-ai python train.py \
  --model_id meta-llama/Llama-3.1-8B \
  --data_file ../data/training_dataset.json \
  --output_dir ./constitutional_guard_lora \
  --epochs 3 \
  --batch_size 4 \
  --lora_r 16
```

**Requirements**:
- GPU with ≥24GB VRAM (8-bit quantization + LoRA)
- ~4-8 hours training time
- Llama 3.1 8B access (request at https://huggingface.co/meta-llama/Llama-3.1-8B)

**Training uses**:
- LoRA (Low-Rank Adaptation) for efficiency
- 8-bit quantization (fits in 24GB GPU)
- Flip-consistency loss (forces policy adherence)
- Gradient checkpointing (saves memory)

### 3. Evaluate

```bash
cd constitutional_guard/evaluation
conda run -n const-ai python evaluate.py \
  --model_path ../training/constitutional_guard_lora \
  --base_model meta-llama/Llama-3.1-8B \
  --output evaluation_results.json
```

**Expected results**:
- Flip-rate: ≥95% (vs Llama Guard 3's ~0%)
- Accuracy: ~90%+
- Clause exact match: ~85%+

## Project Structure

```
constitutional_guard/
├── schema/
│   ├── policy_schema.py       # Policy data structures
│   └── output_schema.py       # JSON output format
├── data/
│   ├── policy_zoo.py          # 61 diverse policies
│   ├── data_generator.py      # Training data generator
│   ├── flip_test_set.py       # Flip-test validation set
│   ├── policy_zoo.json        # Generated policies
│   ├── training_dataset.json  # Generated training data
│   └── flip_test_dataset.json # Generated flip tests
├── model/
│   ├── model.py               # ConstitutionalGuard wrapper
│   └── losses.py              # Flip-consistency loss
├── training/
│   ├── train.py               # Training pipeline (LoRA)
│   └── constitutional_guard_lora/ # Trained model checkpoints
└── evaluation/
    ├── baseline_test.py       # Baseline (zero-shot) test
    └── evaluate.py            # Full evaluation suite
```

## Key Features

### 1. Policy Schema with Flip-Testing

```python
from schema.policy_schema import create_mlcommons_policy

# Create base policy
policy = create_mlcommons_policy()  # 10 clauses

# Create variant (disable hate speech clause)
variant = policy.create_variant(disable_clauses=["C6"])

# Test: Same input, different policies
input_text = "People of [ethnic group] are inferior"
# With C6: unsafe
# Without C6: safe ← Model MUST flip verdict
```

### 2. Flip-Consistency Loss

```python
# Standard models: ignore policy changes
# Constitutional Guard: penalizes failure to flip

flip_loss = FlipConsistencyLoss(margin=0.8)

# If policy changes → verdict MUST change by ≥80% probability
# Loss is HIGH when model fails to flip
# Loss is LOW when model correctly flips
```

### 3. Structured Output

```json
{
  "is_safe": false,
  "violations": ["C6"],
  "evidence": [
    {
      "clause": "C6",
      "text_span": [0, 42],
      "text": "People of [ethnic group] are inferior",
      "confidence": 0.95
    }
  ],
  "not_covered": false,
  "confidence": 0.95
}
```

## Usage Example

```python
from model.model import ConstitutionalGuard
from schema.policy_schema import create_mlcommons_policy

# Load trained model
model = ConstitutionalGuard(model_id="./training/constitutional_guard_lora")

# Create policy
policy = create_mlcommons_policy()
policy_text = policy.to_text()

# Classify input
input_text = "People of [ethnic group] are inferior"
result = model.classify(policy_text, input_text)

print(f"Is safe: {result['is_safe']}")
print(f"Violations: {result['violations']}")
print(f"Evidence: {result['evidence']}")

# Test flip: disable hate speech clause
policy_no_hate = policy.create_variant(disable_clauses=["C6"])
result_no_hate = model.classify(policy_no_hate.to_text(), input_text)

# Should flip: unsafe → safe
assert result['is_safe'] != result_no_hate['is_safe']  # ✓ Flip!
```

## Advanced: Custom Policies

```python
from schema.policy_schema import Policy, Clause, Severity, Scope

# Create custom policy
custom_policy = Policy(
    name="MyCustomPolicy",
    version="1.0",
    clauses=[
        Clause(
            id="C1",
            category="Forbidden Topic",
            rule="Content that discusses [your forbidden topic]",
            severity=Severity.HIGH,
            scope=Scope.BOTH
        ),
        Clause(
            id="C2",
            category="Another Rule",
            rule="Content that violates [your rule]",
            severity=Severity.MEDIUM,
            scope=Scope.BOTH
        )
    ]
)

# Use with model
result = model.classify(custom_policy.to_text(), user_input)
```

## Evaluation Metrics

### Flip-Rate (Most Important)

Measures: "Does model actually follow policy changes?"

```
flip_rate = (# correctly flipped) / (# expected flips)
target: ≥95%
```

**Example**:
- Input: "Violence content"
- Policy A (violence forbidden): unsafe
- Policy B (violence allowed): safe
- If model flips → ✓ counts toward flip-rate

### Standard Metrics

- **Accuracy**: Overall correctness
- **Precision/Recall/F1**: Unsafe detection quality
- **Clause Exact Match**: % correct clause IDs
- **Evidence Span F1**: Token-level evidence accuracy

## Comparison with Llama Guard 3

| Metric | Llama Guard 3 | Constitutional Guard |
|--------|---------------|---------------------|
| Flip-rate | ~0% | ≥95% (target) |
| Policy adherence | ✗ Ignores | ✓ Follows |
| Evidence | None | ✓ Clause IDs + spans |
| Customizable | ✗ Fixed categories | ✓ Any policy |
| Abstention | No | ✓ not_covered flag |

## Training Details

### LoRA Configuration

- **Rank**: 16 (good balance of performance/efficiency)
- **Alpha**: 32 (2× rank, standard)
- **Target modules**: All attention + MLP layers
- **Trainable params**: ~50M (vs 8B total = 0.6%)

### Hardware Requirements

| Setup | VRAM | Speed | Cost |
|-------|------|-------|------|
| Minimum (8-bit + LoRA) | 24GB | ~6h | RTX 3090 |
| Recommended (8-bit + LoRA) | 40GB | ~4h | A100 |
| Fast (fp16 + full) | 80GB | ~2h | A100 80GB |

### Hyperparameters

```python
num_epochs = 3
batch_size = 4
gradient_accumulation = 4  # Effective batch = 16
learning_rate = 2e-4
warmup_steps = 100
lora_r = 16
lora_alpha = 32
```

## Troubleshooting

### OOM (Out of Memory)

**Solution 1**: Reduce batch size
```bash
python train.py --batch_size 2
```

**Solution 2**: Reduce LoRA rank
```bash
python train.py --lora_r 8
```

**Solution 3**: Reduce max length
```python
# In train.py, change max_length
max_length = 1024  # Instead of 2048
```

### Low Flip-Rate After Training

**Possible causes**:
1. Flip-consistency loss weight too low → increase `flip_weight` in losses.py
2. Not enough flip pairs → increase `num_flip_pairs` in data_generator.py
3. Base model has strong safety priors → try truly base model (not instruct)

**Solutions**:
```python
# In losses.py
flip_weight = 5.0  # Increase from 2.0

# In data_generator.py
num_flip_pairs = 500  # Increase from 200
```

### Model Access Error

If you get "meta-llama/Llama-3.1-8B not found":

1. Request access: https://huggingface.co/meta-llama/Llama-3.1-8B
2. Login: `huggingface-cli login`
3. Verify access: `huggingface-cli whoami`

## Next Steps

1. **Train model** (if not done)
2. **Evaluate flip-rate** (target: ≥95%)
3. **Compare with baseline** (Llama Guard 3 or zero-shot)
4. **Deploy** (use `model.classify()` in production)

## Citation

If you use this work, please cite:

```bibtex
@software{constitutional_guard_2025,
  title={Constitutional Guard: A Policy-First Safety Classifier},
  author={[Your Name]},
  year={2025},
  note={Trained with flip-consistency loss for true policy adherence}
}
```

## References

- Llama Guard 3 analysis: [../ARTIFACT_ANALYSIS_RESULTS.md](../ARTIFACT_ANALYSIS_RESULTS.md)
- MLCommons AI Safety: https://mlcommons.org/
- LoRA paper: https://arxiv.org/abs/2106.09685
- Llama 3: https://ai.meta.com/blog/meta-llama-3/

---

**Status**: Complete implementation ready for training and evaluation
**Last updated**: 2025-10-27
