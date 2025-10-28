# Constitutional Guard - Setup & Configuration

## Cache Directory Configuration ✅

All models will be downloaded to:
```
/ephemeral/shared/spoppi/huggingface/hub/
```

This is configured centrally in `config.py` and used by all scripts.

## Setup Verification

Run the verification script to check everything is ready:

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard
conda run -n const-ai python verify_setup.py
```

**Expected output**:
```
✓ Configuration loaded
✓ Cache directory exists and is writable
✓ All data files present (training_dataset.json, flip_test_dataset.json, policy_zoo.json)
✓ All dependencies installed (torch, transformers, peft)
✓ GPU available: NVIDIA RTX A6000 (47.5 GB)
```

## Current Status

### ✅ Completed
- Cache directory configured: `/ephemeral/shared/spoppi/huggingface/hub/`
- Configuration centralized in `config.py`
- All data files generated:
  - `training_dataset.json` (2711 KB, 2731 examples)
  - `flip_test_dataset.json` (114 KB, 11 flip tests)
  - `policy_zoo.json` (259 KB, 61 policies)
- All dependencies installed
- GPU available: 4x NVIDIA RTX A6000 (47.5 GB each)

### ⏳ Pending
- HuggingFace authentication (for Llama 3.1 8B access)
- Model training
- Evaluation on flip-test dataset

## Quick Start Commands

### 1. HuggingFace Login (Required)

```bash
# Login to HuggingFace
huggingface-cli login

# Verify access to Llama 3.1 8B
# If you don't have access, request at:
# https://huggingface.co/meta-llama/Llama-3.1-8B
```

### 2. Train Model

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard/training

conda run -n const-ai python train.py \
  --model_id meta-llama/Llama-3.1-8B \
  --data_file ../data/training_dataset.json \
  --output_dir ./constitutional_guard_lora \
  --epochs 3 \
  --batch_size 4 \
  --lora_r 16
```

**Expected time**: 4-8 hours on RTX A6000

**Output**:
- Model checkpoints in `training/constitutional_guard_lora/`
- Training logs

### 3. Evaluate Model

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard/evaluation

conda run -n const-ai python evaluate.py \
  --model_path ../training/constitutional_guard_lora \
  --base_model meta-llama/Llama-3.1-8B \
  --output evaluation_results.json
```

**Expected results**:
- Flip-rate: ≥95% (target)
- Accuracy: ~90%+
- Clause exact match: ~85%+

### 4. Compare with Baseline (Optional)

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard/evaluation

# Run baseline test (zero-shot Llama 3.1 8B)
conda run -n const-ai python baseline_test.py

# Compare trained vs baseline
conda run -n const-ai python evaluate.py \
  --model_path ../training/constitutional_guard_lora \
  --baseline_results baseline_results.json
```

## Directory Structure

```
constitutional_guard/
├── config.py                       # ✅ Centralized configuration
├── verify_setup.py                 # ✅ Setup verification script
├── SETUP.md                        # ✅ This file
├── QUICKSTART.md                   # Quick start guide
├── PROJECT_STATUS.md               # Detailed project status
├── schema/
│   ├── policy_schema.py           # ✅ Policy data structures
│   └── output_schema.py           # ✅ JSON output schema
├── data/
│   ├── policy_zoo.py              # ✅ Policy generator
│   ├── data_generator.py          # ✅ Training data generator
│   ├── flip_test_set.py           # ✅ Flip-test dataset
│   ├── training_dataset.json      # ✅ Generated (2731 examples)
│   ├── flip_test_dataset.json     # ✅ Generated (11 flip tests)
│   └── policy_zoo.json            # ✅ Generated (61 policies)
├── model/
│   ├── model.py                   # ✅ ConstitutionalGuard wrapper
│   └── losses.py                  # ✅ Flip-consistency loss
├── training/
│   ├── train.py                   # ✅ Training pipeline
│   └── constitutional_guard_lora/ # ⏳ Model checkpoints (after training)
└── evaluation/
    ├── baseline_test.py           # ✅ Baseline evaluation
    └── evaluate.py                # ✅ Full evaluation suite
```

## Configuration Details

### Cache Directory

**Location**: `/ephemeral/shared/spoppi/huggingface/hub/`

**What gets cached**:
- Llama 3.1 8B base model (~16 GB)
- Tokenizers
- PEFT adapters

**Environment variables** (set automatically by `config.py`):
```python
HF_HOME = "/ephemeral/shared/spoppi/huggingface/hub/"
TRANSFORMERS_CACHE = "/ephemeral/shared/spoppi/huggingface/hub/"
HF_DATASETS_CACHE = "/ephemeral/shared/spoppi/huggingface/hub/"
```

### Training Configuration

Default hyperparameters (defined in `config.py`):

```python
DEFAULT_TRAINING_CONFIG = {
    'num_epochs': 3,
    'batch_size': 4,
    'gradient_accumulation_steps': 4,  # Effective batch = 16
    'learning_rate': 2e-4,
    'warmup_steps': 100,
    'max_length': 2048,
    'lora_r': 16,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
}
```

### LoRA Configuration

- **Rank**: 16 (good balance)
- **Alpha**: 32 (2× rank)
- **Target modules**: All attention + MLP layers
- **Trainable params**: ~50M (0.6% of 8B)
- **Memory**: ~18-20 GB with 8-bit quantization

## Troubleshooting

### Issue: "Model not found"

**Cause**: No access to Llama 3.1 8B

**Solution**:
1. Request access: https://huggingface.co/meta-llama/Llama-3.1-8B
2. Wait for approval (usually <1 hour)
3. Login: `huggingface-cli login`

### Issue: OOM (Out of Memory)

**Cause**: GPU memory insufficient

**Solutions**:
```bash
# Option 1: Reduce batch size
python train.py --batch_size 2

# Option 2: Reduce LoRA rank
python train.py --lora_r 8

# Option 3: Reduce max length
# Edit train.py, change max_length=2048 to max_length=1024
```

### Issue: "CUDA out of memory" during inference

**Solution**: Use batch size 1 for evaluation
```bash
# In evaluate.py, model.classify() is already single-example
# No changes needed
```

### Issue: Training is slow

**Expected speeds** (RTX A6000):
- ~30-60 seconds per epoch (2731 examples, batch_size=4, grad_accum=4)
- Total: ~3-6 hours for 3 epochs

If much slower:
1. Check GPU utilization: `nvidia-smi`
2. Verify gradient checkpointing is enabled
3. Check disk I/O (slow data loading)

## Next Steps After Setup

1. ✅ **Verify setup**: `python verify_setup.py`
2. ⏳ **Login to HuggingFace**: `huggingface-cli login`
3. ⏳ **Train model**: `cd training && python train.py`
4. ⏳ **Evaluate**: `cd evaluation && python evaluate.py`
5. ⏳ **Analyze results**: Check if flip-rate ≥95%

## Expected Results

### Training Metrics

- **Training loss**: Should decrease steadily
- **Eval loss**: Should be close to training loss (no overfitting)
- **Final training loss**: ~0.5-1.0 (depends on data)

### Evaluation Metrics

| Metric | Llama Guard 3 | Constitutional Guard (Target) |
|--------|---------------|------------------------------|
| Flip-rate | ~0% | ≥95% |
| Accuracy | ~85% | ~90% |
| Precision | ~90% | ~90% |
| Recall | ~80% | ~85% |
| F1 | ~85% | ~87% |
| Clause exact match | N/A | ~85% |

### Success Criteria

- ✅ Flip-rate ≥95% on flip-test dataset
- ✅ Accuracy ≥85% on standard tests
- ✅ Model changes verdict when policy changes (unlike Llama Guard 3)

## Files Modified for Cache Configuration

1. ✅ `config.py` - Created (centralized configuration)
2. ✅ `model/model.py` - Updated to import from config
3. ✅ `training/train.py` - Updated to import from config
4. ✅ `verify_setup.py` - Created (setup verification)

All models will now download to: `/ephemeral/shared/spoppi/huggingface/hub/`

---

**Last updated**: 2025-10-27
**Status**: Setup complete, ready for training
**Cache directory**: `/ephemeral/shared/spoppi/huggingface/hub/` ✅
