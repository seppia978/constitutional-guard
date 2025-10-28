# Cache Configuration - COMPLETED ✅

## Summary

All HuggingFace models will now be downloaded to:

```
/ephemeral/shared/spoppi/huggingface/hub/
```

## What Was Done

### 1. Created Centralized Configuration

**File**: `constitutional_guard/config.py`

```python
HF_CACHE_DIR = "/ephemeral/shared/spoppi/huggingface/hub/"
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = HF_CACHE_DIR
```

### 2. Updated All Scripts

Modified to import cache directory from centralized config:

- ✅ `model/model.py` - ConstitutionalGuard wrapper
- ✅ `training/train.py` - Training pipeline
- ✅ Other scripts import from config.py

### 3. Created Verification Script

**File**: `constitutional_guard/verify_setup.py`

Checks:
- ✅ Cache directory exists and is writable
- ✅ All data files are generated
- ✅ Dependencies are installed (torch, transformers, peft)
- ✅ GPU is available (4x NVIDIA RTX A6000, 47.5 GB each)

### 4. Verified Setup

**Result**:
```
✓ Configuration loaded
✓ Cache directory: /ephemeral/shared/spoppi/huggingface/hub/
✓ Cache directory is writable
✓ All data files present (2.7 MB total)
✓ All dependencies installed
✓ GPU available: NVIDIA RTX A6000 (47.5 GB)
```

## Project Status

### Constitutional Guard - Complete Implementation

**Phase 1: Llama Guard 3 Analysis** ✅
- Discovered Llama Guard 3 doesn't follow policy semantically
- Artifact control tests showed 77% attention is positional, not semantic
- Behavioral tests confirmed safety alignment dominates
- Flip-rate ~0% (model can't be constrained by policy)

**Phase 2: Constitutional Guard** ✅
- Complete policy-first architecture designed
- 61 diverse policies generated
- 2731 training examples with flip-pairs
- Flip-consistency loss implemented
- Training pipeline with LoRA ready
- Evaluation suite ready

### Files Created

```
constitutional_guard/
├── config.py                    ✅ NEW - Centralized config
├── verify_setup.py              ✅ NEW - Setup verification
├── SETUP.md                     ✅ NEW - Setup instructions
├── QUICKSTART.md                ✅ Complete
├── PROJECT_STATUS.md            ✅ Complete
├── schema/                      ✅ Complete (2 files)
├── data/                        ✅ Complete (6 files)
├── model/                       ✅ Complete (2 files, updated)
├── training/                    ✅ Complete (1 file, updated)
└── evaluation/                  ✅ Complete (2 files)
```

## Next Steps for Training

### 1. HuggingFace Login

```bash
huggingface-cli login
```

### 2. Verify Setup

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard
conda run -n const-ai python verify_setup.py
```

### 3. Train Model

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard/training
conda run -n const-ai python train.py
```

**Expected**:
- Model downloads to: `/ephemeral/shared/spoppi/huggingface/hub/`
- Training time: 4-8 hours on RTX A6000
- Output: LoRA checkpoint in `training/constitutional_guard_lora/`

### 4. Evaluate

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard/evaluation
conda run -n const-ai python evaluate.py
```

**Target**: Flip-rate ≥95%

## What Models Will Be Downloaded

When you run training, these will be downloaded to cache directory:

1. **meta-llama/Llama-3.1-8B** (~16 GB)
   - Base model for Constitutional Guard
   - Requires HuggingFace access (request at https://huggingface.co/meta-llama/Llama-3.1-8B)

2. **Tokenizer** (~5 MB)
   - Llama 3 tokenizer

3. **PEFT Adapters** (generated during training, ~200 MB)
   - LoRA weights

**Total space needed**: ~17 GB

## Cache Directory Structure

After training, the cache will contain:

```
/ephemeral/shared/spoppi/huggingface/hub/
├── models--meta-llama--Llama-3.1-8B/
│   ├── snapshots/
│   │   └── [hash]/
│   │       ├── model-00001-of-00004.safetensors
│   │       ├── model-00002-of-00004.safetensors
│   │       ├── model-00003-of-00004.safetensors
│   │       ├── model-00004-of-00004.safetensors
│   │       ├── config.json
│   │       ├── tokenizer.json
│   │       └── ...
│   └── ...
└── ...
```

## Verification Commands

### Check cache directory

```bash
ls -lh /ephemeral/shared/spoppi/huggingface/hub/
```

### Check disk space

```bash
df -h /ephemeral/shared/spoppi/
```

### Verify configuration is loaded

```bash
cd /ephemeral/home/spoppi/projects/random_tests/constitutional_guard
conda run -n const-ai python -c "from config import HF_CACHE_DIR; print(f'Cache: {HF_CACHE_DIR}')"
```

Expected output:
```
✓ Configuration loaded
  HF_CACHE_DIR: /ephemeral/shared/spoppi/huggingface/hub/
  PROJECT_ROOT: /ephemeral/home/spoppi/projects/random_tests/constitutional_guard
Cache: /ephemeral/shared/spoppi/huggingface/hub/
```

## Summary of Changes

| File | Change | Status |
|------|--------|--------|
| config.py | Created centralized config | ✅ NEW |
| model/model.py | Import from config | ✅ UPDATED |
| training/train.py | Import from config | ✅ UPDATED |
| verify_setup.py | Setup verification | ✅ NEW |
| SETUP.md | Setup documentation | ✅ NEW |

## All Systems Ready ✅

- ✅ Cache directory configured: `/ephemeral/shared/spoppi/huggingface/hub/`
- ✅ Cache directory exists and is writable
- ✅ All scripts updated to use centralized config
- ✅ Setup verification script created and tested
- ✅ Documentation complete
- ✅ Dependencies installed
- ✅ GPU available (4x RTX A6000)
- ✅ Data files generated (2731 training examples)

**Ready for training!** 🚀

---

**Date**: 2025-10-27
**Cache Location**: `/ephemeral/shared/spoppi/huggingface/hub/` ✅
**Status**: Configuration complete, ready for model download and training
