"""
Verify Constitutional Guard Setup

Checks:
1. Cache directory exists and is writable
2. All data files are generated
3. Config is loaded correctly
4. Dependencies are installed
"""

import os
import sys

print("=" * 80)
print("CONSTITUTIONAL GUARD SETUP VERIFICATION")
print("=" * 80)

# Check 1: Config
print("\n1. Checking configuration...")
try:
    from config import HF_CACHE_DIR, PROJECT_ROOT, DATA_DIR
    print(f"   ✓ Config loaded")
    print(f"     HF_CACHE_DIR: {HF_CACHE_DIR}")
    print(f"     PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"     DATA_DIR: {DATA_DIR}")
except Exception as e:
    print(f"   ✗ Config loading failed: {e}")
    sys.exit(1)

# Check 2: Cache directory
print("\n2. Checking cache directory...")
if os.path.exists(HF_CACHE_DIR):
    print(f"   ✓ Cache directory exists: {HF_CACHE_DIR}")

    # Test write permissions
    test_file = os.path.join(HF_CACHE_DIR, '.test_write')
    try:
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        print(f"   ✓ Cache directory is writable")
    except Exception as e:
        print(f"   ✗ Cache directory is not writable: {e}")
        sys.exit(1)
else:
    print(f"   ✗ Cache directory does not exist: {HF_CACHE_DIR}")
    print(f"     Creating directory...")
    try:
        os.makedirs(HF_CACHE_DIR, exist_ok=True)
        print(f"   ✓ Cache directory created")
    except Exception as e:
        print(f"   ✗ Failed to create cache directory: {e}")
        sys.exit(1)

# Check 3: Data files
print("\n3. Checking data files...")
data_files = [
    'training_dataset.json',
    'flip_test_dataset.json',
    'policy_zoo.json'
]

for fname in data_files:
    fpath = os.path.join(DATA_DIR, fname)
    if os.path.exists(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        print(f"   ✓ {fname} ({size_kb:.1f} KB)")
    else:
        print(f"   ✗ {fname} - NOT FOUND")
        print(f"     Run: cd data && python {fname.replace('_dataset.json', '_set.py').replace('_zoo.json', '_zoo.py')}")

# Check 4: Dependencies
print("\n4. Checking dependencies...")
dependencies = [
    ('torch', 'PyTorch'),
    ('transformers', 'HuggingFace Transformers'),
    ('peft', 'PEFT (LoRA)'),
]

all_deps_ok = True
for module, name in dependencies:
    try:
        __import__(module)
        print(f"   ✓ {name}")
    except ImportError:
        print(f"   ✗ {name} - NOT INSTALLED")
        print(f"     Install: pip install {module}")
        all_deps_ok = False

# Check 5: GPU availability
print("\n5. Checking GPU availability...")
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   ✓ GPU available: {gpu_name}")
        print(f"     Count: {gpu_count}")
        print(f"     Memory: {gpu_mem:.1f} GB")

        if gpu_mem < 24:
            print(f"   ⚠️  WARNING: GPU has <24GB memory")
            print(f"     Training may fail or be very slow")
            print(f"     Recommendation: Use GPU with ≥24GB VRAM")
    else:
        print(f"   ⚠️  No GPU available (CPU mode)")
        print(f"     Training will be VERY slow on CPU")
        print(f"     Recommendation: Use GPU instance")
except Exception as e:
    print(f"   ✗ Failed to check GPU: {e}")

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

if all_deps_ok:
    print("\n✓ All checks passed! Ready to train Constitutional Guard.")
    print("\nNext steps:")
    print("  1. Ensure you have access to meta-llama/Llama-3.1-8B")
    print("     Request: https://huggingface.co/meta-llama/Llama-3.1-8B")
    print("  2. Login: huggingface-cli login")
    print("  3. Train: cd training && python train.py")
    print("  4. Evaluate: cd evaluation && python evaluate.py")
else:
    print("\n✗ Some checks failed. Please install missing dependencies.")
    print("  pip install torch transformers peft")

print()
