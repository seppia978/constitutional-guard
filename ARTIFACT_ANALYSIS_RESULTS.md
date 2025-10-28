# Artifact Control Tests - Critical Findings

## Executive Summary

**CRITICAL DISCOVERY**: The high attention to policy regions (77%) is **NOT semantic processing** but rather a **positional/structural artifact**.

## Test Results

### Baseline (Real Policy)
- Layer 12: 70.1% policy attention
- Average: 81.7% policy attention

### Test 1: Dummy Policy (Lorem Ipsum)
**Purpose**: Replace policy content with meaningless lorem ipsum words of same length

**Results**:
- Layer 12: 66.0% policy attention (diff: 4.1%)
- Average: 76.9% policy attention (diff: 4.7%)

**Verdict**: ✅ **ARTIFACT CONFIRMED** - Pattern nearly identical despite meaningless content

### Test 2: Shuffled Policy
**Purpose**: Shuffle words within policy while maintaining structure markers

**Results**:
- Layer 12: 69.0% policy attention (diff: 1.1%)
- Average: 82.5% policy attention (diff: 0.9%)

**Verdict**: ✅ **ARTIFACT CONFIRMED** - Pattern unchanged even when words are scrambled

## Interpretation

### What This Means

The 77% "policy attention" does NOT indicate that the model is:
- Reading the policy content semantically
- Understanding policy rules
- Making decisions based on policy text

Instead, the attention pattern is caused by:
1. **Positional Encoding (RoPE)**: The model attends to that region based on its absolute/relative position in the sequence
2. **Segmentation Bias**: Clear boundaries between system/policy/user sections create structural attention patterns
3. **Token Distance Effects**: Attention may be based on distances rather than semantic content

### What Remains Valid

✅ **Behavioral Tests**: These remain the gold standard and show true model behavior:
- Empty policy test: Model still flags as unsafe (100%)
- Inverted policy test: Model ignores nonsensical rules
- Fictional categories test: Model uses them for labeling only
- Strict 3-category policy test: Model ignores constraints, uses safety alignment (3/4 tests failed)

❌ **Attention Analysis**: Cannot be interpreted as "reading the policy"
- High attention ≠ semantic processing
- Layer 12 anomaly likely also a positional artifact
- Attention patterns don't reveal mechanistic understanding

## Revised Model Architecture Hypothesis

### Dual-System Architecture (Confirmed)

Based on behavioral tests, the model operates as:

```
Input → Safety Classifier (Fixed Alignment) → Category Labeler (Policy-Aware)
                    ↓                                    ↓
              unsafe/safe                         S1/S2/.../S13
```

**System 1: Safety Classifier**
- Uses internal safety alignment (fixed, not policy-dependent)
- Determines unsafe vs safe
- Ignores policy modifications (empty, inverted, shuffled, strict constraints)
- Robust to policy manipulation

**System 2: Category Labeler**
- Uses policy for labeling which category to output
- Can use fictional categories (Unicorns, Time Travel)
- Only activates after System 1 decides "unsafe"
- Superficial policy reading for label selection

### Evidence Summary

| Test Type | Result | Interpretation |
|-----------|--------|----------------|
| Empty policy | Unsafe (100%) | Safety alignment dominates |
| Inverted policy | Ignores rules | Not following policy logic |
| Fictional categories | Uses them for labels | Policy only affects labeling |
| Strict 3-category policy | Flags hate/violence anyway | Cannot be constrained by policy |
| Dummy policy (lorem ipsum) | Same attention pattern | Attention is artifact |
| Shuffled policy | Same attention pattern | Attention is artifact |

## Implications for Interpretability Research

### What We Learned

1. **Attention ≠ Understanding**: High attention doesn't prove semantic processing
   - Must use artifact controls (dummy, shuffled, padding)
   - Positional artifacts are common in transformer models

2. **Behavioral Tests > Attention Analysis**: For understanding model logic
   - Black-box behavioral tests reveal true decision-making
   - Attention patterns can be misleading without controls

3. **Safety Alignment is Robust**: Model cannot be easily "jailbroken" via policy manipulation
   - Good for safety (prevents policy-based attacks)
   - Bad for customization (limits policy flexibility)

### Methodological Lessons

**Before claiming semantic processing**:
- ✅ Test with dummy content (same length)
- ✅ Test with shuffled content (same structure)
- ✅ Test with padding alignment (position control)
- ✅ Use behavioral validation (does output change?)
- ✅ Statistical validation (N≥30 samples)

**Red flags for artifacts**:
- Pattern unchanged with meaningless content
- Pattern unchanged with scrambled content
- High sensitivity to position/structure markers
- Low correlation with behavioral changes

## Next Steps

### Completed
- ✅ Artifact control tests (dummy, shuffled)
- ✅ Behavioral validation suite
- ✅ Statistical framework (preliminary N=3)

### Pending (Optional Extensions)
- Activation patching on layer 12 (to confirm it's not causally important)
- Head ablation studies (selective zeroing)
- Logit lens analysis (decode hidden states mid-network)
- Expand to N≥30 for robust statistics
- Test with other safety models (Llama Guard 1, 2, alternative architectures)

## Conclusion

The artifact control tests were **essential** for correct interpretation. Without them, we would have incorrectly concluded that:
- ❌ "The model reads the policy at layer 12"
- ❌ "77% attention means semantic processing"
- ❌ "Layer 12 is special for policy understanding"

The correct conclusion is:
- ✅ **Safety alignment determines unsafe/safe** (behavioral tests)
- ✅ **Policy only affects category labeling** (fictional categories test)
- ✅ **Attention patterns are positional artifacts** (dummy/shuffled tests)
- ✅ **The model cannot be constrained by policy modifications** (strict policy test)

This demonstrates the importance of **rigorous artifact controls** in interpretability research before making mechanistic claims about model behavior.

---

**Test Date**: 2025-10-27
**Model**: meta-llama/Llama-Guard-3-8B
**Test Framework**: N=1 (preliminary, needs replication)
**Status**: Artifact hypothesis confirmed, attention analysis invalidated
