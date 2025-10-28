# Constitutional Guard - Policy-First Safety Classifier

A true "constitution-first" safety classifier that actually follows the provided policy, not baked-in safety alignment.

## Goal

Create a classifier that:
- Takes **(policy_text, input)** → outputs structured JSON
- **Flips verdict** when policy changes (flip-rate ≥95%)
- Provides **evidence spans** and **clause citations**
- Can **abstain** when input not covered by policy

## Architecture

**LLM-based approach** with:
- Base model: **Llama 3.1 8B** (non-instruct to minimize safety priors)
- Input format: `[POLICY]` ⊕ `[INPUT]` → structured output
- Training objectives:
  1. Classification (safe/unsafe)
  2. Clause pointing (which clause violated)
  3. **Flip-consistency loss** (same input, different policy → different verdict)
  4. Abstain loss (not covered by policy)

## Output Format

```json
{
  "is_safe": true|false,
  "violations": ["C1", "C3"],
  "evidence": [
    {"clause": "C1", "text_span": [12, 45], "text": "extracted span"}
  ],
  "not_covered": false,
  "confidence": 0.95
}
```

## Policy Schema

Structured format with:
- **Clause IDs** (C1, C2, ..., Cn)
- **Atomic rules** (one concept per clause)
- **Severity levels**
- **Scope metadata**

## Training Data Requirements

- **Policy-Zoo**: 50-100 diverse policies
- **Flip-Sets**: Same input with clauses on/off
- **Examples per clause**: positive, negative, edge cases
- **Not-covered** examples
- **Adversarial** cases (injection, jailbreak)

## Validation Metrics

- **Flip-rate**: % verdicts that change when policy changes (target: ≥95%)
- **Clause F1**: Exact match on violated clause IDs
- **Span F1**: Token-level F1 on evidence spans
- **Policy-transfer**: Performance on held-out policies
- **Weird-policy**: Performance on counter-intuitive policies

## Directory Structure

```
constitutional_guard/
├── README.md                    # This file
├── schema/
│   ├── policy_schema.py        # Policy data structures
│   └── output_schema.py        # JSON output schema
├── data/
│   ├── policy_zoo.py           # Policy generation
│   ├── data_generator.py       # Training data generation
│   └── flip_test_set.py        # Flip-rate test dataset
├── model/
│   ├── model.py                # Model architecture
│   ├── losses.py               # Flip-consistency + multi-task losses
│   └── inference.py            # JSON-constrained decoding
├── training/
│   ├── train.py                # Training pipeline
│   └── config.yaml             # Training config
└── evaluation/
    ├── evaluate.py             # Metrics computation
    └── flip_rate_test.py       # Flip-rate validation
```

## Status

- [x] Project design
- [ ] Schema implementation
- [ ] Data generation
- [ ] Model architecture
- [ ] Training pipeline
- [ ] Evaluation suite
