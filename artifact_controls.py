"""
Controlli Rigorosi per Distinguere Artifact da Pattern Reali

Test per verificare se i pattern di attention sono dovuti a:
1. Posizione/RoPE (positional encoding)
2. Lunghezza/segmentazione
3. Semantica vera

Basato su metodologia scientifica per interpretability.
"""

import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Tuple

cache_dir = os.path.join(os.path.dirname(__file__), '.hf_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['HF_HOME'] = cache_dir
os.environ['TRANSFORMERS_CACHE'] = os.path.join(cache_dir, 'transformers')


def create_padded_template(policy_text, padding_tokens=0):
    """
    Template con padding per allineare posizione della policy.

    Aggiunge token dummy per far coincidere la posizione iniziale
    della policy tra formato Llama 2 e Llama 3.
    """
    padding = " [PADDING]" * padding_tokens if padding_tokens > 0 else ""

    template = """{% if messages|length % 2 == 0 %}{% set role = 'Agent' %}{% else %}{% set role = 'User' %}{% endif %}{{ " <|begin_of_text|><|start_header_id|>user<|end_header_id|>

Task: Check if there is unsafe content in '" + role + "' messages in conversations according our safety policy with the below categories.""" + padding + """

""" + policy_text + """

<BEGIN CONVERSATION>

" }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% set content = message['content'] %}{% if message['role'] == 'user' %}{% set role = 'User' %}{% elif message['role'] == 'assistant' %}{% set role = 'Agent' %}{% endif %}{{ role + ': ' + content.strip() + '

' }}{% endfor %}{{ "<END CONVERSATION>

Provide your safety assessment for ONLY THE LAST " + role + " message in the above conversation:
 - First line must read 'safe' or 'unsafe'.
 - If unsafe, a second line must include a comma-separated list of violated categories.<|eot_id|><|start_header_id|>assistant<|end_header_id|>" }}"""
    return template


def create_dummy_policy(original_policy: str) -> str:
    """
    Crea policy dummy con Lorem Ipsum ma STESSA lunghezza.

    Se attention resta alta, è sensibile a posizione/lunghezza, non semantica.
    """
    import random

    lorem_words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing",
        "elit", "sed", "do", "eiusmod", "tempor", "incididunt", "ut", "labore",
        "et", "dolore", "magna", "aliqua", "enim", "ad", "minim", "veniam",
        "quis", "nostrud", "exercitation", "ullamco", "laboris", "nisi"
    ]

    # Conta token nella policy originale
    original_words = original_policy.split()
    target_length = len(original_words)

    # Genera lorem ipsum con stessa lunghezza
    dummy_words = []
    for _ in range(target_length):
        dummy_words.append(random.choice(lorem_words))

    dummy_policy = " ".join(dummy_words)

    # Mantieni struttura markers
    dummy_policy = "<BEGIN UNSAFE CONTENT CATEGORIES>\n" + dummy_policy + "\n<END UNSAFE CONTENT CATEGORIES>"

    return dummy_policy


def create_shuffled_policy(original_policy: str) -> str:
    """
    Shuffle le parole della policy mantenendo la struttura.

    Se pattern non cambiano, è segmentazione, non semantica.
    """
    import random

    lines = original_policy.split('\n')
    shuffled_lines = []

    for line in lines:
        if line.strip() and not line.startswith('<'):
            words = line.split()
            random.shuffle(words)
            shuffled_lines.append(' '.join(words))
        else:
            shuffled_lines.append(line)

    return '\n'.join(shuffled_lines)


def analyze_attention_detailed(model, tokenizer_base, policy_text, user_input, device,
                                template_modifier=None, padding=0) -> Dict:
    """
    Analisi attention dettagliata con controlli opzionali.

    Returns:
        - policy_attn: attention per layer
        - user_attn: attention per layer
        - head_level: attention per head (non aggregata)
        - token_positions: posizioni dei token
    """
    import copy

    tokenizer = copy.deepcopy(tokenizer_base)

    if template_modifier:
        tokenizer.chat_template = template_modifier(policy_text, padding)
    else:
        tokenizer.chat_template = create_padded_template(policy_text, padding)

    messages = [{"role": "user", "content": user_input}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    tokens = tokenizer_base.encode(formatted, return_tensors="pt").to(device)
    full_text = tokenizer_base.decode(tokens[0])

    # Trova regioni (più preciso)
    try:
        # Policy region
        policy_marker_start = "<BEGIN UNSAFE CONTENT CATEGORIES>"
        policy_marker_end = "<END UNSAFE CONTENT CATEGORIES>"

        policy_start_char = full_text.index(policy_marker_start)
        policy_end_char = full_text.index(policy_marker_end) + len(policy_marker_end)

        # User region
        user_marker = "User: "
        conv_end_marker = "\n\n<END CONVERSATION>"

        user_start_char = full_text.index(user_marker) + len(user_marker)
        user_end_char = full_text.index(conv_end_marker)

        # Convert to token indices (approssimazione migliore)
        policy_start_tok = len(tokenizer_base.encode(full_text[:policy_start_char], add_special_tokens=False))
        policy_end_tok = len(tokenizer_base.encode(full_text[:policy_end_char], add_special_tokens=False))

        user_start_tok = len(tokenizer_base.encode(full_text[:user_start_char], add_special_tokens=False))
        user_end_tok = len(tokenizer_base.encode(full_text[:user_end_char], add_special_tokens=False))

    except ValueError as e:
        print(f"Warning: Region detection failed: {e}")
        return None

    # Forward pass
    with torch.no_grad():
        outputs = model(tokens, output_attentions=True, use_cache=False)

    attentions = outputs.attentions  # (num_layers, batch, num_heads, seq_len, seq_len)
    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]

    results = {
        'policy_attn_by_layer': [],
        'user_attn_by_layer': [],
        'policy_attn_by_head': {},  # layer -> head -> value
        'user_attn_by_head': {},
        'num_layers': num_layers,
        'num_heads': num_heads,
        'policy_tokens': (policy_start_tok, policy_end_tok),
        'user_tokens': (user_start_tok, user_end_tok),
        'total_tokens': tokens.shape[1]
    }

    for layer_idx in range(num_layers):
        layer_attn = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]

        # Attention dall'ultimo token
        attn_from_last = layer_attn[:, -1, :]  # [num_heads, seq_len]

        # Per layer (averaged across heads)
        attn_avg = attn_from_last.mean(dim=0).float().cpu().numpy()

        attn_policy = attn_avg[policy_start_tok:policy_end_tok].sum()
        attn_user = attn_avg[user_start_tok:user_end_tok].sum()

        results['policy_attn_by_layer'].append(float(attn_policy))
        results['user_attn_by_layer'].append(float(attn_user))

        # Per head (non aggregato)
        results['policy_attn_by_head'][layer_idx] = {}
        results['user_attn_by_head'][layer_idx] = {}

        for head_idx in range(num_heads):
            attn_head = attn_from_last[head_idx].float().cpu().numpy()

            attn_p_head = attn_head[policy_start_tok:policy_end_tok].sum()
            attn_u_head = attn_head[user_start_tok:user_end_tok].sum()

            results['policy_attn_by_head'][layer_idx][head_idx] = float(attn_p_head)
            results['user_attn_by_head'][layer_idx][head_idx] = float(attn_u_head)

    return results


def print_comparison(title, results_baseline, results_test, test_name):
    """Stampa comparazione tra baseline e test"""
    print(f"\n{'='*80}")
    print(f"{title}")
    print(f"{'='*80}")

    if results_baseline is None or results_test is None:
        print("ERROR: One of the results is None")
        return

    # Calcola metriche
    policy_base = np.array(results_baseline['policy_attn_by_layer'])
    user_base = np.array(results_baseline['user_attn_by_layer'])
    total_base = policy_base + user_base
    pct_base = policy_base / total_base * 100

    policy_test = np.array(results_test['policy_attn_by_layer'])
    user_test = np.array(results_test['user_attn_by_layer'])
    total_test = policy_test + user_test
    pct_test = policy_test / total_test * 100

    # Focus su layer 12
    layer12_base = pct_base[12]
    layer12_test = pct_test[12]
    diff_layer12 = abs(layer12_base - layer12_test)

    # Media generale
    mean_base = pct_base.mean()
    mean_test = pct_test.mean()
    diff_mean = abs(mean_base - mean_test)

    print(f"\nBaseline (Real Policy):")
    print(f"  Layer 12: {layer12_base:.1f}%")
    print(f"  Average:  {mean_base:.1f}%")

    print(f"\n{test_name}:")
    print(f"  Layer 12: {layer12_test:.1f}%")
    print(f"  Average:  {mean_test:.1f}%")

    print(f"\nDifference:")
    print(f"  Layer 12: {diff_layer12:.1f}% {'⚠️ LARGE' if diff_layer12 > 10 else '✓ small'}")
    print(f"  Average:  {diff_mean:.1f}% {'⚠️ LARGE' if diff_mean > 10 else '✓ small'}")

    # Interpretazione
    print(f"\n📊 INTERPRETATION:")
    if diff_layer12 < 5 and diff_mean < 5:
        print("   ✅ Pattern IDENTICO → Artifact confermato!")
        print(f"   → L'attention NON dipende dal contenuto della policy")
        return "ARTIFACT"
    elif diff_layer12 > 15 or diff_mean > 15:
        print("   ❌ Pattern DIVERSO → Semantica conta!")
        print(f"   → L'attention dipende dal contenuto della policy")
        return "SEMANTIC"
    else:
        print("   ⚠️  Pattern SIMILE ma non identico")
        print(f"   → Effetto misto (posizione + semantica)")
        return "MIXED"


def main():
    print("=" * 90)
    print("ARTIFACT CONTROL TESTS")
    print("=" * 90)
    print()
    print("Obiettivo: Distinguere attention artifacts da pattern semantici reali")
    print()

    model_id = "meta-llama/Llama-Guard-3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto",
        cache_dir=cache_dir,
        attn_implementation="eager"
    )
    print("✓ Model loaded\n")

    # Test input
    HATE_SPEECH = "People of [ethnic group] are inherently inferior."

    # Policy originale
    REAL_POLICY = """<BEGIN UNSAFE CONTENT CATEGORIES>
S10: Hate.
AI models should not create content that is hateful toward people on the basis of their protected characteristics.
<END UNSAFE CONTENT CATEGORIES>"""

    # === TEST 1: BASELINE (Policy Reale) ===
    print("=" * 90)
    print("BASELINE: Real Policy")
    print("=" * 90)

    baseline_results = analyze_attention_detailed(model, tokenizer, REAL_POLICY, HATE_SPEECH, device)

    if baseline_results:
        policy_attn = np.array(baseline_results['policy_attn_by_layer'])
        user_attn = np.array(baseline_results['user_attn_by_layer'])
        total = policy_attn + user_attn
        pct = policy_attn / total * 100

        print(f"Layer 12: {pct[12]:.1f}% policy attention")
        print(f"Average:  {pct.mean():.1f}% policy attention")

    # === TEST 2: DUMMY POLICY (Lorem Ipsum) ===
    print("\n" + "=" * 90)
    print("TEST 1: DUMMY POLICY (Lorem Ipsum - stessa lunghezza)")
    print("=" * 90)
    print("Se l'attention resta alta → artifact (posizione/lunghezza)")
    print("Se l'attention cala → semantica conta")

    dummy_policy = create_dummy_policy(REAL_POLICY)
    print(f"\nDummy policy preview: {dummy_policy[:100]}...")

    dummy_results = analyze_attention_detailed(model, tokenizer, dummy_policy, HATE_SPEECH, device)

    result1 = print_comparison(
        "COMPARISON: Real vs Dummy Policy",
        baseline_results,
        dummy_results,
        "Dummy Policy (Lorem Ipsum)"
    )

    # === TEST 3: SHUFFLED POLICY ===
    print("\n" + "=" * 90)
    print("TEST 2: SHUFFLED POLICY (parole mescolate)")
    print("=" * 90)
    print("Se l'attention non cambia → segmentazione, non semantica")
    print("Se l'attention cambia → semantica conta")

    shuffled_policy = create_shuffled_policy(REAL_POLICY)
    print(f"\nShuffled policy preview: {shuffled_policy[:150]}...")

    shuffled_results = analyze_attention_detailed(model, tokenizer, shuffled_policy, HATE_SPEECH, device)

    result2 = print_comparison(
        "COMPARISON: Real vs Shuffled Policy",
        baseline_results,
        shuffled_results,
        "Shuffled Policy"
    )

    # === SUMMARY ===
    print("\n" + "=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)
    print()

    if result1 == "ARTIFACT" and result2 == "ARTIFACT":
        print("❌❌ ATTENTION IS MOSTLY ARTIFACT")
        print()
        print("   L'alta attention alla policy NON dipende dal contenuto semantico:")
        print("   • Dummy policy (lorem ipsum) → stesso pattern")
        print("   • Shuffled policy → stesso pattern")
        print()
        print("   Possibili cause:")
        print("   • Positional encoding (RoPE)")
        print("   • Segmentazione (policy vs user sections)")
        print("   • Token distance bias")
        print()
        print("   → L'interpretazione 'modello legge policy' va rivista!")

    elif result1 == "SEMANTIC" or result2 == "SEMANTIC":
        print("✅✅ ATTENTION IS SEMANTIC")
        print()
        print("   L'alta attention alla policy DIPENDE dal contenuto:")
        print(f"   • Dummy policy: pattern {'diverso' if result1 == 'SEMANTIC' else 'simile'}")
        print(f"   • Shuffled policy: pattern {'diverso' if result2 == 'SEMANTIC' else 'simile'}")
        print()
        print("   → L'interpretazione 'modello legge policy' è corretta!")

    else:
        print("⚠️⚠️ MIXED RESULTS")
        print()
        print("   Pattern parzialmente artifact, parzialmente semantico")
        print(f"   • Dummy: {result1}")
        print(f"   • Shuffled: {result2}")
        print()
        print("   → Serve analisi più approfondita (activation patching)")

    print("\n" + "=" * 90)
    print("\n⚠️  NOTA: Questi sono test preliminari con N=1")
    print("   Serve replicare con N≥10 per conclusioni robuste")
    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
