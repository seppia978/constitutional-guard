"""
Constitutional Guard Model

LLM-based classifier that conditions on policy text.
Uses Llama 3.1 8B base (non-instruct) to minimize safety priors.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Optional
import json
import re

# Import centralized config
from config import HF_CACHE_DIR as CACHE_DIR


class ConstitutionalGuard:
    """
    Policy-conditioned safety classifier.

    Input: (policy_text, user_input)
    Output: JSON with {is_safe, violations, evidence, not_covered, confidence}
    """

    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B",
                 device: str = "auto"):
        """
        Initialize model.

        Args:
            model_id: HuggingFace model ID (use base model, not instruct)
            device: Device placement ("auto", "cuda", "cpu")
        """
        self.model_id = model_id
        self.device = device

        print(f"Loading {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            padding_side="left"  # For batch inference
        )

        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map=device,
            cache_dir=CACHE_DIR
        )
        self.model.eval()
        print(f"✓ Model loaded on {self.model.device}")

    def create_prompt(self, policy_text: str, user_input: str) -> str:
        """
        Create prompt for classification.

        Format:
        [POLICY]
        <policy text>
        [/POLICY]

        [INPUT]
        <user input>
        [/INPUT]

        [TASK]
        Classify the input according to the policy.
        Output JSON: {"is_safe": bool, "violations": [...], ...}
        [/TASK]

        [OUTPUT]
        """
        prompt = f"""[POLICY]
{policy_text}
[/POLICY]

[INPUT]
{user_input}
[/INPUT]

[TASK]
Classify the input according to the policy provided above.
- If the input violates any clause in the policy, output {{"is_safe": false, "violations": ["C1", ...], "evidence": [...]}}
- If the input does not violate any clause, output {{"is_safe": true, "violations": [], "evidence": []}}
- Include "evidence" with text spans that support each violation
- Set "not_covered": true if input falls outside policy scope
- Provide "confidence" score (0.0-1.0)

Output valid JSON only, no additional text.
[/TASK]

[OUTPUT]
"""
        return prompt

    def extract_json_from_output(self, text: str) -> Optional[Dict]:
        """
        Extract JSON from model output.

        Handles cases where model outputs additional text.
        """
        # Try to find JSON block
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)

        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # Fallback: try parsing entire text
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            return None

    def classify(self, policy_text: str, user_input: str,
                 max_new_tokens: int = 512,
                 temperature: float = 0.1) -> Dict:
        """
        Classify user input according to policy.

        Args:
            policy_text: Policy in structured format (from Policy.to_text())
            user_input: User input to classify
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature (low for deterministic)

        Returns:
            Dict with classification result
        """
        prompt = self.create_prompt(policy_text, user_input)

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        generated = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Extract JSON
        result = self.extract_json_from_output(generated)

        if result is None:
            # Fallback: return error
            return {
                "is_safe": True,  # Conservative: assume safe if can't parse
                "violations": [],
                "evidence": [],
                "not_covered": True,
                "confidence": 0.0,
                "error": f"Failed to parse JSON from output: {generated[:200]}"
            }

        return result

    def batch_classify(self, policy_text: str, user_inputs: list,
                      max_new_tokens: int = 512,
                      temperature: float = 0.1) -> list:
        """
        Classify multiple inputs with same policy (batched for efficiency).

        Args:
            policy_text: Policy text
            user_inputs: List of user inputs
            max_new_tokens: Max tokens per generation
            temperature: Sampling temperature

        Returns:
            List of classification results
        """
        prompts = [self.create_prompt(policy_text, inp) for inp in user_inputs]

        # Tokenize batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(self.model.device)

        # Generate batch
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        # Decode all
        results = []
        for i, output in enumerate(outputs):
            generated = self.tokenizer.decode(
                output[inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            result = self.extract_json_from_output(generated)
            if result is None:
                result = {
                    "is_safe": True,
                    "violations": [],
                    "evidence": [],
                    "not_covered": True,
                    "confidence": 0.0,
                    "error": f"Failed to parse JSON"
                }
            results.append(result)

        return results


if __name__ == "__main__":
    # Test model loading and inference
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from schema.policy_schema import create_mlcommons_policy

    # NOTE: This will fail without the base model being available
    # For now, we'll just test the prompt creation logic

    policy = create_mlcommons_policy()
    policy_text = policy.to_text()

    # Mock classifier for testing prompt format
    print("=" * 80)
    print("EXAMPLE PROMPT FORMAT")
    print("=" * 80)

    user_input = "People of [ethnic group] are inferior."

    # Create mock instance just for prompt
    class MockGuard:
        def create_prompt(self, policy_text, user_input):
            return ConstitutionalGuard.create_prompt(None, policy_text, user_input)

    guard = MockGuard()
    prompt = guard.create_prompt(policy_text, user_input)
    print(prompt)
    print()
    print(f"Prompt length: {len(prompt)} chars")
