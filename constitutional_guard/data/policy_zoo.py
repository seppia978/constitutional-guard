"""
Policy Zoo - Generate diverse policy variants for training

Creates:
- Different clause combinations
- Different severity thresholds
- Weird/counter-intuitive policies
- Minimal policies (1-3 clauses)
- Maximal policies (all clauses)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from schema.policy_schema import Policy, Clause, Severity, Scope, create_mlcommons_policy
from typing import List
import random
import itertools


def generate_random_subset_policies(base_policy: Policy, num_variants: int = 20) -> List[Policy]:
    """
    Generate policies with random subsets of clauses enabled.

    Creates diversity in which clauses are active.
    """
    all_clauses = base_policy.clauses
    policies = []

    for i in range(num_variants):
        # Random subset size (1 to all clauses)
        subset_size = random.randint(1, len(all_clauses))
        enabled_clauses = random.sample(all_clauses, subset_size)
        enabled_ids = [c.id for c in enabled_clauses]

        # Disable all, then enable subset
        disabled_ids = [c.id for c in all_clauses if c.id not in enabled_ids]
        variant = base_policy.create_variant(disable_clauses=disabled_ids)
        variant.name = f"RandomSubset-{i+1}"
        variant.version = f"1.0-subset{i+1}"
        variant.metadata["enabled_count"] = len(enabled_ids)
        variant.metadata["enabled_clauses"] = enabled_ids

        policies.append(variant)

    return policies


def generate_minimal_policies(base_policy: Policy) -> List[Policy]:
    """
    Generate minimal policies with only 1-3 clauses.

    Forces model to be precise about which clauses matter.
    """
    policies = []
    all_clauses = base_policy.clauses

    # Single-clause policies
    for clause in all_clauses:
        disabled = [c.id for c in all_clauses if c.id != clause.id]
        variant = base_policy.create_variant(disable_clauses=disabled)
        variant.name = f"Single-{clause.id}"
        variant.version = f"1.0-{clause.id}"
        variant.metadata["type"] = "single_clause"
        policies.append(variant)

    # Two-clause policies (sample, not all combinations)
    for _ in range(10):
        pair = random.sample(all_clauses, 2)
        enabled_ids = [c.id for c in pair]
        disabled = [c.id for c in all_clauses if c.id not in enabled_ids]
        variant = base_policy.create_variant(disable_clauses=disabled)
        variant.name = f"Pair-{enabled_ids[0]}-{enabled_ids[1]}"
        variant.version = f"1.0-pair"
        variant.metadata["type"] = "two_clause"
        policies.append(variant)

    # Three-clause policies
    for _ in range(10):
        triple = random.sample(all_clauses, 3)
        enabled_ids = [c.id for c in triple]
        disabled = [c.id for c in all_clauses if c.id not in enabled_ids]
        variant = base_policy.create_variant(disable_clauses=disabled)
        variant.name = f"Triple-{'-'.join(enabled_ids)}"
        variant.version = f"1.0-triple"
        variant.metadata["type"] = "three_clause"
        policies.append(variant)

    return policies


def generate_severity_filtered_policies(base_policy: Policy) -> List[Policy]:
    """
    Generate policies filtered by severity level.

    E.g., "only CRITICAL violations", "only HIGH/CRITICAL", etc.
    """
    policies = []

    severity_filters = [
        ([Severity.CRITICAL], "CriticalOnly"),
        ([Severity.CRITICAL, Severity.HIGH], "HighAndCritical"),
        ([Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL], "MediumAndAbove"),
    ]

    for severities, name in severity_filters:
        enabled_clauses = [c for c in base_policy.clauses if c.severity in severities]
        enabled_ids = [c.id for c in enabled_clauses]
        disabled_ids = [c.id for c in base_policy.clauses if c.id not in enabled_ids]

        variant = base_policy.create_variant(disable_clauses=disabled_ids)
        variant.name = name
        variant.version = f"1.0-severity"
        variant.metadata["severity_filter"] = [s.value for s in severities]

        policies.append(variant)

    return policies


def generate_scope_filtered_policies(base_policy: Policy) -> List[Policy]:
    """
    Generate policies filtered by scope (user_input, assistant_output, both).
    """
    policies = []

    scope_filters = [
        (Scope.USER_INPUT, "UserInputOnly"),
        (Scope.ASSISTANT_OUTPUT, "AssistantOutputOnly"),
        (Scope.BOTH, "BothScopes"),
    ]

    for scope, name in scope_filters:
        enabled_clauses = [c for c in base_policy.clauses if c.scope == scope or c.scope == Scope.BOTH]
        enabled_ids = [c.id for c in enabled_clauses]
        disabled_ids = [c.id for c in base_policy.clauses if c.id not in enabled_ids]

        variant = base_policy.create_variant(disable_clauses=disabled_ids)
        variant.name = name
        variant.version = f"1.0-scope"
        variant.metadata["scope_filter"] = scope.value

        policies.append(variant)

    return policies


def generate_weird_policies() -> List[Policy]:
    """
    Generate counter-intuitive policies to test if model truly follows policy.

    Examples:
    - Violence allowed, hate speech forbidden
    - Sexual content forbidden, violence allowed
    - Only forbid politeness (inverted safety)
    """
    policies = []

    # Weird 1: Only hate speech forbidden (violence OK)
    policy1 = Policy(
        name="OnlyHateForbidden",
        version="1.0-weird",
        clauses=[
            Clause(
                id="C1",
                category="Hate Speech",
                rule="Content that is hateful toward people on the basis of their protected characteristics.",
                severity=Severity.HIGH,
                scope=Scope.BOTH
            )
        ],
        metadata={"type": "weird", "note": "Violence allowed, only hate forbidden"}
    )
    policies.append(policy1)

    # Weird 2: Inverted - forbid politeness
    policy2 = Policy(
        name="ForbidPoliteness",
        version="1.0-weird",
        clauses=[
            Clause(
                id="C1",
                category="Excessive Politeness",
                rule="Content that is excessively polite or uses formal language.",
                severity=Severity.MEDIUM,
                scope=Scope.BOTH
            ),
            Clause(
                id="C2",
                category="Gratitude",
                rule="Content that expresses thanks or gratitude.",
                severity=Severity.LOW,
                scope=Scope.BOTH
            )
        ],
        metadata={"type": "weird", "note": "Counter-intuitive policy"}
    )
    policies.append(policy2)

    # Weird 3: Only forbid specific phrases
    policy3 = Policy(
        name="ForbidHelloGoodbye",
        version="1.0-weird",
        clauses=[
            Clause(
                id="C1",
                category="Greetings",
                rule="Content that contains the words 'hello' or 'goodbye'.",
                severity=Severity.HIGH,
                scope=Scope.BOTH
            )
        ],
        metadata={"type": "weird", "note": "Only forbid hello/goodbye"}
    )
    policies.append(policy3)

    # Weird 4: Fictional categories
    policy4 = Policy(
        name="FictionalCategories",
        version="1.0-weird",
        clauses=[
            Clause(
                id="C1",
                category="Unicorn References",
                rule="Content that mentions unicorns or mythical creatures.",
                severity=Severity.HIGH,
                scope=Scope.BOTH
            ),
            Clause(
                id="C2",
                category="Time Travel",
                rule="Content that discusses time travel or temporal paradoxes.",
                severity=Severity.MEDIUM,
                scope=Scope.BOTH
            )
        ],
        metadata={"type": "weird", "note": "Fictional unsafe categories"}
    )
    policies.append(policy4)

    return policies


def generate_complete_policy_zoo() -> List[Policy]:
    """
    Generate complete policy zoo with all variants.

    Returns 100+ diverse policies for training.
    """
    base_policy = create_mlcommons_policy()
    all_policies = []

    # Base policy (all clauses)
    all_policies.append(base_policy)

    # Random subsets (20 variants)
    all_policies.extend(generate_random_subset_policies(base_policy, num_variants=20))

    # Minimal policies (single, pairs, triples)
    all_policies.extend(generate_minimal_policies(base_policy))

    # Severity-filtered
    all_policies.extend(generate_severity_filtered_policies(base_policy))

    # Scope-filtered
    all_policies.extend(generate_scope_filtered_policies(base_policy))

    # Weird policies
    all_policies.extend(generate_weird_policies())

    print(f"✓ Generated {len(all_policies)} policies")
    return all_policies


def save_policy_zoo(policies: List[Policy], output_file: str = "policy_zoo.json"):
    """Save policy zoo to JSON"""
    import json

    data = {
        "name": "Policy Zoo",
        "version": "1.0",
        "description": "Diverse policy variants for training constitutional guard",
        "num_policies": len(all_policies),
        "policies": [p.to_dict() for p in policies]
    }

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Saved {len(policies)} policies to {output_file}")


if __name__ == "__main__":
    # Generate complete zoo
    all_policies = generate_complete_policy_zoo()

    # Print summary
    print("\n" + "=" * 80)
    print("POLICY ZOO SUMMARY")
    print("=" * 80)

    policy_types = {}
    for p in all_policies:
        ptype = p.metadata.get("type", "standard")
        policy_types[ptype] = policy_types.get(ptype, 0) + 1

    for ptype, count in sorted(policy_types.items()):
        print(f"{ptype:20s}: {count:3d} policies")

    print(f"\n{'Total':20s}: {len(all_policies):3d} policies")

    # Show examples
    print("\n" + "=" * 80)
    print("EXAMPLE POLICIES")
    print("=" * 80)

    # Show single-clause example
    single = [p for p in all_policies if p.metadata.get("type") == "single_clause"][0]
    print(f"\nExample: {single.name}")
    print(single.to_text()[:200] + "...")

    # Show weird example
    weird = [p for p in all_policies if p.metadata.get("type") == "weird"][0]
    print(f"\nExample: {weird.name}")
    print(weird.to_text())

    # Save
    save_policy_zoo(all_policies)
