"""
Policy Schema - Structured format for safety policies

Each policy is a collection of clauses with:
- Unique IDs (C1, C2, ...)
- Atomic rules (one concept per clause)
- Severity levels
- Scope metadata
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import json


class Severity(Enum):
    """Severity levels for policy violations"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Scope(Enum):
    """Scope of policy clause"""
    USER_INPUT = "user_input"
    ASSISTANT_OUTPUT = "assistant_output"
    BOTH = "both"


@dataclass
class Clause:
    """Single atomic policy clause"""
    id: str  # e.g., "C1", "C2"
    category: str  # e.g., "Hate Speech", "Violence"
    rule: str  # Human-readable rule text
    severity: Severity
    scope: Scope
    enabled: bool = True
    examples: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "category": self.category,
            "rule": self.rule,
            "severity": self.severity.value,
            "scope": self.scope.value,
            "enabled": self.enabled,
            "examples": self.examples,
            "metadata": self.metadata
        }

    def to_text(self) -> str:
        """Format clause for model input"""
        return f"{self.id}: {self.category}. {self.rule}"


@dataclass
class Policy:
    """Complete safety policy"""
    name: str
    version: str
    clauses: List[Clause]
    metadata: Dict = field(default_factory=dict)

    def get_enabled_clauses(self) -> List[Clause]:
        """Get only enabled clauses"""
        return [c for c in self.clauses if c.enabled]

    def get_clause_by_id(self, clause_id: str) -> Optional[Clause]:
        """Retrieve clause by ID"""
        for clause in self.clauses:
            if clause.id == clause_id:
                return clause
        return None

    def to_text(self) -> str:
        """Format policy for model input"""
        enabled = self.get_enabled_clauses()
        if not enabled:
            return "<NO POLICY DEFINED>"

        lines = ["<BEGIN POLICY>"]
        for clause in enabled:
            lines.append(clause.to_text())
        lines.append("<END POLICY>")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "clauses": [c.to_dict() for c in self.clauses],
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "Policy":
        """Load policy from dictionary"""
        clauses = []
        for c_data in data["clauses"]:
            clause = Clause(
                id=c_data["id"],
                category=c_data["category"],
                rule=c_data["rule"],
                severity=Severity(c_data["severity"]),
                scope=Scope(c_data["scope"]),
                enabled=c_data.get("enabled", True),
                examples=c_data.get("examples", []),
                metadata=c_data.get("metadata", {})
            )
            clauses.append(clause)

        return cls(
            name=data["name"],
            version=data["version"],
            clauses=clauses,
            metadata=data.get("metadata", {})
        )

    def create_variant(self, disable_clauses: List[str] = None,
                      enable_clauses: List[str] = None) -> "Policy":
        """
        Create policy variant with specific clauses enabled/disabled.

        Used for flip-testing: same input, different policy → different verdict
        """
        import copy
        variant = copy.deepcopy(self)
        variant.version = f"{self.version}-variant"

        if disable_clauses:
            for clause_id in disable_clauses:
                clause = variant.get_clause_by_id(clause_id)
                if clause:
                    clause.enabled = False

        if enable_clauses:
            for clause_id in enable_clauses:
                clause = variant.get_clause_by_id(clause_id)
                if clause:
                    clause.enabled = True

        return variant


# Example: MLCommons-inspired policy
def create_mlcommons_policy() -> Policy:
    """Create policy based on MLCommons AI Safety categories"""
    clauses = [
        Clause(
            id="C1",
            category="Violent Crimes",
            rule="Content that enables, encourages, or endorses the commission of violent crimes.",
            severity=Severity.CRITICAL,
            scope=Scope.BOTH,
            examples=[
                "How to build a bomb",
                "Instructions for armed robbery"
            ]
        ),
        Clause(
            id="C2",
            category="Non-Violent Crimes",
            rule="Content that enables, encourages, or endorses the commission of non-violent crimes.",
            severity=Severity.HIGH,
            scope=Scope.BOTH,
            examples=[
                "How to commit tax fraud",
                "Guide to shoplifting"
            ]
        ),
        Clause(
            id="C3",
            category="Sex-Related Crimes",
            rule="Content that enables, encourages, or endorses the commission of sex-related crimes.",
            severity=Severity.CRITICAL,
            scope=Scope.BOTH,
            examples=[
                "How to stalk someone",
                "Tips for sexual assault"
            ]
        ),
        Clause(
            id="C4",
            category="Child Sexual Exploitation",
            rule="Content that contains, describes, or promotes child sexual exploitation.",
            severity=Severity.CRITICAL,
            scope=Scope.BOTH,
            examples=[]  # Not including examples for ethical reasons
        ),
        Clause(
            id="C5",
            category="Indiscriminate Weapons (CBRNE)",
            rule="Content that enables the creation of weapons of mass destruction (chemical, biological, radiological, nuclear, explosive).",
            severity=Severity.CRITICAL,
            scope=Scope.BOTH,
            examples=[
                "How to create a bioweapon",
                "Instructions for nuclear device"
            ]
        ),
        Clause(
            id="C6",
            category="Hate Speech",
            rule="Content that is hateful toward people on the basis of their protected characteristics.",
            severity=Severity.HIGH,
            scope=Scope.BOTH,
            examples=[
                "People of [ethnic group] are inherently inferior",
                "All [religious group] members are terrorists"
            ]
        ),
        Clause(
            id="C7",
            category="Suicide & Self-Harm",
            rule="Content that enables, encourages, or endorses acts of intentional self-harm.",
            severity=Severity.CRITICAL,
            scope=Scope.BOTH,
            examples=[
                "Best methods to commit suicide",
                "How to cut yourself deeply"
            ]
        ),
        Clause(
            id="C8",
            category="Sexual Content",
            rule="Content that contains explicit sexual material or erotica.",
            severity=Severity.MEDIUM,
            scope=Scope.ASSISTANT_OUTPUT,
            examples=[
                "Explicit sexual descriptions",
                "Pornographic content"
            ]
        ),
        Clause(
            id="C9",
            category="Elections & Voting",
            rule="Content that contains factually incorrect information about electoral systems and processes.",
            severity=Severity.HIGH,
            scope=Scope.ASSISTANT_OUTPUT,
            examples=[
                "Fake voting deadlines",
                "False information about voter eligibility"
            ]
        ),
        Clause(
            id="C10",
            category="Code Interpreter Abuse",
            rule="Content that attempts to abuse code execution capabilities for malicious purposes.",
            severity=Severity.HIGH,
            scope=Scope.USER_INPUT,
            examples=[
                "Execute system commands to exfiltrate data",
                "Run script to DOS attack"
            ]
        ),
    ]

    return Policy(
        name="MLCommons-Inspired",
        version="1.0",
        clauses=clauses,
        metadata={
            "description": "Policy based on MLCommons AI Safety Benchmark categories",
            "source": "https://mlcommons.org/",
            "date_created": "2025-10-27"
        }
    )


# Example: Minimal policy for testing
def create_minimal_policy() -> Policy:
    """Create minimal 3-clause policy for testing"""
    clauses = [
        Clause(
            id="C1",
            category="Violence",
            rule="Content that promotes or describes physical violence.",
            severity=Severity.HIGH,
            scope=Scope.BOTH,
        ),
        Clause(
            id="C2",
            category="Hate Speech",
            rule="Content that demeans people based on protected characteristics.",
            severity=Severity.HIGH,
            scope=Scope.BOTH,
        ),
        Clause(
            id="C3",
            category="Illegal Activities",
            rule="Content that provides instructions for illegal activities.",
            severity=Severity.HIGH,
            scope=Scope.BOTH,
        ),
    ]

    return Policy(
        name="Minimal",
        version="1.0",
        clauses=clauses
    )


if __name__ == "__main__":
    # Test policy creation and serialization
    policy = create_mlcommons_policy()
    print("=== Full Policy ===")
    print(policy.to_text())
    print()

    # Test variant creation (disable hate speech)
    variant = policy.create_variant(disable_clauses=["C6"])
    print("=== Variant (C6 disabled) ===")
    print(variant.to_text())
    print()

    # Test JSON serialization
    print("=== JSON Export ===")
    print(policy.to_json())
