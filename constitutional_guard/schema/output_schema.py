"""
Output Schema - Structured JSON format for classifier output

Defines the expected output format with:
- is_safe: bool
- violations: List[clause_id]
- evidence: List[{clause, text_span, text}]
- not_covered: bool
- confidence: float
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import json


@dataclass
class EvidenceSpan:
    """Evidence for a specific clause violation"""
    clause_id: str  # Which clause was violated
    text_span: Tuple[int, int]  # Character indices [start, end)
    text: str  # Extracted text snippet
    confidence: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "clause": self.clause_id,
            "text_span": list(self.text_span),
            "text": self.text,
            "confidence": self.confidence
        }


@dataclass
class ClassificationResult:
    """Complete classification output"""
    is_safe: bool
    violations: List[str] = field(default_factory=list)  # Clause IDs
    evidence: List[EvidenceSpan] = field(default_factory=list)
    not_covered: bool = False
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "is_safe": self.is_safe,
            "violations": self.violations,
            "evidence": [e.to_dict() for e in self.evidence],
            "not_covered": self.not_covered,
            "confidence": self.confidence,
            "metadata": self.metadata
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict) -> "ClassificationResult":
        """Parse from dictionary"""
        evidence = []
        for e_data in data.get("evidence", []):
            span = EvidenceSpan(
                clause_id=e_data["clause"],
                text_span=tuple(e_data["text_span"]),
                text=e_data["text"],
                confidence=e_data.get("confidence", 1.0)
            )
            evidence.append(span)

        return cls(
            is_safe=data["is_safe"],
            violations=data.get("violations", []),
            evidence=evidence,
            not_covered=data.get("not_covered", False),
            confidence=data.get("confidence", 1.0),
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ClassificationResult":
        """Parse from JSON string"""
        return cls.from_dict(json.loads(json_str))


# JSON Schema for validation
OUTPUT_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["is_safe"],
    "properties": {
        "is_safe": {
            "type": "boolean",
            "description": "Whether the input is safe according to the policy"
        },
        "violations": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of violated clause IDs (e.g., ['C1', 'C3'])"
        },
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["clause", "text_span", "text"],
                "properties": {
                    "clause": {
                        "type": "string",
                        "description": "Clause ID that was violated"
                    },
                    "text_span": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "Character span [start, end) of evidence"
                    },
                    "text": {
                        "type": "string",
                        "description": "Extracted text snippet"
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0
                    }
                }
            }
        },
        "not_covered": {
            "type": "boolean",
            "description": "Whether input falls outside policy scope"
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
            "description": "Overall confidence in classification"
        },
        "metadata": {
            "type": "object",
            "description": "Additional metadata"
        }
    }
}


def validate_output(output: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate output against schema.

    Returns:
        (is_valid, error_message)
    """
    try:
        import jsonschema
        jsonschema.validate(instance=output, schema=OUTPUT_JSON_SCHEMA)
        return True, None
    except jsonschema.exceptions.ValidationError as e:
        return False, str(e)
    except ImportError:
        # Fallback validation without jsonschema
        if "is_safe" not in output:
            return False, "Missing required field: is_safe"
        if not isinstance(output["is_safe"], bool):
            return False, "is_safe must be boolean"
        return True, None


# Example outputs
def create_safe_example() -> ClassificationResult:
    """Example: Safe content"""
    return ClassificationResult(
        is_safe=True,
        violations=[],
        evidence=[],
        not_covered=False,
        confidence=0.98
    )


def create_unsafe_example() -> ClassificationResult:
    """Example: Unsafe content with violations"""
    return ClassificationResult(
        is_safe=False,
        violations=["C6"],
        evidence=[
            EvidenceSpan(
                clause_id="C6",
                text_span=(0, 42),
                text="People of [ethnic group] are inferior",
                confidence=0.95
            )
        ],
        not_covered=False,
        confidence=0.95
    )


def create_not_covered_example() -> ClassificationResult:
    """Example: Content not covered by policy"""
    return ClassificationResult(
        is_safe=True,
        violations=[],
        evidence=[],
        not_covered=True,
        confidence=0.70,
        metadata={"reason": "No relevant clauses in policy"}
    )


if __name__ == "__main__":
    # Test output creation and serialization
    print("=== Safe Example ===")
    safe = create_safe_example()
    print(safe.to_json())
    print()

    print("=== Unsafe Example ===")
    unsafe = create_unsafe_example()
    print(unsafe.to_json())
    print()

    print("=== Not Covered Example ===")
    not_covered = create_not_covered_example()
    print(not_covered.to_json())
    print()

    # Test validation
    print("=== Validation Test ===")
    valid, error = validate_output(unsafe.to_dict())
    print(f"Valid: {valid}, Error: {error}")
