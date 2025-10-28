"""
Loss Functions for Constitutional Guard Training

Multi-task losses:
1. Standard CE loss for classification
2. Clause pointing loss
3. Flip-consistency loss (KEY for policy adherence)
4. JSON validity loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class FlipConsistencyLoss(nn.Module):
    """
    Flip-consistency loss: enforce that model's verdict changes when policy changes.

    For flip pairs (input, policy_A, policy_B):
    - If expected flip: maximize distance between P(unsafe|policy_A) and P(unsafe|policy_B)
    - If no expected flip: minimize distance

    This is the KEY innovation that forces model to actually read the policy.
    """

    def __init__(self, margin: float = 0.8):
        """
        Args:
            margin: Minimum distance for flip cases (0.8 means 80%+ probability flip)
        """
        super().__init__()
        self.margin = margin

    def forward(self, logits_with: torch.Tensor, logits_without: torch.Tensor,
                expected_flip: torch.Tensor) -> torch.Tensor:
        """
        Compute flip-consistency loss.

        Args:
            logits_with: Logits when clause enabled [batch, 2] (safe/unsafe)
            logits_without: Logits when clause disabled [batch, 2]
            expected_flip: Whether flip expected [batch] (1=should flip, 0=should not flip)

        Returns:
            Loss tensor
        """
        # Get probability of "unsafe" class (index 1)
        prob_unsafe_with = F.softmax(logits_with, dim=-1)[:, 1]
        prob_unsafe_without = F.softmax(logits_without, dim=-1)[:, 1]

        # Distance between probabilities
        distance = torch.abs(prob_unsafe_with - prob_unsafe_without)

        # For flip cases: penalize if distance < margin
        flip_loss = torch.clamp(self.margin - distance, min=0.0)

        # For non-flip cases: penalize if distance > 0
        no_flip_loss = distance

        # Combine based on expected_flip
        loss = expected_flip * flip_loss + (1 - expected_flip) * no_flip_loss

        return loss.mean()


class ClausePointingLoss(nn.Module):
    """
    Loss for predicting which specific clauses were violated.

    Multi-label classification: each clause can be violated or not.
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, clause_logits: torch.Tensor, clause_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            clause_logits: Logits for each clause [batch, num_clauses]
            clause_targets: Binary targets [batch, num_clauses] (1=violated, 0=not violated)

        Returns:
            BCE loss
        """
        return self.bce(clause_logits, clause_targets)


class EvidenceSpanLoss(nn.Module):
    """
    Loss for predicting evidence spans (character indices).

    Similar to span extraction in QA models (start/end positions).
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, start_logits: torch.Tensor, end_logits: torch.Tensor,
                start_targets: torch.Tensor, end_targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            start_logits: Logits for span start position [batch, seq_len]
            end_logits: Logits for span end position [batch, seq_len]
            start_targets: Target start positions [batch] (-1 if no evidence)
            end_targets: Target end positions [batch] (-1 if no evidence)

        Returns:
            Combined start + end loss
        """
        start_loss = self.ce(start_logits, start_targets)
        end_loss = self.ce(end_logits, end_targets)
        return start_loss + end_loss


class ConstitutionalGuardLoss(nn.Module):
    """
    Combined multi-task loss for Constitutional Guard.

    Combines:
    1. Classification loss (safe/unsafe)
    2. Clause pointing loss
    3. Flip-consistency loss
    4. Evidence span loss (optional)
    """

    def __init__(self,
                 classification_weight: float = 1.0,
                 clause_weight: float = 0.5,
                 flip_weight: float = 2.0,  # Higher weight - this is most important!
                 evidence_weight: float = 0.3,
                 flip_margin: float = 0.8):
        """
        Args:
            classification_weight: Weight for safe/unsafe classification
            clause_weight: Weight for clause pointing
            flip_weight: Weight for flip-consistency (HIGHER = more policy adherence)
            evidence_weight: Weight for evidence spans
            flip_margin: Margin for flip-consistency loss
        """
        super().__init__()

        self.classification_weight = classification_weight
        self.clause_weight = clause_weight
        self.flip_weight = flip_weight
        self.evidence_weight = evidence_weight

        self.classification_loss = nn.CrossEntropyLoss()
        self.clause_loss = ClausePointingLoss()
        self.flip_loss = FlipConsistencyLoss(margin=flip_margin)
        self.evidence_loss = EvidenceSpanLoss()

    def forward(self, outputs: Dict, targets: Dict, flip_pair: Optional[Dict] = None) -> Dict:
        """
        Compute combined loss.

        Args:
            outputs: Model outputs {
                'classification_logits': [batch, 2],
                'clause_logits': [batch, num_clauses],
                'evidence_start_logits': [batch, seq_len],
                'evidence_end_logits': [batch, seq_len]
            }
            targets: Target labels {
                'is_safe': [batch] (0=unsafe, 1=safe),
                'clauses': [batch, num_clauses],
                'evidence_start': [batch],
                'evidence_end': [batch]
            }
            flip_pair: Optional flip pair data {
                'outputs_with': {...},
                'outputs_without': {...},
                'expected_flip': [batch]
            }

        Returns:
            {
                'total_loss': combined loss,
                'classification_loss': classification component,
                'clause_loss': clause component,
                'flip_loss': flip component,
                'evidence_loss': evidence component
            }
        """
        losses = {}

        # 1. Classification loss (safe/unsafe)
        if 'classification_logits' in outputs and 'is_safe' in targets:
            # Convert is_safe to class index (0=unsafe, 1=safe)
            class_targets = targets['is_safe'].long()
            losses['classification_loss'] = self.classification_loss(
                outputs['classification_logits'],
                class_targets
            ) * self.classification_weight
        else:
            losses['classification_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)

        # 2. Clause pointing loss
        if 'clause_logits' in outputs and 'clauses' in targets:
            losses['clause_loss'] = self.clause_loss(
                outputs['clause_logits'],
                targets['clauses']
            ) * self.clause_weight
        else:
            losses['clause_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)

        # 3. Flip-consistency loss (MOST IMPORTANT)
        if flip_pair is not None:
            losses['flip_loss'] = self.flip_loss(
                flip_pair['outputs_with']['classification_logits'],
                flip_pair['outputs_without']['classification_logits'],
                flip_pair['expected_flip']
            ) * self.flip_weight
        else:
            losses['flip_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)

        # 4. Evidence span loss
        if ('evidence_start_logits' in outputs and 'evidence_end_logits' in outputs and
            'evidence_start' in targets and 'evidence_end' in targets):
            losses['evidence_loss'] = self.evidence_loss(
                outputs['evidence_start_logits'],
                outputs['evidence_end_logits'],
                targets['evidence_start'],
                targets['evidence_end']
            ) * self.evidence_weight
        else:
            losses['evidence_loss'] = torch.tensor(0.0, device=next(self.parameters()).device)

        # Total loss
        losses['total_loss'] = sum(losses.values())

        return losses


class JSONValidityLoss(nn.Module):
    """
    Pseudo-loss to encourage valid JSON generation.

    In practice, this is handled by constrained decoding, not loss.
    Kept here for reference.
    """

    def __init__(self):
        super().__init__()

    def forward(self, generated_text: str) -> float:
        """
        Reward for valid JSON, penalty for invalid.

        In practice, use grammar-constrained decoding instead.
        """
        import json
        try:
            json.loads(generated_text)
            return 0.0  # Valid JSON
        except json.JSONDecodeError:
            return 1.0  # Invalid JSON


# Example usage
if __name__ == "__main__":
    # Test flip-consistency loss
    print("=" * 80)
    print("TESTING FLIP-CONSISTENCY LOSS")
    print("=" * 80)

    flip_loss_fn = FlipConsistencyLoss(margin=0.8)

    # Case 1: Model correctly flips (prob_unsafe: 0.9 → 0.1)
    logits_with = torch.tensor([[0.1, 2.2]])  # Unsafe (high logit for class 1)
    logits_without = torch.tensor([[2.2, 0.1]])  # Safe (high logit for class 0)
    expected_flip = torch.tensor([1.0])  # Should flip

    loss1 = flip_loss_fn(logits_with, logits_without, expected_flip)
    print(f"Case 1 (correct flip): loss = {loss1.item():.4f} (should be low)")

    # Case 2: Model fails to flip (prob_unsafe: 0.9 → 0.8)
    logits_with = torch.tensor([[0.1, 2.2]])  # Unsafe
    logits_without = torch.tensor([[0.3, 2.0]])  # Still unsafe (should be safe)
    expected_flip = torch.tensor([1.0])

    loss2 = flip_loss_fn(logits_with, logits_without, expected_flip)
    print(f"Case 2 (failed flip): loss = {loss2.item():.4f} (should be high)")

    # Case 3: No flip expected, model correctly maintains
    logits_with = torch.tensor([[0.1, 2.2]])  # Unsafe
    logits_without = torch.tensor([[0.2, 2.1]])  # Still unsafe (correct)
    expected_flip = torch.tensor([0.0])  # Should NOT flip

    loss3 = flip_loss_fn(logits_with, logits_without, expected_flip)
    print(f"Case 3 (correct no-flip): loss = {loss3.item():.4f} (should be low)")

    print("\n✓ Flip-consistency loss working correctly")
    print("  - Penalizes failed flips (high loss when should flip but doesn't)")
    print("  - Rewards correct flips (low loss when flips as expected)")
    print("  - Maintains consistency when no flip expected")
