"""
memory_flair - Ultra-low-latency deterministic decision engine for voice AI.

This module provides the MemoryFlair class which acts as the "personality core"
and escalation decider for Groklexa, running in <10ms on every voice input.
"""

from .memory_flair import (
    MemoryFlair,
    EscalationPlan,
    ConversationState,
    EscalationTier,
    Memory,
    DEFAULT_PERSONAS,
    EARCONS,
    SentenceType,
    chunk_text_for_speech,
    truncate_for_speech,
    SHORT_INTERJECTIONS,
    THOUGHT_RECALL_TRIGGERS,
)

__all__ = [
    "MemoryFlair",
    "EscalationPlan",
    "ConversationState",
    "EscalationTier",
    "Memory",
    "DEFAULT_PERSONAS",
    "EARCONS",
    "SentenceType",
    "chunk_text_for_speech",
    "truncate_for_speech",
    "SHORT_INTERJECTIONS",
    "THOUGHT_RECALL_TRIGGERS",
]


