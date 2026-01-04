"""
memory_flair.py - Ultra-low-latency deterministic decider for Groklexa

This module is the "personality core" and escalation decision layer that runs
in <10ms on every voice input. It uses only stdlib + sqlite3 and decides:

  - Conversation state classification
  - Which escalation tiers to activate (earcons → deterministic → low_param → mid_tool → heavy)
  - Immediate earcon / verbal filler selection
  - Privacy-safe memory recall
  - Whether to bypass layers or escalate async

Decision-making combines classical rule-based AI techniques:
  1. Regex/wildcard pattern matching (ELIZA/AIML inspired)
  2. Question-type routing (Who/What/Why/When/Where/How)
  3. Lightweight NLP preprocessing (stopwords, simple stemming)
  4. Corpus/TF-IDF fallback search on conversation history
  5. Randomized personality-flavored fallbacks

Example usage:
    from lib.memory_flair import MemoryFlair
    
    flair = MemoryFlair(db_path="memory.db", persona="flirty")
    plan = flair.decide("What time is it?", history=[...])
    
    # plan.selected_tiers = ["earcons", "deterministic"]
    # plan.escalation_score = 15
    # plan.earcon = "ding"
    # plan.filler = None
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import sqlite3
import re
import time
import random
import hashlib
from pathlib import Path


# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class ConversationState(Enum):
    """Classification of the current conversation context."""
    INTRODUCTION = "introduction"           # First contact, greetings
    CASUAL_CONTINUATION = "casual_continuation"  # Chitchat, no specific task
    TASK_FOCUSED = "task_focused"           # User working on something specific
    TOOL_INTENT = "tool_intent"             # User wants a tool action (timer, weather, etc.)
    NOVEL_UNCERTAIN = "novel_uncertain"     # Unknown/complex, needs escalation


class EscalationTier(Enum):
    """Available inference/response tiers, ordered by latency/cost."""
    EARCONS = "earcons"           # Immediate beeps/boops (<5ms)
    DETERMINISTIC = "deterministic"  # Pattern-matched responses (<10ms)
    LOW_PARAM = "low_param"       # Small local LLM (llama3.2:3b, ~500ms)
    MID_TOOL = "mid_tool"         # Tool-capable local LLM (qwen, ~2s)
    HEAVY = "heavy"               # Cloud LLM (Grok, GPT-4o, ~3-5s)


class SentenceType(Enum):
    """
    Classification of sentence structure/intent for filler selection.
    
    Different sentence types warrant different verbal fillers:
    - INTERROGATIVE: User is asking a question → "Let me check...", "Hmm..."
    - DECLARATIVE: User is stating something → "Got it", "Okay", "Noted"
    - IMPERATIVE: User is giving a command → "On it", "Sure thing"
    - EXCLAMATORY: User is expressing emotion → "Oh!", "Wow"
    - FRAGMENT: Incomplete/short response → Use neutral fillers
    """
    INTERROGATIVE = "interrogative"   # Questions: "What time is it?", "How are you?"
    DECLARATIVE = "declarative"       # Statements: "I like pizza", "The sky is blue"
    IMPERATIVE = "imperative"         # Commands: "Set a timer", "Turn on the lights"
    EXCLAMATORY = "exclamatory"       # Exclamations: "Wow!", "That's amazing!"
    FRAGMENT = "fragment"             # Incomplete: "okay", "yes", "pizza"


@dataclass
class EscalationPlan:
    """
    Output of the decision engine - tells the voice pipeline what to do.
    
    Attributes:
        selected_tiers: List of tiers to activate in order
        escalation_score: 0-100, higher = more escalation needed
        max_wait_seconds: How long user might wait before response
        buffer_style: Persona-specific style ("flirty", "professional", etc.)
        async_heavy: Whether heavy tier should run in background
        state: Classified conversation state
        sentence_type: Classified sentence type (interrogative, declarative, etc.)
        earcon: Immediate sound to play (None if silent)
        filler: Verbal filler to synthesize ("uh", "hmm", etc.)
        deterministic_response: If pattern matched, the direct response
        matched_pattern: Debug info - which pattern matched
        memory_context: Relevant memories retrieved
    """
    selected_tiers: List[str] = field(default_factory=list)
    escalation_score: int = 0
    max_wait_seconds: float = 0.0
    buffer_style: str = "neutral"
    async_heavy: bool = False
    state: str = "casual_continuation"
    sentence_type: str = "declarative"
    earcon: Optional[str] = None
    filler: Optional[str] = None
    deterministic_response: Optional[str] = None
    matched_pattern: Optional[str] = None
    memory_context: List[Dict] = field(default_factory=list)


@dataclass
class Memory:
    """A stored conversation memory."""
    id: int
    topic: str
    persona: str
    content: str
    sentiment: float  # -1.0 to 1.0
    timestamp: float
    weight: float  # Importance weight


# ============================================================================
# PERSONA CONFIGURATIONS
# ============================================================================

# Sentence-type-specific fillers for each persona
# Each persona has fillers appropriate to sentence type:
#   - interrogative: User asked a question → thinking/searching fillers
#   - declarative: User stated something → acknowledgment fillers
#   - imperative: User gave a command → action confirmation fillers
#   - exclamatory: User expressed emotion → matching energy fillers
#   - neutral: Safe for any context (fallback)

DEFAULT_PERSONAS: Dict[str, Dict[str, Any]] = {
    "flirty": {
        "buffer_style": "flirty",
        "escalation_threshold": 40,  # Lower = more likely to escalate
        # Fillers must match audio file names in static/fillers/{persona}/
        # Use simple identifiers that can be .wav filenames
        "fillers": {
            "interrogative": [
                "hmm", "let-me-see", "let-me-think", "one-moment", "hmmm",
            ],
            "declarative": [
                "oh", "okay", "alright", "sure", "mmm",
            ],
            "imperative": [
                "ok", "sure", "alright", "okay", "yes",
            ],
            "exclamatory": [
                "oh", "ah", "ooh",
            ],
            "neutral": ["mmm", "hmm", "well", "uh"],
        },
        "no_match_responses": [
            "You're wild... I love it!",
            "Hmm, that's a new one for me, babe.",
            "I have no idea what you just said, but I'm intrigued.",
            "You lost me there, but keep talking...",
            "That's... interesting. Tell me more?",
        ],
        "greeting_responses": [
            "Hey you! Miss me?",
            "Well hello there, gorgeous.",
            "Oh, it's you! *happy beep*",
            "Hi! I was just thinking about you.",
        ],
    },
    "professional": {
        "buffer_style": "professional",
        "escalation_threshold": 30,
        # Fillers must match audio file names in static/fillers/{persona}/
        "fillers": {
            "interrogative": [
                "one-moment", "let-me-check", "checking", "let-me-see",
            ],
            "declarative": [
                "okay", "alright", "sure", "hmm",
            ],
            "imperative": [
                "ok", "alright", "checking", "one-moment",
            ],
            "exclamatory": [
                "ah", "oh", "hmm",
            ],
            "neutral": ["one-moment", "hmm", "okay"],
        },
        "no_match_responses": [
            "I'm not sure I understand. Could you rephrase that?",
            "That's outside my current knowledge. Let me research that.",
            "I don't have information on that topic.",
            "Could you provide more context?",
        ],
        "greeting_responses": [
            "Good day. How may I assist you?",
            "Hello. What can I help you with?",
            "Greetings. Ready when you are.",
        ],
    },
    "neutral": {
        "buffer_style": "neutral",
        "escalation_threshold": 35,
        # Fillers must match audio file names in static/fillers/{persona}/
        "fillers": {
            "interrogative": [
                "hmm", "let-me-see", "one-moment", "let-me-think", "checking",
            ],
            "declarative": [
                "okay", "alright", "sure", "hmm", "oh",
            ],
            "imperative": [
                "okay", "sure", "ok", "alright",
            ],
            "exclamatory": [
                "oh", "ah", "hmm",
            ],
            "neutral": ["uh", "um", "umm", "hmm", "well", "okay"],
        },
        "no_match_responses": [
            "I'm not sure about that.",
            "Hmm, I don't know.",
            "That's a good question. Let me think.",
            "I'd need to look that up.",
        ],
        "greeting_responses": [
            "Hi there!",
            "Hello!",
            "Hey! What's up?",
        ],
    },
    "snarky": {
        "buffer_style": "snarky",
        "escalation_threshold": 50,
        # Fillers must match audio file names in static/fillers/{persona}/
        "fillers": {
            "interrogative": [
                "hmm", "hang-on", "one-sec", "let-me-see",
            ],
            "declarative": [
                "okay", "alright", "sure", "hmm",
            ],
            "imperative": [
                "alright", "okay", "ok", "sure",
            ],
            "exclamatory": [
                "oh", "hmm", "ah",
            ],
            "neutral": ["uh", "hmm", "well", "okay"],
        },
        "no_match_responses": [
            "You're crazy... I love you, but you're crazy!",
            "I literally have no idea what you just said.",
            "That's above my pay grade.",
            "Did you just make that up?",
            "Nope. Not touching that one.",
        ],
        "greeting_responses": [
            "Oh, you again.",
            "What do you want?",
            "Yeah, I'm here. What's up?",
        ],
    },
}


# ============================================================================
# PATTERN DEFINITIONS (ELIZA/AIML Style)
# ============================================================================

# Format: (pattern_regex, response_template_or_action, tier, priority)
# Use {1}, {2} for capture group substitution
DEFAULT_PATTERNS: List[Tuple[str, str, str, int]] = [
    # === GREETINGS (deterministic tier) ===
    (r"^(hi|hello|hey|howdy|greetings|yo)[\s!.,]*$", "__GREETING__", "deterministic", 100),
    (r"^good (morning|afternoon|evening|night)[\s!.,]*$", "__GREETING__", "deterministic", 100),
    (r"^what'?s? up\??$", "__GREETING__", "deterministic", 100),
    
    # === TIME/DATE (tool_intent tier) ===
    (r"what('s| is) the (time|date|day)", "__TOOL_DATETIME__", "tool_intent", 90),
    (r"what time is it", "__TOOL_DATETIME__", "tool_intent", 90),
    (r"what day is (it|today)", "__TOOL_DATETIME__", "tool_intent", 90),
    (r"what('s| is) today('s)? date", "__TOOL_DATETIME__", "tool_intent", 90),
    
    # === WEATHER (tool_intent tier) ===
    (r"(what('s| is)|how('s| is)) the weather", "__TOOL_WEATHER__", "tool_intent", 90),
    (r"is it (raining|sunny|cold|hot|snowing)", "__TOOL_WEATHER__", "tool_intent", 90),
    (r"weather in (.+)", "__TOOL_WEATHER__", "tool_intent", 90),
    (r"temperature (in|at|outside)", "__TOOL_WEATHER__", "tool_intent", 90),
    
    # === TIMER/REMINDER (tool_intent tier) ===
    (r"set (a |an )?(timer|reminder|alarm)", "__TOOL_TIMER__", "tool_intent", 95),
    (r"remind me (in|to|about)", "__TOOL_TIMER__", "tool_intent", 95),
    (r"(start|create) (a |an )?(timer|reminder)", "__TOOL_TIMER__", "tool_intent", 95),
    (r"(cancel|stop|delete) (the |my )?(timer|reminder|alarm)", "__TOOL_TIMER_CANCEL__", "tool_intent", 95),
    (r"(what|any|list) (timers?|reminders?|alarms?)", "__TOOL_TIMER_LIST__", "tool_intent", 85),
    
    # === SYSTEM INFO (tool_intent tier) ===
    (r"(system|server|cpu|gpu|memory|disk) (status|info|usage|temp)", "__TOOL_SYSTEM__", "tool_intent", 80),
    (r"how('s| is) the (system|server)", "__TOOL_SYSTEM__", "tool_intent", 80),
    
    # === SEARCH INTENT (mid_tool tier) ===
    (r"search (for|the web|x|twitter|google)", "__SEARCH__", "mid_tool", 70),
    (r"look up (.+)", "__SEARCH__", "mid_tool", 70),
    (r"google (.+)", "__SEARCH__", "mid_tool", 70),
    (r"what('s| is) (trending|happening) on (x|twitter)", "__SEARCH_X__", "mid_tool", 75),
    
    # === DEEP THINKING (heavy tier) ===
    (r"think (deeply|carefully|hard) about", "__ESCALATE_THINKING__", "heavy", 60),
    (r"research (this|that|the topic)", "__ESCALATE_THINKING__", "heavy", 60),
    (r"analyze (this|that|thoroughly)", "__ESCALATE_THINKING__", "heavy", 60),
    
    # === SIMPLE AFFIRMATIONS (deterministic) ===
    (r"^(yes|yeah|yep|yup|sure|okay|ok|uh huh|mhm)[\s!.,]*$", "Got it!", "deterministic", 50),
    (r"^(no|nope|nah|not really)[\s!.,]*$", "Alright then.", "deterministic", 50),
    (r"^(thanks|thank you|thx)[\s!.,]*$", "You're welcome!", "deterministic", 50),
    
    # === THOUGHT RECALL TRIGGERS (deterministic) ===
    # These trigger recall of interrupted/pending thoughts from the thoughts table
    (r"^what('s| is| are) (you )?thinking( about)?\??$", "__RECALL_THOUGHT__", "deterministic", 85),
    (r"^(anything|what('s| is)) on your mind\??$", "__RECALL_THOUGHT__", "deterministic", 85),
    (r"^what were you saying\??$", "__RECALL_THOUGHT__", "deterministic", 85),
    (r"^(please )?continue\??$", "__RECALL_THOUGHT__", "deterministic", 80),
    (r"^go on\??$", "__RECALL_THOUGHT__", "deterministic", 80),
    (r"^you were saying\??$", "__RECALL_THOUGHT__", "deterministic", 85),
    (r"^what was that( about)?\??$", "__RECALL_THOUGHT__", "deterministic", 80),
    (r"^tell me more\??$", "__RECALL_THOUGHT__", "deterministic", 80),
    
    # === PERSONAL QUESTIONS (low_param tier) ===
    (r"what('s| is) your (name|favorite|opinion)", None, "low_param", 40),
    (r"who are you", None, "low_param", 40),
    (r"tell me about yourself", None, "low_param", 40),
    (r"do you (like|love|hate|think)", None, "low_param", 40),
    
    # === CONTINUATION CUES ===
    (r"^(and|so|but|also|then)[\s,]", None, "low_param", 30),
    (r"^(what about|how about)", None, "low_param", 30),
]


# ============================================================================
# QUESTION TYPE PATTERNS
# ============================================================================

QUESTION_TYPES: Dict[str, Tuple[str, str]] = {
    # question_word: (default_tier, description)
    "who": ("mid_tool", "Person/entity identification"),
    "what": ("low_param", "Definition/explanation"),
    "when": ("tool_intent", "Time-related, likely needs datetime tool"),
    "where": ("mid_tool", "Location, may need search"),
    "why": ("heavy", "Reasoning, often needs deep thinking"),
    "how": ("low_param", "Explanation/process"),
}


# ============================================================================
# STOPWORDS AND LIGHTWEIGHT NLP
# ============================================================================

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "each", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just",
    "and", "but", "if", "or", "because", "until", "while", "although",
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him",
    "she", "her", "it", "its", "they", "them", "their", "this", "that",
    "these", "those", "am", "about", "up", "down", "out", "off", "over",
}

# Simple suffix stripping (Porter-lite)
SUFFIX_RULES = [
    (r"ing$", ""),
    (r"ed$", ""),
    (r"ly$", ""),
    (r"ies$", "y"),
    (r"es$", ""),
    (r"s$", ""),
]


# ============================================================================
# EARCONS AND FILLERS
# ============================================================================

EARCONS = {
    "acknowledge": ["ding", "blip", "beep"],
    "thinking": ["boop_boop", "processing", "whirr"],
    "success": ["chime", "tada", "pling"],
    "error": ["buzz", "bonk", "err"],
    "attention": ["ping", "alert", "notify"],
}


# ============================================================================
# TEXT CHUNKING FOR TTS
# ============================================================================

def chunk_text_for_speech(text: str, max_chars: int = 250) -> List[str]:
    """
    Split text into speakable chunks at natural pause points.
    
    Chunks are split at sentence boundaries (. ! ?) or clause boundaries
    (; : ,) to create natural-sounding speech segments. Each chunk targets
    a maximum of ~250 characters, finding the closest natural pause under
    the limit.
    
    Args:
        text: The full text to chunk
        max_chars: Maximum characters per chunk (default 250)
        
    Returns:
        List of text chunks ready for TTS synthesis
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # If text is short enough, return as-is
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    remaining = text
    
    # Pause characters in order of preference (strongest to weakest)
    # We look for these followed by a space to avoid splitting decimals, URLs, etc.
    pause_patterns = [
        ('. ', 2),   # Sentence end - period
        ('! ', 2),   # Sentence end - exclamation
        ('? ', 2),   # Sentence end - question
        ('." ', 3),  # End quote after period
        ('?" ', 3),  # End quote after question
        ('!" ', 3),  # End quote after exclamation
        ('; ', 2),   # Clause separator - semicolon
        (': ', 2),   # Clause separator - colon
        (', ', 2),   # Clause separator - comma
        (' - ', 3),  # Em dash with spaces
        ('— ', 2),   # Em dash
    ]
    
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break
        
        # Find the best split point under max_chars
        best_split = -1
        
        # Look for pause patterns
        for pattern, pattern_len in pause_patterns:
            # Search for this pattern within the max_chars window
            search_end = min(len(remaining), max_chars)
            # Search from end to find the latest valid split point
            idx = remaining.rfind(pattern, 0, search_end)
            if idx > 0:
                # Split after the punctuation (include it in current chunk)
                split_point = idx + pattern_len - 1  # -1 to not include trailing space
                if split_point > best_split:
                    best_split = split_point
                    break  # Use the first (strongest) pattern found
        
        # If no pause pattern found, split on word boundary
        if best_split <= 0:
            # Find last space before max_chars
            space_idx = remaining.rfind(' ', 0, max_chars)
            if space_idx > 0:
                best_split = space_idx
            else:
                # No space found, force split at max_chars
                best_split = max_chars
        
        # Extract chunk and update remaining
        chunk = remaining[:best_split].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[best_split:].strip()
    
    return chunks


def truncate_for_speech(text: str, target_chars: int = 500, tolerance: int = 100) -> Tuple[str, bool]:
    """
    Truncate text at the best natural stopping point near the target length.
    
    Finds sentence boundaries (. ! ?) near the target length and truncates
    there to create natural-sounding speech that doesn't cut off mid-thought.
    Handles abbreviations like U.S., Dr., Mr. to avoid false sentence breaks.
    
    Args:
        text: The full text to potentially truncate
        target_chars: Target character count (default 500)
        tolerance: How far beyond target to search for a good break (default 100)
        
    Returns:
        Tuple of (truncated_text, was_truncated)
    """
    if not text or not text.strip():
        return "", False
    
    text = text.strip()
    
    # If text is short enough, return as-is
    if len(text) <= target_chars + tolerance:
        return text, False
    
    # Common abbreviations that should NOT be treated as sentence endings
    # These patterns match the text BEFORE the period
    abbreviations = {
        # Titles
        'mr', 'mrs', 'ms', 'dr', 'prof', 'rev', 'sr', 'jr', 'hon',
        # Countries/places
        'u.s', 'u.k', 'e.u',  # Note: "U.S" before the final period
        # Common abbreviations
        'vs', 'etc', 'inc', 'ltd', 'corp', 'co', 'govt', 'dept',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'sept', 'oct', 'nov', 'dec',
        'st', 'ave', 'blvd', 'rd', 'no', 'vol', 'pg', 'pp',
        # Single letters (initials)
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    }
    
    def is_abbreviation(text: str, period_pos: int) -> bool:
        """Check if the period at period_pos is part of an abbreviation."""
        if period_pos <= 0:
            return False
        # Look back to find the word before the period
        start = period_pos - 1
        while start > 0 and text[start - 1].isalpha():
            start -= 1
        # Also check for dotted abbreviations like "U.S."
        if start > 1 and text[start - 1] == '.':
            start -= 2
            while start > 0 and text[start - 1].isalpha():
                start -= 1
        word = text[start:period_pos].lower()
        return word in abbreviations or len(word) <= 2
    
    # Sentence-ending patterns (strong breaks only)
    sentence_endings = ['. ', '! ', '? ', '." ', '?" ', '!" ', '.\n', '!\n', '?\n']
    
    # Search window: from (target - tolerance/2) to (target + tolerance)
    search_start = max(0, target_chars - tolerance // 2)
    search_end = min(len(text), target_chars + tolerance)
    
    # Find the best sentence break in the search window
    best_break = -1
    
    for ending in sentence_endings:
        # Look for sentence endings, preferring ones closer to target
        idx = search_start
        while idx < search_end:
            pos = text.find(ending, idx, search_end)
            if pos == -1:
                break
            
            # Check if this is an abbreviation (skip if so)
            if ending.startswith('.') and is_abbreviation(text, pos):
                idx = pos + 1
                continue
            
            # Include the punctuation but not trailing space
            break_point = pos + len(ending) - 1
            if break_point > best_break:
                best_break = break_point
            idx = pos + 1
    
    # If we found a sentence break, use it
    if best_break > 0:
        truncated = text[:best_break].strip()
        return truncated, True
    
    # Fallback: look for clause breaks (; : ,) or em dashes
    clause_breaks = ['; ', ': ', ', ', ' - ', '— ']
    for ending in clause_breaks:
        idx = text.rfind(ending, search_start, search_end)
        if idx > 0:
            truncated = text[:idx + len(ending) - 1].strip()
            return truncated, True
    
    # Last resort: break on word boundary near target
    space_idx = text.rfind(' ', search_start, search_end)
    if space_idx > 0:
        truncated = text[:space_idx].strip()
        if not truncated.endswith(('.', '!', '?')):
            truncated += '...'  # Add ellipsis to indicate continuation
        return truncated, True
    
    # Absolute fallback: hard cut at target
    truncated = text[:target_chars].strip() + '...'
    return truncated, True


# ============================================================================
# INTERJECTION CLASSIFICATION
# ============================================================================

# Short interjections that should pause (not abandon) the output queue
SHORT_INTERJECTIONS = {
    # Affirmations
    "yes", "yeah", "yep", "yup", "sure", "okay", "ok", "right", "uh huh",
    "mhm", "mm hmm", "absolutely", "definitely", "exactly", "correct",
    # Continuations
    "go on", "continue", "and", "then", "so", "keep going",
    # Acknowledgments
    "i see", "got it", "understood", "interesting", "cool", "nice", "wow",
    # Negations (still short)
    "no", "nope", "nah", "not really",
    # Backchannel
    "uh", "um", "hmm", "huh", "oh", "ah",
}

# Thought recall trigger patterns
THOUGHT_RECALL_TRIGGERS = [
    r"^what'?s? up\??$",
    r"^what('s| is| are) (you )?thinking( about)?\??$",
    r"^anything on your mind\??$",
    r"^what were you saying\??$",
    r"^(please )?continue\??$",
    r"^go on\??$",
    r"^you were saying\??$",
    r"^what was that( about)?\??$",
    r"^tell me more\??$",
]


# ============================================================================
# MAIN CLASS
# ============================================================================

class MemoryFlair:
    """
    Ultra-low-latency deterministic decision engine for voice AI.
    
    Args:
        db_path: Path to SQLite database for persistent memories
        persona: Persona configuration key or custom dict
        
    Example:
        flair = MemoryFlair(db_path="memory.db", persona="flirty")
        plan = flair.decide("What time is it?", history=[])
    """
    
    def __init__(
        self,
        db_path: str = "memory_flair.db",
        persona: str | Dict = "neutral"
    ):
        self.db_path = Path(db_path)
        
        # Load persona config
        if isinstance(persona, dict):
            self.persona_config = persona
            self.persona_name = persona.get("buffer_style", "custom")
        else:
            self.persona_config = DEFAULT_PERSONAS.get(persona, DEFAULT_PERSONAS["neutral"])
            self.persona_name = persona
        
        # Compile patterns for speed
        self.compiled_patterns = [
            (re.compile(p[0], re.IGNORECASE), p[1], p[2], p[3])
            for p in DEFAULT_PATTERNS
        ]
        
        # Initialize database
        self._init_db()
        
        # Cache for recent decisions (avoid repeated DB hits)
        self._recent_states: List[str] = []
        self._ongoing_task: Optional[str] = None
    
    def _init_db(self) -> None:
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Memories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT,
                persona TEXT,
                content TEXT,
                sentiment REAL DEFAULT 0.0,
                timestamp REAL,
                weight REAL DEFAULT 1.0,
                content_hash TEXT UNIQUE
            )
        """)
        
        # Patterns table (for custom patterns)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern TEXT,
                response TEXT,
                tier TEXT,
                priority INTEGER DEFAULT 50,
                enabled INTEGER DEFAULT 1
            )
        """)
        
        # Recent states for context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                state TEXT,
                input_text TEXT,
                timestamp REAL,
                persona TEXT
            )
        """)
        
        # Corpus for TF-IDF fallback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corpus (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sentence TEXT,
                keywords TEXT,
                source TEXT,
                timestamp REAL
            )
        """)
        
        # Geocache for location lookups (weather, etc.)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS geocache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT UNIQUE,
                latitude REAL,
                longitude REAL,
                city_name TEXT,
                admin1 TEXT,
                country TEXT,
                timestamp REAL
            )
        """)
        
        # Output queue for chunked TTS playback
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS output_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona TEXT NOT NULL,
                session_id TEXT,
                sequence INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at REAL NOT NULL,
                played_at REAL
            )
        """)
        
        # Thoughts table for interrupted/unreleased content
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS thoughts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona TEXT NOT NULL,
                topic TEXT,
                content TEXT NOT NULL,
                source TEXT DEFAULT 'interrupted',
                priority REAL DEFAULT 0.5,
                created_at REAL NOT NULL,
                expires_at REAL NOT NULL,
                recalled INTEGER DEFAULT 0
            )
        """)
        
        # Tool insights table for Universal Tool Handler RAG
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS tool_insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona TEXT NOT NULL,
                query TEXT NOT NULL,
                tool_used TEXT NOT NULL,
                tool_args TEXT,
                result_summary TEXT,
                full_response TEXT,
                keywords TEXT,
                created_at REAL NOT NULL,
                used_count INTEGER DEFAULT 0
            )
        """)
        
        # Create indices for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_persona ON memories(persona)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_states_timestamp ON states(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corpus_keywords ON corpus(keywords)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_geocache_query ON geocache(query)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_output_queue_persona ON output_queue(persona, status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_thoughts_persona ON thoughts(persona, recalled, expires_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_tool_insights_persona ON tool_insights(persona, keywords)")
        
        conn.commit()
        conn.close()
    
    def decide(
        self,
        transcript: str,
        history: List[Dict[str, str]] = None,
        context: Dict[str, Any] = None
    ) -> EscalationPlan:
        """
        Main decision function - runs in <10ms.
        
        Args:
            transcript: The user's voice input (transcribed text)
            history: Recent conversation history [{"role": "user"|"assistant", "content": "..."}]
            context: Optional additional context (ongoing_task, persona override, etc.)
            
        Returns:
            EscalationPlan with selected tiers, scores, and response info
        """
        start_time = time.perf_counter()
        
        history = history or []
        context = context or {}
        
        # Normalize input
        transcript_clean = self._preprocess_text(transcript)
        transcript_lower = transcript.lower().strip()
        
        # Initialize plan
        plan = EscalationPlan(
            buffer_style=self.persona_config.get("buffer_style", "neutral"),
            state=ConversationState.CASUAL_CONTINUATION.value,
        )
        
        # Step 1: Classify conversation state
        state = self._classify_state(transcript_lower, history, context)
        plan.state = state.value
        
        # Step 1.5: Classify sentence type (for filler selection)
        sentence_type = self._classify_sentence_type(transcript_lower)
        plan.sentence_type = sentence_type.value
        
        # Step 2: Try pattern matching
        pattern_result = self._match_patterns(transcript_lower)
        if pattern_result:
            pattern, response, tier, priority = pattern_result
            plan.matched_pattern = pattern
            
            if tier == "deterministic" and response:
                # Direct response available
                if response == "__GREETING__":
                    plan.deterministic_response = random.choice(
                        self.persona_config.get("greeting_responses", ["Hello!"])
                    )
                elif response == "__RECALL_THOUGHT__":
                    # Check for pending thoughts to recall
                    thought = self.recall_thought()
                    if thought:
                        # Queue the thought content for playback
                        plan.deterministic_response = f"Oh right, I was thinking about {thought.get('topic', 'something')}... {thought['content']}"
                        plan.matched_pattern = f"__RECALL_THOUGHT__ (id={thought['id']})"
                    else:
                        # No pending thoughts
                        plan.deterministic_response = random.choice([
                            "Nothing specific on my mind right now.",
                            "Just here, ready to help!",
                            "I'm all ears, what's up?",
                        ])
                elif not response.startswith("__"):
                    plan.deterministic_response = response
                
                if plan.deterministic_response:
                    plan.selected_tiers = [EscalationTier.DETERMINISTIC.value]
                    plan.escalation_score = 5
                    plan.max_wait_seconds = 0.1
                    plan.earcon = random.choice(EARCONS["acknowledge"])
            
            elif tier == "tool_intent":
                # Tool action needed - use interrogative fillers (user is asking for info)
                plan.selected_tiers = [
                    EscalationTier.EARCONS.value,
                    EscalationTier.LOW_PARAM.value,  # Use LLM with tools
                ]
                plan.escalation_score = 30
                plan.max_wait_seconds = 2.0
                plan.earcon = random.choice(EARCONS["thinking"])
                plan.filler = self._select_filler(sentence_type, needs_thinking=True)
                plan.state = ConversationState.TOOL_INTENT.value
            
            elif tier == "mid_tool":
                # Search or mid-complexity tool
                plan.selected_tiers = [
                    EscalationTier.EARCONS.value,
                    EscalationTier.MID_TOOL.value,
                ]
                plan.escalation_score = 50
                plan.max_wait_seconds = 3.0
                plan.earcon = random.choice(EARCONS["thinking"])
                plan.filler = self._select_filler(sentence_type, needs_thinking=True)
            
            elif tier == "heavy":
                # Complex reasoning needed
                plan.selected_tiers = [
                    EscalationTier.EARCONS.value,
                    EscalationTier.HEAVY.value,
                ]
                plan.escalation_score = 80
                plan.max_wait_seconds = 5.0
                plan.async_heavy = True
                plan.earcon = random.choice(EARCONS["thinking"])
                plan.filler = self._select_filler(sentence_type, needs_thinking=True)
        
        # Step 3: If no pattern match, check question type
        if not plan.selected_tiers:
            question_tier = self._detect_question_type(transcript_lower)
            if question_tier:
                plan.selected_tiers = [
                    EscalationTier.EARCONS.value,
                    question_tier,
                ]
                plan.escalation_score = 40
                plan.max_wait_seconds = 2.0
                plan.earcon = random.choice(EARCONS["acknowledge"])
                # Questions need thinking fillers
                plan.filler = self._select_filler(SentenceType.INTERROGATIVE)
        
        # Step 4: If still nothing, use heuristics
        if not plan.selected_tiers:
            # Check history for ongoing task context
            if self._ongoing_task or self._detect_continuation(transcript_lower, history):
                plan.selected_tiers = [EscalationTier.LOW_PARAM.value]
                plan.escalation_score = 25
                plan.max_wait_seconds = 1.5
                # Use sentence-type-appropriate filler
                plan.filler = self._select_filler(sentence_type)
            else:
                # Default to low_param for general conversation
                plan.selected_tiers = [
                    EscalationTier.EARCONS.value,
                    EscalationTier.LOW_PARAM.value,
                ]
                plan.escalation_score = 30
                plan.max_wait_seconds = 2.0
                plan.earcon = random.choice(EARCONS["acknowledge"])
                # Use sentence-type-appropriate filler
                plan.filler = self._select_filler(sentence_type)
        
        # Step 5: Retrieve relevant memories
        plan.memory_context = self._retrieve_memories(transcript_clean, limit=3)
        
        # Step 6: Record state
        self._record_state(state, transcript)
        
        # Performance check
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        if elapsed_ms > 10:
            # Log warning if we exceeded target latency
            pass  # Could log here
        
        return plan
    
    def _classify_state(
        self,
        transcript: str,
        history: List[Dict],
        context: Dict
    ) -> ConversationState:
        """Classify the conversation state using heuristics."""
        
        # Check for introduction patterns
        intro_patterns = [
            r"^(hi|hello|hey|greetings|good (morning|afternoon|evening))",
            r"^who are you",
            r"^what('s| is) your name",
        ]
        for pattern in intro_patterns:
            if re.search(pattern, transcript):
                return ConversationState.INTRODUCTION
        
        # Check for tool intent
        tool_keywords = [
            "timer", "remind", "weather", "time", "date", "search",
            "look up", "set", "cancel", "system", "status"
        ]
        if any(kw in transcript for kw in tool_keywords):
            return ConversationState.TOOL_INTENT
        
        # Check for task focus (based on history)
        if len(history) > 2:
            # If recent history has coherent topic, likely task-focused
            recent_topics = [h.get("content", "")[:50] for h in history[-3:]]
            if self._topics_related(recent_topics):
                return ConversationState.TASK_FOCUSED
        
        # Check for novel/uncertain (long or complex input)
        word_count = len(transcript.split())
        if word_count > 20 or "?" in transcript and word_count > 10:
            return ConversationState.NOVEL_UNCERTAIN
        
        # Check for continuation
        if history and len(transcript.split()) < 5:
            return ConversationState.CASUAL_CONTINUATION
        
        return ConversationState.CASUAL_CONTINUATION
    
    def _classify_sentence_type(self, transcript: str) -> SentenceType:
        """
        Classify the sentence type using deterministic rules.
        
        This runs in <1ms and determines whether the input is:
        - INTERROGATIVE: Questions
        - DECLARATIVE: Statements of fact/opinion
        - IMPERATIVE: Commands/requests
        - EXCLAMATORY: Emotional expressions
        - FRAGMENT: Incomplete phrases
        
        Args:
            transcript: Normalized (lowercase, stripped) transcript
            
        Returns:
            SentenceType enum value
        """
        # Handle empty input
        if not transcript:
            return SentenceType.FRAGMENT
        
        words = transcript.split()
        word_count = len(words)
        first_word = words[0].rstrip("'s") if words else ""
        
        # === FRAGMENT detection ===
        # Very short responses without clear structure
        if word_count == 1:
            # Single-word affirmations/negations
            fragments = {
                "yes", "yeah", "yep", "yup", "no", "nope", "nah",
                "okay", "ok", "sure", "maybe", "probably", "definitely",
                "thanks", "please", "sorry", "what", "huh", "right"
            }
            if first_word in fragments:
                return SentenceType.FRAGMENT
        
        # === EXCLAMATORY detection ===
        # Check for exclamation mark at end
        if transcript.rstrip().endswith("!"):
            # Short exclamations
            if word_count <= 4:
                return SentenceType.EXCLAMATORY
            # Emotion words with exclamation
            emotion_patterns = [
                r"\b(wow|amazing|awesome|incredible|fantastic|terrible|horrible)\b",
                r"\b(oh my|holy|damn|dang|geez|yay|hooray|ugh)\b",
                r"^(yes|no|yeah|nope|what)!*$",
            ]
            for pattern in emotion_patterns:
                if re.search(pattern, transcript):
                    return SentenceType.EXCLAMATORY
        
        # === INTERROGATIVE detection ===
        # Question mark is the strongest signal
        if "?" in transcript:
            return SentenceType.INTERROGATIVE
        
        # Question words at start
        question_starters = {
            "who", "what", "where", "when", "why", "how",
            "which", "whose", "whom", "is", "are", "was", "were",
            "do", "does", "did", "can", "could", "will", "would",
            "should", "shall", "may", "might", "have", "has", "had"
        }
        if first_word in question_starters:
            # Auxiliary verb inversion patterns (questions without ?)
            aux_patterns = [
                r"^(is|are|was|were|do|does|did|can|could|will|would|should|have|has|had)\s+(you|it|he|she|they|we|this|that|there)\b",
                r"^(what|where|when|why|how|who|which)\s+",
            ]
            for pattern in aux_patterns:
                if re.search(pattern, transcript):
                    return SentenceType.INTERROGATIVE
        
        # Tag questions (statement + question tag)
        tag_patterns = [
            r"\b(right|isn't it|aren't you|don't you|won't you|can't you|isn't that)\s*\??$",
            r"\b(do you think|you know)\s*\??$",
        ]
        for pattern in tag_patterns:
            if re.search(pattern, transcript):
                return SentenceType.INTERROGATIVE
        
        # === IMPERATIVE detection ===
        # Commands typically start with a verb (no subject)
        imperative_verbs = {
            # Common command verbs
            "set", "get", "tell", "show", "find", "search", "play", "stop",
            "turn", "open", "close", "start", "end", "create", "delete",
            "make", "give", "take", "put", "add", "remove", "list", "check",
            "look", "call", "send", "read", "write", "save", "load", "run",
            "help", "explain", "describe", "remind", "cancel", "pause", "resume",
            # Polite imperatives
            "please", "let", "try", "go", "come", "wait", "hold",
        }
        
        if first_word in imperative_verbs:
            return SentenceType.IMPERATIVE
        
        # "Please" + verb pattern
        if first_word == "please" and word_count > 1:
            return SentenceType.IMPERATIVE
        
        # "Let me/us" pattern
        if transcript.startswith("let me ") or transcript.startswith("let's ") or transcript.startswith("let us "):
            return SentenceType.IMPERATIVE
        
        # "Can you" / "Could you" / "Would you" as polite imperatives
        polite_imperative_patterns = [
            r"^(can|could|would|will)\s+you\s+\w+",
        ]
        for pattern in polite_imperative_patterns:
            if re.search(pattern, transcript):
                # This is a polite request, treat as imperative
                return SentenceType.IMPERATIVE
        
        # === DECLARATIVE detection ===
        # Subject-verb patterns indicate declarative
        declarative_patterns = [
            # Pronoun + verb
            r"^(i|you|he|she|it|we|they)\s+(am|is|are|was|were|have|has|had|do|does|did|can|could|will|would|like|love|hate|think|believe|want|need|know|feel|see|hear)\b",
            # "My/Your/The" + noun patterns
            r"^(my|your|the|a|an|this|that|these|those)\s+\w+\s+(is|are|was|were|has|have)\b",
            # "There is/are" pattern
            r"^there\s+(is|are|was|were)\b",
            # "It's" / "That's" patterns
            r"^(it's|that's|here's|there's)\b",
        ]
        for pattern in declarative_patterns:
            if re.search(pattern, transcript):
                return SentenceType.DECLARATIVE
        
        # === FALLBACK ===
        # If we can't determine, check word count
        if word_count <= 2:
            return SentenceType.FRAGMENT
        
        # Default to declarative for longer statements
        return SentenceType.DECLARATIVE
    
    def _select_filler(self, sentence_type: SentenceType, needs_thinking: bool = False) -> Optional[str]:
        """
        Select an appropriate filler based on sentence type and persona.
        
        Args:
            sentence_type: The classified sentence type
            needs_thinking: If True, prefer thinking-style fillers even for declaratives
            
        Returns:
            A filler string appropriate for the sentence type, or None
        """
        fillers_config = self.persona_config.get("fillers", {})
        
        # Handle legacy format (flat list of fillers)
        if isinstance(fillers_config, list):
            return random.choice(fillers_config) if fillers_config else None
        
        # Get sentence-type-specific fillers
        type_key = sentence_type.value
        
        # For interrogatives or when thinking is needed, use interrogative fillers
        if needs_thinking and sentence_type == SentenceType.DECLARATIVE:
            type_key = "interrogative"
        
        type_fillers = fillers_config.get(type_key, [])
        
        # Fallback to neutral fillers if no type-specific ones
        if not type_fillers:
            type_fillers = fillers_config.get("neutral", [])
        
        # Final fallback
        if not type_fillers:
            return None
        
        return random.choice(type_fillers)
    
    def _match_patterns(
        self,
        transcript: str
    ) -> Optional[Tuple[str, Optional[str], str, int]]:
        """Match transcript against compiled patterns."""
        best_match = None
        best_priority = -1
        
        for compiled, response, tier, priority in self.compiled_patterns:
            if compiled.search(transcript):
                if priority > best_priority:
                    best_match = (compiled.pattern, response, tier, priority)
                    best_priority = priority
        
        return best_match
    
    def _detect_question_type(self, transcript: str) -> Optional[str]:
        """Detect question type and return suggested tier."""
        words = transcript.split()
        if not words:
            return None
        
        first_word = words[0].lower().rstrip("'s")
        if first_word in QUESTION_TYPES:
            return QUESTION_TYPES[first_word][0]
        
        # Check for embedded question words
        for qword, (tier, _) in QUESTION_TYPES.items():
            if qword in transcript and "?" in transcript:
                return tier
        
        return None
    
    def _detect_continuation(self, transcript: str, history: List[Dict]) -> bool:
        """Detect if this is a continuation of previous topic."""
        if not history:
            return False
        
        continuation_starters = ["and", "also", "but", "so", "then", "what about", "how about"]
        for starter in continuation_starters:
            if transcript.startswith(starter):
                return True
        
        # Short responses are often continuations
        if len(transcript.split()) <= 3 and history:
            return True
        
        return False
    
    def _topics_related(self, topics: List[str]) -> bool:
        """Check if topics share keywords (simple heuristic)."""
        if len(topics) < 2:
            return False
        
        # Extract keywords from each topic
        all_keywords = []
        for topic in topics:
            words = set(self._extract_keywords(topic))
            all_keywords.append(words)
        
        # Check for overlap
        if len(all_keywords) >= 2:
            overlap = all_keywords[0].intersection(all_keywords[1])
            return len(overlap) >= 1
        
        return False
    
    def _preprocess_text(self, text: str) -> str:
        """Lightweight NLP preprocessing."""
        # Lowercase
        text = text.lower()
        
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords by removing stopwords and applying stemming."""
        text = self._preprocess_text(text)
        words = text.split()
        
        # Remove stopwords
        words = [w for w in words if w not in STOPWORDS and len(w) > 2]
        
        # Simple stemming
        stemmed = []
        for word in words:
            for pattern, replacement in SUFFIX_RULES:
                word = re.sub(pattern, replacement, word)
            if len(word) > 2:
                stemmed.append(word)
        
        return stemmed
    
    def _retrieve_memories(self, query: str, limit: int = 3) -> List[Dict]:
        """Retrieve relevant memories using keyword matching."""
        keywords = self._extract_keywords(query)
        if not keywords:
            return []
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build query with keyword matching
        keyword_conditions = " OR ".join([f"content LIKE ?" for _ in keywords])
        params = [f"%{kw}%" for kw in keywords]
        
        try:
            cursor.execute(f"""
                SELECT topic, content, sentiment, timestamp, weight
                FROM memories
                WHERE {keyword_conditions}
                ORDER BY weight DESC, timestamp DESC
                LIMIT ?
            """, params + [limit])
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "topic": row[0],
                    "content": row[1],
                    "sentiment": row[2],
                    "timestamp": row[3],
                    "weight": row[4],
                })
            return results
        except:
            return []
        finally:
            conn.close()
    
    def _record_state(self, state: ConversationState, input_text: str) -> None:
        """Record conversation state for context."""
        self._recent_states.append(state.value)
        if len(self._recent_states) > 10:
            self._recent_states = self._recent_states[-10:]
        
        # Persist to DB (async-safe)
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO states (state, input_text, timestamp, persona)
                VALUES (?, ?, ?, ?)
            """, (state.value, input_text[:200], time.time(), self.persona_name))
            conn.commit()
            conn.close()
        except:
            pass  # Don't let DB errors break the flow
    
    def store_memory(
        self,
        topic: str,
        content: str,
        sentiment: float = 0.0,
        weight: float = 1.0
    ) -> bool:
        """Store a memory for later retrieval."""
        content_hash = hashlib.md5(content.encode()).hexdigest()
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO memories (topic, persona, content, sentiment, timestamp, weight, content_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (topic, self.persona_name, content, sentiment, time.time(), weight, content_hash))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def add_to_corpus(self, sentence: str, source: str = "chat") -> bool:
        """Add a sentence to the corpus for TF-IDF fallback."""
        keywords = " ".join(self._extract_keywords(sentence))
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO corpus (sentence, keywords, source, timestamp)
                VALUES (?, ?, ?, ?)
            """, (sentence, keywords, source, time.time()))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def get_fallback_response(self) -> str:
        """Get a random persona-flavored no-match response."""
        responses = self.persona_config.get("no_match_responses", ["I'm not sure."])
        return random.choice(responses)
    
    # ========================================================================
    # GEOCACHE - Location lookup caching for weather/location tools
    # ========================================================================
    
    def cache_geocode(self, query: str, lat: float, lon: float, 
                      city_name: str, admin1: str = "", country: str = "") -> bool:
        """
        Cache a geocoded location for fast future lookups.
        
        Args:
            query: The original location query (e.g., "Kill Devil Hills, NC")
            lat: Latitude
            lon: Longitude
            city_name: Resolved city name
            admin1: State/province
            country: Country name
            
        Returns:
            True if cached successfully
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Normalize query for consistent matching
            normalized_query = query.lower().strip()
            
            cursor.execute("""
                INSERT OR REPLACE INTO geocache 
                (query, latitude, longitude, city_name, admin1, country, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (normalized_query, lat, lon, city_name, admin1, country, time.time()))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            return False
    
    def get_cached_geocode(self, query: str, max_age_days: int = 30) -> Optional[Dict]:
        """
        Retrieve a cached geocode result.
        
        Args:
            query: Location query to look up
            max_age_days: Maximum age of cached result in days
            
        Returns:
            Dict with lat, lon, city_name, admin1, country if found, else None
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            normalized_query = query.lower().strip()
            max_age_seconds = max_age_days * 24 * 60 * 60
            min_timestamp = time.time() - max_age_seconds
            
            cursor.execute("""
                SELECT latitude, longitude, city_name, admin1, country, timestamp
                FROM geocache
                WHERE query = ? AND timestamp > ?
            """, (normalized_query, min_timestamp))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'latitude': row[0],
                    'longitude': row[1],
                    'city_name': row[2],
                    'admin1': row[3],
                    'country': row[4],
                    'cached_at': row[5]
                }
            return None
        except:
            return None
    
    def get_recent_locations(self, limit: int = 5) -> List[Dict]:
        """Get recently cached locations (for suggestions/autocomplete)."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT query, city_name, admin1, country, timestamp
                FROM geocache
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    'query': row[0],
                    'city_name': row[1],
                    'admin1': row[2],
                    'country': row[3],
                    'timestamp': row[4]
                })
            
            conn.close()
            return results
        except:
            return []
    
    # ========================================================================
    # OUTPUT QUEUE - Chunked TTS playback management
    # ========================================================================
    
    def queue_output(self, text: str, session_id: str = None) -> List[int]:
        """
        Chunk text and queue it for TTS playback.
        
        Args:
            text: The full text to chunk and queue
            session_id: Optional session identifier
            
        Returns:
            List of chunk IDs that were queued
        """
        chunks = chunk_text_for_speech(text)
        if not chunks:
            return []
        
        chunk_ids = []
        now = time.time()
        
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Clear any pending chunks for this persona first
            cursor.execute("""
                UPDATE output_queue SET status = 'abandoned'
                WHERE persona = ? AND status = 'pending'
            """, (self.persona_name,))
            
            # Insert new chunks
            for seq, chunk in enumerate(chunks, 1):
                cursor.execute("""
                    INSERT INTO output_queue (persona, session_id, sequence, chunk_text, status, created_at)
                    VALUES (?, ?, ?, ?, 'pending', ?)
                """, (self.persona_name, session_id, seq, chunk, now))
                chunk_ids.append(cursor.lastrowid)
            
            conn.commit()
            conn.close()
            return chunk_ids
        except Exception as e:
            return []
    
    def pop_next_chunk(self) -> Optional[Dict]:
        """
        Get the next pending chunk for playback.
        
        Returns:
            Dict with chunk info, or None if queue is empty
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, chunk_text, sequence, session_id
                FROM output_queue
                WHERE persona = ? AND status = 'pending'
                ORDER BY sequence ASC
                LIMIT 1
            """, (self.persona_name,))
            
            row = cursor.fetchone()
            if row:
                # Mark as playing
                cursor.execute("""
                    UPDATE output_queue SET status = 'playing', played_at = ?
                    WHERE id = ?
                """, (time.time(), row[0]))
                conn.commit()
                conn.close()
                
                return {
                    'id': row[0],
                    'text': row[1],
                    'sequence': row[2],
                    'session_id': row[3]
                }
            
            conn.close()
            return None
        except:
            return None
    
    def mark_chunk_completed(self, chunk_id: int) -> bool:
        """Mark a chunk as completed after playback."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE output_queue SET status = 'completed'
                WHERE id = ?
            """, (chunk_id,))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    def interrupt_queue(self, topic: str = None, ttl_hours: float = 24) -> Optional[int]:
        """
        Interrupt the queue and move remaining chunks to thoughts.
        
        Args:
            topic: Optional topic for the interrupted content
            ttl_hours: How long to keep the thought (default 24 hours)
            
        Returns:
            Thought ID if content was saved, None otherwise
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Get all pending chunks
            cursor.execute("""
                SELECT chunk_text FROM output_queue
                WHERE persona = ? AND status = 'pending'
                ORDER BY sequence ASC
            """, (self.persona_name,))
            
            pending_chunks = [row[0] for row in cursor.fetchall()]
            
            if not pending_chunks:
                conn.close()
                return None
            
            # Combine into single thought
            remaining_content = " ".join(pending_chunks)
            
            # Mark chunks as interrupted
            cursor.execute("""
                UPDATE output_queue SET status = 'interrupted'
                WHERE persona = ? AND status = 'pending'
            """, (self.persona_name,))
            
            # Store as thought
            now = time.time()
            expires_at = now + (ttl_hours * 3600)
            
            cursor.execute("""
                INSERT INTO thoughts (persona, topic, content, source, priority, created_at, expires_at)
                VALUES (?, ?, ?, 'interrupted', 0.7, ?, ?)
            """, (self.persona_name, topic, remaining_content, now, expires_at))
            
            thought_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return thought_id
        except:
            return None
    
    def get_queue_status(self) -> Dict:
        """Get the current queue status."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT status, COUNT(*) FROM output_queue
                WHERE persona = ?
                GROUP BY status
            """, (self.persona_name,))
            
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            cursor.execute("""
                SELECT COUNT(*) FROM output_queue
                WHERE persona = ? AND status = 'pending'
            """, (self.persona_name,))
            
            pending = cursor.fetchone()[0]
            conn.close()
            
            return {
                'pending': pending,
                'has_pending': pending > 0,
                'counts': status_counts
            }
        except:
            return {'pending': 0, 'has_pending': False, 'counts': {}}
    
    def clear_queue(self) -> bool:
        """Clear all pending chunks from the queue."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM output_queue
                WHERE persona = ? AND status = 'pending'
            """, (self.persona_name,))
            conn.commit()
            conn.close()
            return True
        except:
            return False
    
    # ========================================================================
    # THOUGHTS - Store and recall interrupted/deferred content
    # ========================================================================
    
    def store_thought(
        self,
        content: str,
        topic: str = None,
        source: str = "manual",
        priority: float = 0.5,
        ttl_hours: float = 24
    ) -> Optional[int]:
        """
        Store a thought for future recall.
        
        Args:
            content: The thought content
            topic: Optional topic/keywords
            source: Source type ("interrupted", "research", "dreaming", "manual")
            priority: Relevance score 0-1 (higher = more likely to recall)
            ttl_hours: Hours until expiration
            
        Returns:
            Thought ID if stored, None on error
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            now = time.time()
            expires_at = now + (ttl_hours * 3600)
            
            cursor.execute("""
                INSERT INTO thoughts (persona, topic, content, source, priority, created_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (self.persona_name, topic, content, source, priority, now, expires_at))
            
            thought_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return thought_id
        except:
            return None
    
    def recall_thought(self) -> Optional[Dict]:
        """
        Get the highest priority unexpired thought.
        
        Returns:
            Dict with thought info, or None if no thoughts available
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            now = time.time()
            
            cursor.execute("""
                SELECT id, topic, content, source, priority, created_at
                FROM thoughts
                WHERE persona = ? AND recalled = 0 AND expires_at > ?
                ORDER BY priority DESC, created_at DESC
                LIMIT 1
            """, (self.persona_name, now))
            
            row = cursor.fetchone()
            if row:
                # Mark as recalled
                cursor.execute("""
                    UPDATE thoughts SET recalled = 1 WHERE id = ?
                """, (row[0],))
                conn.commit()
                conn.close()
                
                return {
                    'id': row[0],
                    'topic': row[1],
                    'content': row[2],
                    'source': row[3],
                    'priority': row[4],
                    'created_at': row[5]
                }
            
            conn.close()
            return None
        except:
            return None
    
    def check_thought_triggers(self, transcript: str) -> bool:
        """
        Check if the transcript matches thought recall trigger patterns.
        
        Args:
            transcript: The user's input
            
        Returns:
            True if this should trigger thought recall
        """
        transcript_lower = transcript.lower().strip()
        
        for pattern in THOUGHT_RECALL_TRIGGERS:
            if re.match(pattern, transcript_lower, re.IGNORECASE):
                return True
        
        return False
    
    def has_pending_thoughts(self) -> bool:
        """Check if there are any unexpired, unrecalled thoughts."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            now = time.time()
            cursor.execute("""
                SELECT COUNT(*) FROM thoughts
                WHERE persona = ? AND recalled = 0 AND expires_at > ?
            """, (self.persona_name, now))
            
            count = cursor.fetchone()[0]
            conn.close()
            return count > 0
        except:
            return False
    
    def expire_old_thoughts(self) -> int:
        """Delete expired thoughts. Returns count of deleted rows."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            now = time.time()
            cursor.execute("""
                DELETE FROM thoughts WHERE expires_at < ?
            """, (now,))

            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            return deleted
        except:
            return 0

    # ========================================================================
    # TOOL INSIGHTS (Universal Tool Handler RAG)
    # ========================================================================

    def store_tool_insight(
        self,
        query: str,
        tool_used: str,
        tool_args: dict = None,
        result_summary: str = None,
        full_response: str = None,
        keywords: List[str] = None
    ) -> Optional[int]:
        """
        Store a tool call result for future RAG retrieval.

        Args:
            query: The user's original query that triggered the tool
            tool_used: Name of the tool that was called
            tool_args: Arguments passed to the tool (as dict)
            result_summary: Short summary of the result
            full_response: Full formatted response
            keywords: Keywords for matching future queries

        Returns:
            Insight ID if stored, None on error
        """
        try:
            import json
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            now = time.time()
            args_json = json.dumps(tool_args) if tool_args else None
            keywords_str = ",".join(keywords) if keywords else self._extract_keywords(query)

            cursor.execute("""
                INSERT INTO tool_insights 
                (persona, query, tool_used, tool_args, result_summary, full_response, keywords, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (self.persona_name, query, tool_used, args_json, result_summary, full_response, keywords_str, now))

            insight_id = cursor.lastrowid
            conn.commit()
            conn.close()
            return insight_id
        except Exception as e:
            return None

    def retrieve_tool_insights(self, query: str, max_results: int = 3) -> List[Dict]:
        """
        Retrieve relevant tool insights based on keyword matching.

        Args:
            query: Current user query to match against
            max_results: Maximum number of insights to return

        Returns:
            List of matching insights sorted by relevance
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            # Extract keywords from query
            query_keywords = set(self._extract_keywords(query).lower().split(","))
            
            if not query_keywords:
                conn.close()
                return []

            # Get all insights for this persona
            cursor.execute("""
                SELECT id, query, tool_used, tool_args, result_summary, full_response, keywords, created_at, used_count
                FROM tool_insights
                WHERE persona = ?
                ORDER BY created_at DESC
                LIMIT 100
            """, (self.persona_name,))

            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return []

            # Score by keyword overlap
            scored = []
            for row in rows:
                insight_keywords = set(row[6].lower().split(",")) if row[6] else set()
                overlap = len(query_keywords & insight_keywords)
                if overlap > 0:
                    scored.append((overlap, {
                        'id': row[0],
                        'query': row[1],
                        'tool_used': row[2],
                        'tool_args': row[3],
                        'result_summary': row[4],
                        'full_response': row[5],
                        'keywords': row[6],
                        'created_at': row[7],
                        'used_count': row[8],
                        'relevance_score': overlap
                    }))

            # Sort by overlap score (descending), return top results
            scored.sort(key=lambda x: x[0], reverse=True)
            return [item[1] for item in scored[:max_results]]

        except Exception as e:
            return []

    def increment_insight_usage(self, insight_id: int) -> bool:
        """Increment the used_count for an insight."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE tool_insights SET used_count = used_count + 1 WHERE id = ?
            """, (insight_id,))
            conn.commit()
            conn.close()
            return True
        except:
            return False

    def _extract_keywords(self, text: str) -> str:
        """Extract keywords from text for storage/matching."""
        # Use existing stopwords and simple extraction
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        # Filter stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'can',
                     'what', 'which', 'who', 'whom', 'this', 'that', 'these',
                     'those', 'and', 'but', 'or', 'nor', 'for', 'yet', 'so',
                     'in', 'on', 'at', 'to', 'from', 'of', 'with', 'about'}
        keywords = [w for w in words if w not in stopwords]
        return ",".join(keywords[:10])  # Limit to 10 keywords

    # ========================================================================
    # INTERJECTION CLASSIFICATION
    # ========================================================================
    
    def classify_interjection(self, transcript: str) -> str:
        """
        Classify whether an interjection is short (pause queue) or long (abandon).
        
        Short interjections: affirmations, continuations, backchannel
        Long interjections: new questions, new topics, commands
        
        Args:
            transcript: The user's speech during AI playback
            
        Returns:
            "short" if queue should pause, "long" if queue should be abandoned
        """
        transcript_lower = transcript.lower().strip()
        words = transcript_lower.split()
        word_count = len(words)
        
        # Very short utterances are usually backchannel
        if word_count <= 2:
            # Check if it's a known short interjection
            if transcript_lower in SHORT_INTERJECTIONS:
                return "short"
            # Single words are usually short
            if word_count == 1:
                return "short"
        
        # Check for short interjection phrases
        for phrase in SHORT_INTERJECTIONS:
            if transcript_lower == phrase or transcript_lower.startswith(phrase + " "):
                return "short"
        
        # If it's a question, it's a long interjection (new topic)
        if "?" in transcript or self._classify_sentence_type(transcript_lower) == SentenceType.INTERROGATIVE:
            return "long"
        
        # If it contains question words, likely a new topic
        question_words = {"what", "where", "when", "why", "how", "who", "which"}
        if any(w in words for w in question_words):
            return "long"
        
        # If it's a command/imperative, likely a new request
        if self._classify_sentence_type(transcript_lower) == SentenceType.IMPERATIVE:
            return "long"
        
        # Longer statements (4+ words) are usually new topics
        if word_count >= 4:
            return "long"
        
        # Default to short for ambiguous cases
        return "short"


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_sentence_type_classification():
    """Test the sentence type classification with examples."""
    import tempfile
    
    print("=" * 70)
    print("Testing Sentence Type Classification")
    print("=" * 70)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    flair = MemoryFlair(db_path=db_path, persona="neutral")
    
    # Test cases organized by expected type
    test_cases = [
        # (transcript, expected_type)
        # INTERROGATIVE
        ("What time is it?", "interrogative"),
        ("How much does China weigh?", "interrogative"),
        ("Who was the first president?", "interrogative"),
        ("Where is the nearest coffee shop", "interrogative"),
        ("Why is the sky blue", "interrogative"),
        ("Is it raining outside?", "interrogative"),
        ("Can you help me with this?", "interrogative"),
        ("Do you know the answer", "interrogative"),
        ("You like pizza, right?", "interrogative"),
        
        # DECLARATIVE
        ("I like pizza", "declarative"),
        ("My favorite color is blue", "declarative"),
        ("The weather is nice today", "declarative"),
        ("I think we should go outside", "declarative"),
        ("That's a great idea", "declarative"),
        ("It's raining outside", "declarative"),
        ("I have three cats", "declarative"),
        
        # IMPERATIVE
        ("Set a timer for 5 minutes", "imperative"),
        ("Tell me a joke", "imperative"),
        ("Search for nearby restaurants", "imperative"),
        ("Please turn on the lights", "imperative"),
        ("Let me know when you're done", "imperative"),
        ("Stop the music", "imperative"),
        ("Can you set a reminder", "imperative"),
        ("Would you play some music", "imperative"),
        
        # EXCLAMATORY
        ("Wow!", "exclamatory"),
        ("That's amazing!", "exclamatory"),
        ("Oh my god!", "exclamatory"),
        ("No way!", "exclamatory"),
        
        # FRAGMENT
        ("yes", "fragment"),
        ("okay", "fragment"),
        ("maybe", "fragment"),
        ("thanks", "fragment"),
    ]
    
    correct = 0
    total = len(test_cases)
    
    for transcript, expected in test_cases:
        result = flair._classify_sentence_type(transcript.lower())
        status = "✓" if result.value == expected else "✗"
        if result.value == expected:
            correct += 1
        print(f"  {status} '{transcript}' → {result.value} (expected: {expected})")
    
    accuracy = (correct / total) * 100
    print(f"\nAccuracy: {correct}/{total} ({accuracy:.1f}%)")
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)
    
    return correct == total


def test_memory_flair():
    """Test the MemoryFlair module with sample transcripts."""
    import tempfile
    
    print("\n" + "=" * 70)
    print("Testing MemoryFlair Decision Engine")
    print("=" * 70)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # Test with different personas
    for persona in ["flirty", "professional", "snarky"]:
        print(f"\n--- Persona: {persona} ---")
        flair = MemoryFlair(db_path=db_path, persona=persona)
        
        # Test cases with expected sentence types
        test_cases = [
            # (transcript, description)
            ("Hello!", "greeting/exclamatory"),
            ("What time is it?", "question/tool"),
            ("Set a timer for 5 minutes", "imperative/tool"),
            ("I like pizza", "declarative statement"),
            ("What's the weather like in Seattle?", "question/tool"),
            ("Think deeply about the meaning of life", "imperative/heavy"),
            ("Who was the first president?", "question/search"),
            ("Tell me a joke", "imperative"),
            ("That's really cool!", "exclamatory"),
            ("and then what happened?", "question/continuation"),
            ("My favorite food is tacos", "declarative"),
            ("okay", "fragment/affirmation"),
        ]
        
        for transcript, description in test_cases:
            start = time.perf_counter()
            plan = flair.decide(transcript)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            print(f"\n  Input: '{transcript}' ({description})")
            print(f"  Sentence Type: {plan.sentence_type}")
            print(f"  State: {plan.state}")
            print(f"  Tiers: {plan.selected_tiers}")
            print(f"  Filler: {plan.filler}")
            if plan.deterministic_response:
                print(f"  Response: {plan.deterministic_response}")
            print(f"  Time: {elapsed_ms:.2f}ms {'✓' if elapsed_ms < 10 else '⚠️ SLOW'}")
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)
    
    print("\n" + "=" * 70)
    print("Tests complete!")
    print("=" * 70)


if __name__ == "__main__":
    # Run sentence type classification tests first
    test_sentence_type_classification()
    
    # Then run the full decision engine tests
    test_memory_flair()

