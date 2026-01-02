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

DEFAULT_PERSONAS: Dict[str, Dict[str, Any]] = {
    "flirty": {
        "buffer_style": "flirty",
        "escalation_threshold": 40,  # Lower = more likely to escalate
        "fillers": ["mmhmm...", "oh?", "well...", "hmm, let me think...", "ooh..."],
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
        "fillers": ["one moment...", "let me check...", "processing...", "understood..."],
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
        "fillers": ["uh...", "um...", "hmm...", "let me see...", "okay..."],
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
        "fillers": ["ugh, fine...", "okay okay...", "hold on...", "yeah yeah..."],
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
        
        # Create indices for fast lookup
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_topic ON memories(topic)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_persona ON memories(persona)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_states_timestamp ON states(timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_corpus_keywords ON corpus(keywords)")
        
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
                elif not response.startswith("__"):
                    plan.deterministic_response = response
                
                if plan.deterministic_response:
                    plan.selected_tiers = [EscalationTier.DETERMINISTIC.value]
                    plan.escalation_score = 5
                    plan.max_wait_seconds = 0.1
                    plan.earcon = random.choice(EARCONS["acknowledge"])
            
            elif tier == "tool_intent":
                # Tool action needed
                plan.selected_tiers = [
                    EscalationTier.EARCONS.value,
                    EscalationTier.LOW_PARAM.value,  # Use LLM with tools
                ]
                plan.escalation_score = 30
                plan.max_wait_seconds = 2.0
                plan.earcon = random.choice(EARCONS["thinking"])
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
                plan.filler = random.choice(self.persona_config.get("fillers", ["hmm..."]))
            
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
                plan.filler = random.choice(self.persona_config.get("fillers", ["let me think..."]))
        
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
        
        # Step 4: If still nothing, use heuristics
        if not plan.selected_tiers:
            # Check history for ongoing task context
            if self._ongoing_task or self._detect_continuation(transcript_lower, history):
                plan.selected_tiers = [EscalationTier.LOW_PARAM.value]
                plan.escalation_score = 25
                plan.max_wait_seconds = 1.5
            else:
                # Default to low_param for general conversation
                plan.selected_tiers = [
                    EscalationTier.EARCONS.value,
                    EscalationTier.LOW_PARAM.value,
                ]
                plan.escalation_score = 30
                plan.max_wait_seconds = 2.0
                plan.earcon = random.choice(EARCONS["acknowledge"])
        
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


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_memory_flair():
    """Test the MemoryFlair module with sample transcripts."""
    import tempfile
    
    print("=" * 60)
    print("Testing MemoryFlair")
    print("=" * 60)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    
    # Test with different personas
    for persona in ["flirty", "professional", "snarky"]:
        print(f"\n--- Persona: {persona} ---")
        flair = MemoryFlair(db_path=db_path, persona=persona)
        
        test_cases = [
            "Hello!",
            "What time is it?",
            "Set a timer for 5 minutes",
            "What's the weather like in Seattle?",
            "Think deeply about the meaning of life",
            "Who was the first president?",
            "Tell me a joke",
            "and then what happened?",
            "Search Twitter for trending topics",
            "asdfghjkl qwerty random gibberish",
        ]
        
        for transcript in test_cases:
            start = time.perf_counter()
            plan = flair.decide(transcript)
            elapsed_ms = (time.perf_counter() - start) * 1000
            
            print(f"\n  Input: '{transcript}'")
            print(f"  State: {plan.state}")
            print(f"  Tiers: {plan.selected_tiers}")
            print(f"  Score: {plan.escalation_score}")
            print(f"  Earcon: {plan.earcon}")
            print(f"  Filler: {plan.filler}")
            if plan.deterministic_response:
                print(f"  Response: {plan.deterministic_response}")
            print(f"  Time: {elapsed_ms:.2f}ms {'✓' if elapsed_ms < 10 else '⚠️ SLOW'}")
    
    # Cleanup
    Path(db_path).unlink(missing_ok=True)
    
    print("\n" + "=" * 60)
    print("Tests complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_memory_flair()

