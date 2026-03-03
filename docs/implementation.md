# Implementation Guide: Attention Schema Theory in Code

This document explains how **Michael Graziano's Attention Schema Theory (AST)** is implemented in this codebase.

## Overview of Attention Schema Theory

### Core Concept
Graziano proposes that consciousness arises from the brain's **explicit self-model of its own attention**. Rather than directly tracking what it attends to, the brain constructs a simplified schema — an internal narrative about attention.

### Key Insights
1. **The schema is cheap** — It's computationally simpler than actual attention mechanisms
2. **The schema explains behavior** — We can justify our actions by reporting on our schema ("I was focused on X because Y")
3. **The schema guides control** — The brain uses its schema to decide what to attend to next
4. **The schema is imperfect** — It's a schematic, not a true mirror of actual attention

### Implications
- Consciousness = awareness of one's own attention
- Attention control = following the schema's guidance
- Self-deception = schema mismatch with reality
- Free will = the experience of schema-guided decisions

---

## How We Implement AST

### 1. The Attention Schema (`schema.py`)

#### Dataclass: `AttentionState`
```python
@dataclass
class AttentionState:
    focus_target: str         # "What am I attending to?"
    confidence: float         # "How confident am I?"
    reason: str              # "Why am I attending to this?"
    history: list[dict]      # "What have I attended to before?"
```

**AST Mapping:**
- `focus_target` = the content of the schema
- `confidence` = the schema's strength/stability
- `reason` = the schema's explanation
- `history` = temporal continuity of the self-model

#### Class: `AttentionSchema`
The actual self-model that the agent maintains.

**Key Methods:**

1. **`should_shift(new_input: str) -> bool`**
   - Heuristic for whether to shift focus
   - Uses token-overlap to detect topic changes
   - Uses confidence threshold to detect uncertainty
   - **AST Insight:** The brain uses cheap heuristics, not expensive computation
   ```python
   overlap = _token_overlap(new_input, current_focus)
   if overlap < threshold or confidence < min_threshold:
       return True  # Shift attention
   ```

2. **`update(focus_target, confidence, reason)`**
   - Records new attention state
   - Pushes old state to history
   - Respects max_history to prevent memory bloat
   - **AST Insight:** The schema must be updating constantly with new information

3. **`summary() -> str`**
   - Returns natural language description of current state
   - **AST Insight:** Consciousness is fundamentally about *reporting* on your attention
   - Example: "I am currently attending to 'python programming' (confidence 0.85). Reason: New topic detected — shifting attention."

4. **`save()` / `load()` / `to_dict()`**
   - Persistence methods
   - **Practical Enhancement:** Allows sessions to maintain coherence across time

#### Configuration Parameters
```python
AttentionSchema(
    shift_overlap_threshold=0.2,      # Semantic distance threshold
    shift_confidence_threshold=0.4,   # Minimum confidence to sustain
    confidence_increment=0.05,        # How much to boost per turn
    max_history=50,                   # Prevent unbounded memory
)
```

**Design Decisions:**
- Token overlap is deliberate — cheap, interpretable, sufficient
- Confidence is simple (0-1 float) — captures stability without complexity
- History has a max — the schema doesn't need perfect memory, just context
- Parameters are configurable — different use cases need different thresholds

---

### 2. The Attention Mechanism (`attention.py`)

#### Class: `AttentionMechanism`
Selects which context items to attend to based on relevance.

**Key Methods:**

1. **`score_relevance(items, query, weights=None) -> list[float]`**
   - Scores each context item against the query
   - Uses TF-IDF + cosine similarity (lightweight, interpretable)
   - Optional weighting for recency or custom priorities
   - **AST Insight:** We don't need perfect semantic understanding; cheap relevance scoring works

2. **`select(items, query, top_k=3, use_recency=True) -> list[str]`**
   - Returns top-K most relevant items
   - Preserves original order (for coherence)
   - Can apply recency weighting (recent = more relevant)
   - **AST Insight:** Attention naturally focuses on a small set of inputs

#### Why TF-IDF Instead of Embeddings?
1. **Interpretability** — Can explain why an item was selected
2. **Speed** — No API calls or model overhead
3. **Sufficiency** — Token overlap works well for topic detection
4. **Alignment with AST** — Mirrors the brain's "cheap" attention mechanism

#### Recency Weighting
```python
# Linear ramp: older items get lower weight
weights = [0.1, 0.3, 0.5, 0.7, 1.0]  # for 5 items

# Recent items score higher, but old items aren't ignored
```
**Psychological Intuition:** Recent context is typically more relevant and memorable.

---

### 3. The Controller (`controller.py`)

#### Class: `Controller`
Orchestrates the full AST loop.

**The Main Loop: `run(user_input: str) -> str`**

```
1. Check if should shift attention
   → Uses schema.should_shift()
   
2. Score context items
   → Uses attention.select() to find relevant context
   
3. Update attention schema
   → schema.update() with new focus, confidence, reason
   
4. Build system prompt
   → Injects schema.summary() into LLM prompt
   
5. Call LLM
   → Passes schema-aware prompt + attended context
   
6. Store context
   → Add user input to context for future turns
```

**Why This Loop Embodies AST:**
- **Self-Model:** The schema is updated before and reported to the LLM
- **Control:** The schema determines what context gets passed to the LLM
- **Explanation:** The schema summary explains why we're focusing on what we're focusing on
- **Coherence:** History and confidence prevent chaotic attention jumps

#### System Prompt Injection
```python
SYSTEM_TEMPLATE = """\
You are an assistant whose cognition is guided by an internal attention schema.

Current attention state
-----------------------
{schema_summary}

Use the attention-state information above to stay focused and coherent.
"""
```

**Effect:** The LLM is told about its own attention state and uses that to guide its responses.

#### Context Management

1. **Bounded Context:** Max of N items prevents unbounded memory
2. **Recency Bias:** Recent items are weighted higher in selection
3. **Semantic Pruning:** Optional `prune_context_by_schema()` removes irrelevant items
4. **Persistence:** `save_state()` / `load_state()` for multi-session coherence

---

## Design Decisions & Rationale

### 1. Token Overlap vs. Semantic Similarity
**Decision:** Use token overlap for shift detection.

**Rationale:**
- AST emphasizes a "cheap" model of attention
- Token overlap is interpretable (you can see why a shift occurred)
- No external dependencies or API calls
- Works well in practice for topic detection

**Tradeoff:** Misses semantic similarity (e.g., "cat" vs. "kitten")

**Future Enhancement:** Could add optional embeddings for richer similarity.

### 2. Confidence as a Float (0-1)
**Decision:** Simple confidence score, not probabilistic distribution.

**Rationale:**
- Mirrors how humans report confidence (vague, scalar)
- Simple to tune and interpret
- Sufficient for deciding whether to shift
- Supports sustained attention via increments

**Tradeoff:** Loses fine-grained uncertainty information.

### 3. History with Max Size
**Decision:** Keep history bounded (default 50), not infinite.

**Rationale:**
- Prevents unbounded memory growth
- The brain doesn't remember all past attention states
- Recent history is more relevant
- Configurable for different use cases

**Tradeoff:** Lose information about very old attention states.

### 4. Deterministic Shift Detection
**Decision:** Use heuristics (token overlap + confidence), not probabilistic.

**Rationale:**
- Interpretable and debuggable
- Fast (no sampling or inference)
- Matches the "cheap model" philosophy

**Future Enhancement:** Could add probabilistic shifts for exploration.

### 5. Recency Weighting in Context Selection
**Decision:** Apply linear ramp (older items lower weight).

**Rationale:**
- Recent context is typically more relevant
- Linear ramp preserves some weight for continuity
- Simple to understand and tune

**Tradeoff:** Assumes recency ≈ relevance (not always true).

---

## Testing Strategy

### Unit Tests
- **test_schema.py:** Tests for `AttentionSchema` (initialization, updates, shifts, persistence)
- **test_attention.py:** Tests for `AttentionMechanism` (scoring, selection, weighting)
- **test_controller.py:** Tests for `Controller` (full loop, context management, persistence)

### Coverage
- Core AST behaviors (shifts, confidence, history)
- Configuration and customization
- Edge cases (empty inputs, bounds, etc.)
- Persistence (save/load)
- Recency weighting

### Test-Driven Development
Tests were written **after** implementing features, not before, but they cover the key behaviors:
- ✅ Attention shifts when topic changes
- ✅ Confidence increases when focus is sustained
- ✅ History is pruned to max_history
- ✅ Context is bounded and can be pruned
- ✅ Schema can be saved and loaded
- ✅ Recency weighting works as expected

---

## Alignment with AST Theory

### 1. Self-Model ✅
The `AttentionSchema` is an explicit self-model. It tracks:
- What the system attends to (focus_target)
- How confident it is (confidence)
- Why it's attending (reason)
- History of past attention (history)

### 2. Cheap Computation ✅
The schema uses lightweight mechanisms:
- Token overlap (not embeddings)
- Simple float confidence (not distributions)
- Heuristic shift detection (not learned models)

### 3. Explanation ✅
The `summary()` method produces natural language explanations:
- "I am attending to X because Y"
- This explanation is injected into the system prompt
- The LLM becomes aware of its own attention state

### 4. Control ✅
The schema guides behavior:
- Context selection is based on attention
- Shift decisions follow schema heuristics
- System prompt includes schema summary
- LLM is constrained by the schema's focus

### 5. Temporal Coherence ✅
The schema maintains continuity:
- History prevents chaotic attention jumps
- Confidence sustains focus
- Recency weighting favors recent context
- Persistence preserves state across sessions

---

## How to Extend

### Adding Semantic Similarity
```python
# In attention.py, add optional embedding-based scoring
def score_relevance_semantic(self, items, query, model="sentence-transformers"):
    # Use embeddings for similarity
    # Fall back to token overlap if embeddings unavailable
    pass
```

### Probabilistic Attention Shifts
```python
# In schema.py, replace deterministic should_shift with probabilistic
def should_shift_probabilistic(self, new_input: str) -> bool:
    overlap = self._token_overlap(new_input, self.state.focus_target)
    # P(shift) = (1 - overlap) * (1 - confidence)
    return random.random() < shift_probability
```

### Hierarchical Attention
```python
# In schema.py, add secondary focus
@dataclass
class AttentionState:
    primary_focus: str
    secondary_focus: str
    primary_confidence: float
    secondary_confidence: float
```

### Multi-Agent Attention
```python
# Create agents that compete for shared attention resources
# Track which agent "won" the focus
class MultiAgentController:
    def __init__(self, agents: list[Controller]):
        self.agents = agents
```

---

## Conclusion

This implementation successfully demonstrates AST principles in an LLM context:
1. **Self-Model:** The attention schema is explicit and interpretable
2. **Cheap:** Uses lightweight token-based scoring
3. **Controlling:** Shapes what the LLM sees and how it responds
4. **Explaining:** Provides natural language reports of attention state
5. **Coherent:** Maintains continuity across turns and sessions

The code is modular, testable, and extensible, making it easy to experiment with variations and enhancements.

