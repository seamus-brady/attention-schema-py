# Attention Schema Theory (Graziano) - Implementation Review

## Overview
This app translates Michael Graziano's **Attention Schema Theory (AST)** into a working LLM agent. AST proposes that the brain constructs a simplified internal model of its own attention—an "attention schema"—which it uses to monitor, control, and explain what it attends to.

## Core Components Analysis

### 1. **AttentionSchema** (schema.py) ✅ **Strong**
The schema captures the essence of AST: a self-model tracking what the system attends to.

**Strengths:**
- Clean dataclass for `AttentionState` (focus_target, confidence, reason)
- History tracking allows temporal awareness of attention shifts
- Token-overlap heuristic for detecting topic changes is lightweight and interpretable
- `summary()` method provides natural-language self-report (aligns with AST's emphasis on *explanation*)

**Opportunities:**
- Confidence metric is binary (shifts reset to 0.8; sustained increment by 0.05). Could be more nuanced.
- `should_shift()` thresholds (0.2 token overlap, 0.4 confidence) are hardcoded. Could be configurable or adaptive.
- No mechanism to weight schema history (all entries equal). Recent context should matter more.

### 2. **AttentionMechanism** (attention.py) ✅ **Solid**
Implements lightweight context scoring without external dependencies.

**Strengths:**
- TF-IDF + cosine similarity is interpretable and efficient
- No heavy embeddings or API calls needed
- Normalizes term frequency, so longer texts don't dominate
- Preserves original order among ties (stable selection)

**Opportunities:**
- Token-level matching misses semantic similarity (e.g., "cat" and "kitten" score as unrelated)
- Could benefit from word-stemming (Porter stemmer) or synonymy awareness
- `top_k=3` is hardcoded in `Controller`; could be dynamic based on confidence or available items

### 3. **Controller** (controller.py) ✅ **Good Integration**
Orchestrates the AST loop: score → shift decision → schema update → LLM call.

**Strengths:**
- Clean separation of concerns (schema decision, context selection, LLM integration)
- System prompt injects schema state, grounding LLM in agent's own self-model
- Accumulates context items for multi-turn coherence
- Interactive REPL loop is user-friendly

**Opportunities:**
- Context accumulation is unbounded—will grow indefinitely (memory issue)
- No persistence of schema between sessions
- `shifting` logic is deterministic; could benefit from confidence-weighted probabilistic shifts
- System prompt template is hardcoded; could be more customizable

### 4. **LLM Integration** (llm.py) ✅ **Clean**
Thin wrapper around Anthropic API with mock for testing.

**Strengths:**
- Protocol-based design allows easy swapping of implementations
- MockLLMClient enables testability
- Captures both system and user messages

**Opportunities:**
- No error handling or retries
- No token counting (could overflow on large context)
- Single model hardcoded; could support model selection

### 5. **Test Coverage** (test_controller.py) ✅ **Adequate**
Tests cover key AST behaviors.

**Strengths:**
- Tests attention shifts on topic changes
- Validates schema history growth
- Checks that context items accumulate
- Uses mock LLM for deterministic testing

**Opportunities:**
- No tests for `AttentionSchema.should_shift()` edge cases
- No tests for `AttentionMechanism` scoring
- Missing tests for schema summary quality
- No integration tests with real LLM (optional but valuable)

---

## Key Insights: How This Embodies AST

1. **Self-Model Over Direct Control**: Instead of the LLM directly deciding what to focus on, it operates *within* the constraints of the attention schema. The schema is not just a log—it actively shapes the system prompt.

2. **Explanation as Central**: Graziano emphasizes that consciousness (or attention schema) is fundamentally about *explaining* what the system is doing. The schema summary in the system prompt serves this role: "I am attending to X because Y."

3. **Temporal Coherence**: By maintaining a history and confidence score, the schema provides continuity across turns. This prevents the agent from chaotically jumping between topics.

4. **Cheap Computational Model**: Following AST, this implementation uses a simplified attention model (token overlap, confidence scores) rather than expensive embeddings or reasoning. This mirrors how the brain maintains a "cheap" model of attention.

---

## Recommendations for Enhancement

### High Priority
1. **Bounded Context**: Implement context pruning (keep last N items or use schema to prune irrelevant items)
2. **Schema Persistence**: Save/load schema state across sessions
3. **Enhanced Shift Detection**: Use semantic similarity or embeddings for more accurate topic detection
4. **Configurable Parameters**: Move hardcoded thresholds to config or constructor

### Medium Priority
1. **Confidence Model**: Replace binary shifts with probabilistic attention shifts
2. **Context Weighting**: Discount older context items in selection
3. **Rich Attention Metadata**: Track what caused each shift (user explicit, topic change, low confidence)
4. **Better Error Handling**: Graceful fallbacks for LLM failures

### Low Priority (Nice-to-Have)
1. **Visualization**: Graph attention shifts over time
2. **Multi-Agent**: Extend to multiple agents with shared/competing attention
3. **Real-time Metrics**: Track attention coherence, topic diversity
4. **Advanced Schemas**: Hierarchical or weighted attention (primary vs. secondary focus)

---

## Conclusion

This is a **faithful and functional** implementation of AST principles. It successfully demonstrates:
- ✅ Internal self-model of attention (AttentionSchema)
- ✅ Attention-guided context selection (AttentionMechanism)
- ✅ Schema-informed LLM behavior (Controller with system prompt injection)
- ✅ Testable, modular design

With the recommended enhancements, this could become a production-grade framework for attention-aware LLM agents.

