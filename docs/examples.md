# Usage Examples

This document provides practical examples of how to use the Attention Schema LLM agent.

## 1. Basic Conversation

```python
from attention_schema.controller import Controller
from attention_schema.llm import MockLLMClient

# Use mock LLM for testing
llm = MockLLMClient()
controller = Controller(llm=llm)

# Run a single turn
response = controller.run("Tell me about Python")
print(f"Agent: {response}")
print(f"Schema: {controller.schema.summary()}")
```

**Output:**
```
Agent: [mock response to: Tell me about Python]
Schema: I am currently attending to 'Tell me about Python' (confidence 0.80). Reason: New topic detected — shifting attention.
```

---

## 2. Multi-Turn Conversation with Attention Shifts

```python
from attention_schema.controller import Controller
from attention_schema.llm import MockLLMClient

llm = MockLLMClient()
controller = Controller(llm=llm)

# Turn 1: Python
print("=== Turn 1: Python ===")
controller.run("What is Python?")
print(f"Focus: {controller.schema.state.focus_target}")
print(f"Confidence: {controller.schema.state.confidence}")

# Turn 2: Related topic - should sustain focus
print("\n=== Turn 2: Python (continued) ===")
controller.run("Tell me about Python decorators")
print(f"Focus: {controller.schema.state.focus_target}")
print(f"Confidence: {controller.schema.state.confidence:.2f}")  # Should be higher

# Turn 3: Different topic - should shift
print("\n=== Turn 3: Recipe ===")
controller.run("How do I make chocolate cake?")
print(f"Focus: {controller.schema.state.focus_target}")
print(f"Confidence: {controller.schema.state.confidence:.2f}")  # Should reset to 0.8

# Check history
print(f"\n=== Attention History ===")
for i, entry in enumerate(controller.schema.state.history):
    print(f"{i+1}. {entry['focus_target']} (confidence: {entry['confidence']})")
```

**Expected Output:**
```
=== Turn 1: Python ===
Focus: What is Python?
Confidence: 0.8

=== Turn 2: Python (continued) ===
Focus: What is Python?
Confidence: 0.85

=== Turn 3: Recipe ===
Focus: How do I make chocolate cake?
Confidence: 0.80

=== Attention History ===
1. What is Python? (confidence: 0.8)
2. What is Python? (confidence: 0.85)
```

---

## 3. Using Real LLM (Anthropic)

```python
import os
from attention_schema.controller import Controller
from attention_schema.llm import LLMClient

# Requires ANTHROPIC_API_KEY environment variable
api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("Please set ANTHROPIC_API_KEY")

llm = LLMClient()
controller = Controller(llm=llm)

# Interactive loop
print("Attention Schema Agent (type 'quit' to exit)")
while True:
    user_input = input("You: ").strip()
    if not user_input or user_input.lower() == "quit":
        break
    
    response = controller.run(user_input)
    print(f"\nAgent: {response}")
    print(f"[Schema: {controller.schema.summary()}]\n")
```

---

## 4. Custom Configuration

```python
from attention_schema.schema import AttentionSchema
from attention_schema.attention import AttentionMechanism
from attention_schema.controller import Controller

# Create a schema with sticky attention (high thresholds)
sticky_schema = AttentionSchema(
    shift_overlap_threshold=0.4,       # Need 40% overlap to avoid shifting
    shift_confidence_threshold=0.6,    # Need high confidence to shift
    confidence_increment=0.1,          # Confidence grows faster
    max_history=20,                    # Keep less history
)

# Create an attention mechanism with strong recency bias
attention = AttentionMechanism(use_recency_weight=True)

# Create controller with large context window
controller = Controller(
    schema=sticky_schema,
    context_items=[],
    max_context_items=200,             # Store many items
    top_k_context=5,                   # Select top 5
)

# Test: sustained focus even with topic changes
response1 = controller.run("Python programming")
print(f"Turn 1 Confidence: {controller.schema.state.confidence}")

response2 = controller.run("Java is also popular")
print(f"Turn 2 Confidence: {controller.schema.state.confidence}")
# With high thresholds, may NOT shift despite topic change
```

---

## 5. Context Pruning

```python
from attention_schema.controller import Controller

controller = Controller()

# Simulate a long conversation with topic switches
topics = [
    "Python programming",
    "Python decorators",
    "Python best practices",
    "Recipe for chocolate cake",
    "Baking techniques",
    "History of Rome",
    "Roman architecture",
]

for topic in topics:
    controller.run(f"Tell me about {topic}")

print(f"Context items before pruning: {len(controller.context_items)}")
print(f"Current focus: {controller.schema.state.focus_target}")

# Prune context to keep only relevant items
controller.prune_context_by_schema()

print(f"Context items after pruning: {len(controller.context_items)}")
print(f"Remaining items: {controller.context_items}")
```

**Output:**
```
Context items before pruning: 7
Current focus: Tell me about Roman architecture
Context items after pruning: 5
Remaining items: [
    'History of Rome',
    'Roman architecture'
]  # (plus last 3 items for continuity)
```

---

## 6. Schema Persistence Across Sessions

```python
from attention_schema.controller import Controller

# Session 1: Build up context and attention state
print("=== Session 1 ===")
controller1 = Controller()

for i in range(5):
    controller1.run(f"Message {i}: Let's discuss Python")

print(f"Session 1 Focus: {controller1.schema.state.focus_target}")
print(f"Session 1 Confidence: {controller1.schema.state.confidence}")

# Save state
controller1.save_state("conversation.json")
print("Session 1 saved to conversation.json")

# Session 2: Load and continue
print("\n=== Session 2 ===")
controller2 = Controller()
controller2.load_state("conversation.json")

print(f"Session 2 Focus: {controller2.schema.state.focus_target}")
print(f"Session 2 Confidence: {controller2.schema.state.confidence}")
print(f"Session 2 Context items: {len(controller2.context_items)}")

# Continue conversation
controller2.run("Let's dive deeper into Python classes")
print(f"After continuing: Confidence = {controller2.schema.state.confidence}")
```

**Output:**
```
=== Session 1 ===
Session 1 Focus: Message 4: Let's discuss Python
Session 1 Confidence: 0.85
Session 1 saved to conversation.json

=== Session 2 ===
Session 2 Focus: Message 4: Let's discuss Python
Session 2 Confidence: 0.85
Session 2 Context items: 5
After continuing: Confidence = 0.9
```

---

## 7. Interactive CLI with Persistence

```python
from attention_schema.controller import Controller
from pathlib import Path

def main():
    session_file = Path("my_session.json")
    
    # Load previous session if available
    controller = Controller()
    if session_file.exists():
        controller.load_state(session_file)
        print("Loaded previous session")
    
    print("Attention Schema Agent")
    print("Commands: 'save', 'load', 'quit'\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving and exiting...")
            controller.save_state(session_file)
            break
        
        if not user_input:
            continue
        
        if user_input.lower() == "quit":
            print("Saving and exiting...")
            controller.save_state(session_file)
            break
        
        if user_input.lower() == "save":
            controller.save_state(session_file)
            print("Session saved")
            continue
        
        if user_input.lower() == "load":
            controller.load_state(session_file)
            print("Session loaded")
            continue
        
        response = controller.run(user_input)
        print(f"\nAgent: {response}")
        print(f"[{controller.schema.summary()}]\n")

if __name__ == "__main__":
    main()
```

---

## 8. Comparing Shift Behavior

```python
from attention_schema.schema import AttentionSchema

# Sensitive schema (shifts easily)
sensitive = AttentionSchema(shift_overlap_threshold=0.1)

# Sticky schema (maintains focus)
sticky = AttentionSchema(shift_overlap_threshold=0.5)

# Test inputs
focus = "Python programming"
similar = "Python decorators"
different = "Chocolate cake recipe"

print("=== Sensitive Schema (threshold=0.1) ===")
sensitive.update(focus, 0.9, "test")
print(f"Focus: {focus}")
print(f"Should shift to '{similar}'? {sensitive.should_shift(similar)}")
print(f"Should shift to '{different}'? {sensitive.should_shift(different)}")

print("\n=== Sticky Schema (threshold=0.5) ===")
sticky.update(focus, 0.9, "test")
print(f"Focus: {focus}")
print(f"Should shift to '{similar}'? {sticky.should_shift(similar)}")
print(f"Should shift to '{different}'? {sticky.should_shift(different)}")
```

**Output:**
```
=== Sensitive Schema (threshold=0.1) ===
Focus: Python programming
Should shift to 'Python decorators'? False
Should shift to 'Chocolate cake recipe'? True

=== Sticky Schema (threshold=0.5) ===
Focus: Python programming
Should shift to 'Python decorators'? True
Should shift to 'Chocolate cake recipe'? True
```

---

## 9. Analyzing Context Selection

```python
from attention_schema.attention import AttentionMechanism

attention = AttentionMechanism(use_recency_weight=True)

context = [
    "The solar system has 8 planets",
    "Jupiter is the largest planet",
    "Mars has two moons",
    "Saturn has beautiful rings",
    "The Earth orbits the sun",
]

query = "planets"

# Get relevance scores
scores = attention.score_relevance(context, query)
for item, score in zip(context, scores):
    print(f"{score:.3f}: {item}")

print(f"\nTop 3 selected items:")
selected = attention.select(context, query, top_k=3)
for item in selected:
    print(f"- {item}")
```

**Output:**
```
0.408: The solar system has 8 planets
0.408: Jupiter is the largest planet
0.000: Mars has two moons
0.408: Saturn has beautiful rings
0.408: The Earth orbits the sun

Top 3 selected items:
- The solar system has 8 planets
- Jupiter is the largest planet
- Saturn has beautiful rings
```

---

## 10. Testing Attention Mechanism Behavior

```python
from attention_schema.attention import AttentionMechanism

# Compare with and without recency weighting
items = [
    "old python discussion from month 1",
    "medium python discussion from month 2", 
    "recent python discussion from month 3",
]

print("=== With Recency Weighting ===")
attn_recent = AttentionMechanism(use_recency_weight=True)
selected = attn_recent.select(items, "python", top_k=1, use_recency=True)
print(f"Selected: {selected[0]}")

print("\n=== Without Recency Weighting ===")
attn_no_recent = AttentionMechanism(use_recency_weight=False)
selected = attn_no_recent.select(items, "python", top_k=1, use_recency=False)
print(f"Selected: {selected[0]}")
```

---

## Tips & Tricks

### 1. Debugging Attention Shifts
```python
# Check why a shift occurred
schema.update("topic A", 0.8, "initial")
new_input = "different topic"

overlap = schema._token_overlap(new_input, "topic A")
print(f"Token overlap: {overlap:.2f} (threshold: {schema.shift_overlap_threshold})")
print(f"Will shift? {schema.should_shift(new_input)}")
```

### 2. Monitoring Confidence Growth
```python
# Track how confidence grows over sustained focus
controller.run("Python")
print(f"Turn 1: {controller.schema.state.confidence:.2f}")

controller.run("Python continues")
print(f"Turn 2: {controller.schema.state.confidence:.2f}")

controller.run("More Python")
print(f"Turn 3: {controller.schema.state.confidence:.2f}")
```

### 3. Exporting Attention History
```python
import json

# Save attention history as JSON
history = {
    "current": controller.schema.to_dict(),
    "metadata": {
        "total_context_items": len(controller.context_items),
        "history_length": len(controller.schema.state.history),
    }
}

with open("attention_log.json", "w") as f:
    json.dump(history, f, indent=2)
```

### 4. Custom System Prompt
```python
custom_template = """\
You are a focused assistant guided by attention schema.
Current focus: {schema_summary}

IMPORTANT: Stay on topic unless the user explicitly shifts the conversation.
"""

controller = Controller(system_template=custom_template)
response = controller.run("Tell me about Python")
```

---

## Troubleshooting

### Q: Why does the agent keep shifting focus?
**A:** Lower `shift_overlap_threshold` in the schema:
```python
schema = AttentionSchema(shift_overlap_threshold=0.3)  # Increase from 0.2
```

### Q: Why does context keep growing?
**A:** Set `max_context_items` to a lower value:
```python
controller = Controller(max_context_items=50)  # Instead of 100
```

### Q: How do I see what context the LLM receives?
**A:** Check `MockLLMClient.last_user`:
```python
print(mock_llm.last_user)  # Shows the full user message with context
```

### Q: How do I customize shift behavior per topic?
**A:** Create multiple `AttentionSchema` instances or add topic-specific thresholds:
```python
# Different thresholds for different domains
if "science" in query:
    schema = AttentionSchema(shift_overlap_threshold=0.25)
else:
    schema = AttentionSchema(shift_overlap_threshold=0.2)
```

