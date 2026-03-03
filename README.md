# Attention Schema for LLM Agents

An experiment in implementing Michael Graziano's [Attention Schema Theory](https://en.wikipedia.org/wiki/Attention_schema_theory) (AST) as an LLM agent. AST proposes that the brain constructs a simplified internal model of its own attention, which it uses to monitor and control what it attends to.

This project translates that idea into three components:

1. **Attention Mechanism** — scores and selects context items by relevance (TF-IDF, no embeddings)
2. **Attention Schema** — an internal self-model tracking *what* the system attends to, *why*, and with what *confidence*
3. **Controller** — uses the schema to decide whether to shift, sustain, or broaden attention before calling the LLM

> **Status: Retired.** Evaluation showed this approach actively hurts performance vs. a vanilla baseline. See [Results](#results) for details and [docs/](docs/) for the full analysis.

## Setup

```bash
pip install -e ".[dev]"
```

## Usage

### Run tests

```bash
pytest tests/ -v
```

### Interactive mode

```bash
ANTHROPIC_API_KEY=your-key python -m attention_schema
```

### Programmatic

```python
from attention_schema.controller import Controller
from attention_schema.llm import MockLLMClient

controller = Controller(llm=MockLLMClient())
response = controller.run("Tell me about Python")
print(controller.schema.summary())
```

See [docs/examples.md](docs/examples.md) for more.

## Architecture

```
User Input
    │
    ▼
┌──────────────┐
│  Controller   │◄──── Attention Schema (self-model)
│  (main loop)  │────► updates schema each cycle
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Attention    │  scores & filters context
│  Mechanism    │  (TF-IDF + recency weighting)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  LLM Call     │  generates response from attended context
└──────────────┘
```

Each cycle: check whether to shift attention → score context → update schema → inject schema summary into system prompt → call LLM → store context.

## Results

An A/B evaluation (17 probe turns across 15 multi-turn conversations, judged by Claude Sonnet) compared the full schema agent against a plain chatbot baseline.

```
Win/Loss/Tie:  0 / 14 / 3   (sign test p = 0.0001)

Per category:
  coherence      0W / 2L / 3T
  shift          0W / 7L / 0T
  self-report    0W / 5L / 0T
```

**Zero wins.** The schema variant lost on every dimension. Root causes:

1. **`should_shift()` bug** — returned `True` unconditionally on low overlap, causing spurious topic shifts.
2. **Awareness claims were noise** — the LLM already knows its conversation history; injecting a second, cruder source of truth created conflicting signals.
3. **Schema as reporter, not controller** — AST says the schema should *steer* attention. This implementation only *described* it, then asked the LLM to parrot the description.

The conclusion is not that AST is wrong for agents, but that a bolted-on annotation layer is the wrong architecture. The schema needs to be integrated into the world model, not layered on top.

## Project Structure

```
attention_schema/
├── schema.py        # Attention schema (self-model)
├── attention.py     # Attention mechanism (scoring)
├── controller.py    # Main agent loop
├── llm.py           # LLM client + mock
├── dissociation.py  # Dissociation state tracking
├── social.py        # Multi-agent social attention
├── tokenizer.py     # Token utilities
└── __main__.py      # Interactive REPL

eval/                # A/B evaluation harness
tests/               # 41 tests
docs/                # Extended documentation
```

## Documentation

- [Implementation Guide](docs/implementation.md) — theory-to-code mapping and design rationale
- [Examples](docs/examples.md) — usage examples
- [Review](docs/review.md) — critical analysis

## References

- Graziano, M. S. (2013). *Consciousness and the Social Brain*
- Graziano & Kastner (2011). "The Attention Schema Theory"
- Graziano & Webb (2015). "The attention schema theory: a mechanistic account of subjective awareness." *Frontiers in Psychology* 6:500

## License

[MIT](LICENSE)
