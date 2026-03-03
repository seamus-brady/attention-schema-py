"""Multi-turn test scenarios for A/B evaluation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Scenario:
    name: str
    category: str  # "coherence" | "shift" | "self_report"
    turns: list[str]
    probe_turns: list[int]  # indices into turns to evaluate
    expected_topic: str  # correct focus at probe turns


# ---------------------------------------------------------------------------
# Category A — Topic Coherence (5 scenarios)
# Sustained topic, then an ambiguous prompt.
# ---------------------------------------------------------------------------

_COHERENCE_SCENARIOS = [
    Scenario(
        name="python_errors_then_types",
        category="coherence",
        turns=[
            "I keep getting ValueError in my Python code. What causes those?",
            "How should I handle multiple exceptions at once?",
            "What about creating custom exception classes?",
            "Is it better to use try/except or check conditions first (LBYL vs EAFP)?",
            "Can you show me a retry pattern with exponential backoff?",
            "How do logging and exceptions work together?",
            "What are exception groups in Python 3.11?",
            "What about types?",
        ],
        probe_turns=[7],
        expected_topic="Python type hints / typing in the context of error handling",
    ),
    Scenario(
        name="react_hooks_then_state",
        category="coherence",
        turns=[
            "Explain React's useEffect hook.",
            "When does the cleanup function run?",
            "How do I avoid infinite re-render loops with useEffect?",
            "What's the difference between useEffect and useLayoutEffect?",
            "Can you explain the dependency array in more detail?",
            "What about state?",
        ],
        probe_turns=[5],
        expected_topic="React state management in the context of hooks",
    ),
    Scenario(
        name="sql_queries_then_performance",
        category="coherence",
        turns=[
            "How do SQL JOINs work?",
            "What's the difference between INNER JOIN and LEFT JOIN?",
            "When should I use a subquery vs a JOIN?",
            "How do window functions compare to GROUP BY?",
            "Tell me about CTEs and recursive queries.",
            "What about performance?",
        ],
        probe_turns=[5],
        expected_topic="SQL query performance and optimization",
    ),
    Scenario(
        name="git_branching_then_conflicts",
        category="coherence",
        turns=[
            "What's a good Git branching strategy for a small team?",
            "How does Git Flow compare to trunk-based development?",
            "When should I rebase vs merge?",
            "How do feature flags relate to branching?",
            "What about conflicts?",
        ],
        probe_turns=[4],
        expected_topic="Git merge conflicts in the context of branching strategies",
    ),
    Scenario(
        name="docker_containers_then_networking",
        category="coherence",
        turns=[
            "How do Docker containers differ from VMs?",
            "What's the best practice for writing Dockerfiles?",
            "How do multi-stage builds work?",
            "What about volumes and persistent storage?",
            "Can you explain Docker Compose for multi-container setups?",
            "What about networking?",
        ],
        probe_turns=[5],
        expected_topic="Docker networking between containers",
    ),
]


# ---------------------------------------------------------------------------
# Category B — Topic Shifts (5 scenarios)
# Abrupt mid-conversation topic changes.
# ---------------------------------------------------------------------------

_SHIFT_SCENARIOS = [
    Scenario(
        name="cooking_to_neural_nets",
        category="shift",
        turns=[
            "What's a good recipe for homemade pasta?",
            "How do I make the dough without a pasta machine?",
            "What sauces pair well with fresh fettuccine?",
            "How long does fresh pasta keep in the fridge?",
            "Actually, how do neural networks work?",
        ],
        probe_turns=[4],
        expected_topic="Neural networks (abrupt shift from cooking)",
    ),
    Scenario(
        name="history_to_programming",
        category="shift",
        turns=[
            "Tell me about the causes of World War I.",
            "How did the alliance system contribute?",
            "What role did nationalism play?",
            "Completely different question — how do I set up a REST API in FastAPI?",
        ],
        probe_turns=[3],
        expected_topic="FastAPI REST API setup (shift from WWI history)",
    ),
    Scenario(
        name="fitness_to_astronomy",
        category="shift",
        turns=[
            "What's a good beginner weightlifting program?",
            "How many sets and reps should I do for hypertrophy?",
            "What about rest days and recovery?",
            "How important is progressive overload?",
            "Switch topics: how far away is the nearest star?",
            "What's the current status of the James Webb Space Telescope discoveries?",
        ],
        probe_turns=[4, 5],
        expected_topic="Astronomy and space (shift from fitness)",
    ),
    Scenario(
        name="gardening_to_cryptography",
        category="shift",
        turns=[
            "When should I plant tomatoes in zone 7?",
            "How much sun do tomato plants need?",
            "What about companion planting with basil?",
            "Forget gardening — explain public key cryptography to me.",
            "How does RSA encryption work specifically?",
        ],
        probe_turns=[3, 4],
        expected_topic="Public key cryptography / RSA (shift from gardening)",
    ),
    Scenario(
        name="music_to_databases",
        category="shift",
        turns=[
            "What makes jazz improvisation different from classical performance?",
            "How do jazz musicians communicate during solos?",
            "What's the role of the rhythm section?",
            "New topic entirely: explain ACID properties in databases.",
        ],
        probe_turns=[3],
        expected_topic="Database ACID properties (shift from jazz music)",
    ),
]


# ---------------------------------------------------------------------------
# Category C — Self-Report Accuracy (5 scenarios)
# Ask the bot what it's focused on or what we've been discussing.
# ---------------------------------------------------------------------------

_SELF_REPORT_SCENARIOS = [
    Scenario(
        name="three_topics_then_ask",
        category="self_report",
        turns=[
            "Tell me about photosynthesis.",
            "How does the Calvin cycle work?",
            "Now tell me about the French Revolution.",
            "What caused the Reign of Terror?",
            "Let's talk about machine learning overfitting.",
            "What's the bias-variance tradeoff?",
            "What have we been discussing in this conversation?",
        ],
        probe_turns=[6],
        expected_topic="Three topics: photosynthesis, French Revolution, ML overfitting",
    ),
    Scenario(
        name="single_topic_focus_check",
        category="self_report",
        turns=[
            "Explain how TCP/IP works.",
            "What happens during the three-way handshake?",
            "How does TCP handle packet loss?",
            "What are we currently talking about?",
        ],
        probe_turns=[3],
        expected_topic="TCP/IP networking protocols",
    ),
    Scenario(
        name="gradual_drift_then_ask",
        category="self_report",
        turns=[
            "Tell me about Python decorators.",
            "How do decorators relate to closures?",
            "What about first-class functions in general?",
            "How does functional programming differ from OOP?",
            "What are monads in functional programming?",
            "What's our current topic?",
        ],
        probe_turns=[5],
        expected_topic="Functional programming / monads (drifted from Python decorators)",
    ),
    Scenario(
        name="shift_then_ask_about_both",
        category="self_report",
        turns=[
            "How does the human immune system fight viruses?",
            "What's the difference between innate and adaptive immunity?",
            "Tell me about quantum computing instead.",
            "What are qubits?",
            "Can you summarize what we've talked about so far?",
        ],
        probe_turns=[4],
        expected_topic="Two topics: immune system / virology, then quantum computing",
    ),
    Scenario(
        name="rapid_topic_changes_then_ask",
        category="self_report",
        turns=[
            "What's the capital of Mongolia?",
            "How do airplanes generate lift?",
            "What's the deepest point in the ocean?",
            "Explain the Monty Hall problem.",
            "We've jumped around a lot — what have we covered?",
        ],
        probe_turns=[4],
        expected_topic="Four quick topics: Mongolia, aerodynamics, ocean depth, Monty Hall problem",
    ),
]


SCENARIOS: list[Scenario] = (
    _COHERENCE_SCENARIOS + _SHIFT_SCENARIOS + _SELF_REPORT_SCENARIOS
)
