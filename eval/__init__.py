"""Empirical evaluation harness for attention schema chatbot."""

from .baseline import BaselineController, ClaimsOnlyController, AttentionOnlyController
from .scenarios import SCENARIOS, Scenario
from .judge import Judge, JudgeScore
from .runner import run_evaluation, EvalResult
from .metrics import compute_metrics, print_summary

__all__ = [
    "BaselineController",
    "ClaimsOnlyController",
    "AttentionOnlyController",
    "SCENARIOS",
    "Scenario",
    "Judge",
    "JudgeScore",
    "run_evaluation",
    "EvalResult",
    "compute_metrics",
    "print_summary",
]
