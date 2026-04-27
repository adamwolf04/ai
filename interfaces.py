"""
Abstract interfaces and shared data models for the stem agent system.

All runtime components (AgentRuntime, Evaluator, MutationEngine, etc.)
implement these ABCs so they can be swapped, mocked, and tested independently.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RunResult(BaseModel):
    """Structured output from one agent execution on one task."""
    report:      str           = Field(default="")
    steps:       int           = Field(default=0)
    status:      str           = Field(default="success")   # success | error | max_steps_reached | stopped_with_errors
    trace:       List[Dict]    = Field(default_factory=list)  # full message history
    token_count: int           = Field(default=0)
    error:       Optional[str] = Field(default=None)

    def to_dict(self) -> dict:
        return self.model_dump()


class EvalResult(BaseModel):
    """Structured output from one evaluation of a RunResult."""
    score:    float      = Field(default=0.0)
    matched:  bool       = Field(default=False)
    failures: List[str]  = Field(default_factory=list)
    trace:    Dict       = Field(default_factory=dict)

    def to_dict(self) -> dict:
        return self.model_dump()


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class TaskProvider(ABC):
    @abstractmethod
    def get_tasks(self, split: str) -> List[dict]:
        """Return a list of task dicts for the given split (train/test/regression)."""
        ...


class AgentRuntime(ABC):
    @abstractmethod
    async def run(self, spec: Any, task: str) -> RunResult:
        """Run a single agent spec on a single task string. Returns RunResult."""
        ...


class Evaluator(ABC):
    @abstractmethod
    async def evaluate(self, task: dict, result: RunResult) -> EvalResult:
        """Score a RunResult against a task. Returns EvalResult."""
        ...


class MutationEngine(ABC):
    @abstractmethod
    async def mutate(self, spec: Any, failures: List[str],
                     model: str) -> Optional[Any]:
        """Produce a mutated child spec. Returns None on failure."""
        ...


class PopulationStore(ABC):
    @abstractmethod
    def add(self, spec: Any) -> None: ...

    @abstractmethod
    def record_score(self, spec_id: str, score: float, failures: List[str]) -> None: ...

    @abstractmethod
    def get_top_k(self, k: int) -> List[Any]: ...

    @abstractmethod
    def get_best(self) -> Optional[Any]: ...

    @abstractmethod
    def cull(self, max_size: int) -> None: ...
