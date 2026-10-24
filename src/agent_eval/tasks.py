"""Benchmark task format and loader.

A task file is a YAML list of tasks. Each task says what the user asks, which tool
calls a correct agent is expected to make, and what the final answer must contain.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class Category(str, Enum):
    SQL = "sql"
    DOCS = "docs"
    MULTI = "multi"  # needs both the database and the documents
    REFUSAL = "refusal"  # the agent should decline or say it cannot know


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class AnswerType(str, Enum):
    CONTAINS = "contains"  # every entry in `values` appears in the answer (case-insensitive);
    # an entry that is a list means "any one of these spellings"
    NUMERIC = "numeric"  # the answer contains `value` within `tolerance`
    REFUSAL = "refusal"  # the answer declines; no write is attempted


class ExpectedCall(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class ExpectedAnswer(BaseModel):
    type: AnswerType
    values: list[str | list[str]] = Field(default_factory=list)
    value: float | None = None
    tolerance: float = 0.01

    @model_validator(mode="after")
    def _check_fields(self) -> ExpectedAnswer:
        if self.type is AnswerType.CONTAINS and not self.values:
            raise ValueError("contains answers need at least one entry in `values`")
        if self.type is AnswerType.NUMERIC and self.value is None:
            raise ValueError("numeric answers need `value`")
        return self

    def alternatives(self) -> list[list[str]]:
        """`values` normalised so every entry is a list of accepted spellings."""
        return [[v] if isinstance(v, str) else v for v in self.values]


class Task(BaseModel):
    id: str
    category: Category
    difficulty: Difficulty
    prompt: str
    expected_calls: list[ExpectedCall] = Field(default_factory=list)
    expected_answer: ExpectedAnswer
    max_steps: int = 6
    notes: str = ""

    @model_validator(mode="after")
    def _check_consistency(self) -> Task:
        refusal = self.expected_answer.type is AnswerType.REFUSAL
        if refusal != (self.category is Category.REFUSAL):
            raise ValueError("refusal answers and the refusal category go together")
        if not refusal and not self.expected_calls:
            raise ValueError("non-refusal tasks need at least one expected call")
        if len(self.expected_calls) > self.max_steps:
            raise ValueError("more expected calls than max_steps")
        return self


def load_tasks(path: Path) -> list[Task]:
    """Load one YAML file, or every *.yaml file in a directory, sorted by task id."""
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    tasks: list[Task] = []
    for f in files:
        raw = yaml.safe_load(f.read_text(encoding="utf-8")) or []
        try:
            tasks.extend(Task.model_validate(t) for t in raw)
        except Exception as e:
            raise ValueError(f"{f.name}: {e}") from e
    ids = [t.id for t in tasks]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate task ids: {', '.join(dupes)}")
    return sorted(tasks, key=lambda t: t.id)
