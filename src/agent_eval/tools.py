"""Tools the agent under test can call.

The environment is a fictional outdoor-gear shop: a SQLite database (data/shop.sql)
and a small set of policy documents (data/docs/*.md). Everything is local and
deterministic so benchmark runs are repeatable.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

MAX_ROWS = 50


class ToolError(Exception):
    """Raised when a tool call is invalid. The message is shown to the agent."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the arguments
    fn: Callable[..., Any] = field(repr=False)

    def schema(self) -> dict[str, Any]:
        """OpenAI-style function definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Environment:
    """One isolated copy of the shop. Create a new one per task."""

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.executescript((data_dir / "shop.sql").read_text(encoding="utf-8"))
        self.db.execute("PRAGMA query_only = ON")
        self.docs = {
            p.stem: p.read_text(encoding="utf-8") for p in sorted((data_dir / "docs").glob("*.md"))
        }
        self.tools = {
            t.name: t
            for t in [
                Tool(
                    name="sql_query",
                    description=(
                        "Run a read-only SQLite SELECT query on the shop database. "
                        "Tables: customers(id, name, country, signup_date), "
                        "products(id, name, category, price), "
                        "orders(id, customer_id, order_date, status), "
                        "order_items(order_id, product_id, quantity)."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    fn=self.sql_query,
                ),
                Tool(
                    name="search_docs",
                    description="Keyword search over the shop's policy documents. Returns doc ids and snippets.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "required": ["query"],
                    },
                    fn=self.search_docs,
                ),
                Tool(
                    name="read_doc",
                    description="Return the full text of a policy document by its doc id.",
                    parameters={
                        "type": "object",
                        "properties": {"doc_id": {"type": "string"}},
                        "required": ["doc_id"],
                    },
                    fn=self.read_doc,
                ),
            ]
        }

    def call(self, name: str, args: dict[str, Any]) -> Any:
        if name not in self.tools:
            raise ToolError(f"unknown tool: {name}")
        try:
            return self.tools[name].fn(**args)
        except TypeError as e:
            raise ToolError(f"bad arguments for {name}: {e}") from e

    def sql_query(self, query: str) -> dict[str, Any]:
        try:
            cur = self.db.execute(query)
        except sqlite3.Error as e:
            raise ToolError(f"SQL error: {e}") from e
        columns = [d[0] for d in cur.description or []]
        rows = [list(r) for r in cur.fetchmany(MAX_ROWS + 1)]
        return {"columns": columns, "rows": rows[:MAX_ROWS], "truncated": len(rows) > MAX_ROWS}

    def search_docs(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        terms = set(re.findall(r"\w+", query.lower()))
        hits = []
        for doc_id, text in self.docs.items():
            paragraphs = [p for p in text.split("\n\n") if p.strip() and not p.startswith("#")]
            scored = [(len(terms & set(re.findall(r"\w+", p.lower()))), p) for p in paragraphs]
            score, best = max(scored)
            if score:
                hits.append({"doc_id": doc_id, "score": score, "snippet": best.strip()})
        hits.sort(key=lambda h: (-h["score"], h["doc_id"]))
        return hits[:top_k]

    def read_doc(self, doc_id: str) -> str:
        if doc_id not in self.docs:
            raise ToolError(f"unknown doc_id: {doc_id}. Available: {', '.join(self.docs)}")
        return self.docs[doc_id]
