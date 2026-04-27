"""
Evolution operators — seed, LLM mutation, random mutation, crossover.

Key improvements:
  - LLM mutations can now change: system_prompt, tools, planning_strategy,
    stop_condition.must_include_citations  (i.e., architectural axes, not just text)
  - Failure prompt includes actual bad output snippets from the lineage DB
  - crossover_mutation: takes system_prompt from parent A, tools + strategy from B
  - random_mutation: can also flip planning_strategy
  - All operators return None on failure (never return an unmodified parent)
"""
from __future__ import annotations

import json
import os
import random
from typing import Optional

from openai import OpenAI
from spec import AgentSpec
from runtime.tool_belt import AVAILABLE_TOOLS


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

_SEED_PROMPT = """
You are an evolutionary stem agent. Generate 5 diverse JSON specs for a Deep Research agent.

Seeding rules:
- Start minimal and crash-free. Add sophistication only where justified.
- Vary: planning_strategy (react OR plan_and_solve), tool choices (subsets of the available tools), system prompt tone.
- Use max_steps no higher than 8.
- No two specs may be identical.

Available tools: search_web, scrape_page, python_repl

Output JSON:
{
  "specs": [
    {
      "id": "seed_1",
      "system_prompt": "You are a research assistant. Answer with citations.",
      "tools": ["search_web", "scrape_page"],
      "planning_strategy": "react",
      "stop_condition": {"min_report_length": 200, "must_include_citations": true, "max_steps": 6}
    }
  ]
}
"""


def seed_population(model: str = "gpt-4o-mini") -> list[AgentSpec]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _SEED_PROMPT}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        specs = []
        for s in data.get("specs", []):
            s.setdefault("generation", 0)
            s.setdefault("parent_id", None)
            s.setdefault("mutation_type", "seed")
            try:
                spec = AgentSpec.from_dict(s)
                if spec.validate_spec(AVAILABLE_TOOLS):
                    specs.append(spec)
            except Exception:
                pass
        return specs
    except Exception as e:
        print(f"[seed_population] Error: {e}")
        return []


# ---------------------------------------------------------------------------
# LLM mutation  (targeted, architecture-aware)
# ---------------------------------------------------------------------------

def llm_mutation(
    spec: AgentSpec,
    failure_stack: list[str],
    model: str = "gpt-4o-mini",
    failure_examples: Optional[list[dict]] = None,
) -> Optional[AgentSpec]:
    """
    Ask an LLM to fix a failing spec.

    The prompt includes:
      - Current spec JSON
      - The top failure message
      - Actual bad output snippets (from lineage DB, if provided)
      - Explicit permission to change architecture (tools, planning_strategy)
    """
    if failure_stack:
        top_issue = failure_stack[0]
    else:
        top_issue = (
            "The agent is performing adequately but may improve. "
            "Suggest any change that increases answer accuracy and citation usage."
        )

    # Format failure examples
    examples_section = ""
    if failure_examples:
        snippets = "\n".join(
            f"  Task: {ex['task_id']}\n  Output snippet: {ex['output_snippet']!r}"
            for ex in failure_examples[:3]
        )
        examples_section = f"\nFailing output examples:\n{snippets}\n"

    prompt = f"""
You are an evolutionary stem agent. Improve a failing research agent spec.

Current spec:
{json.dumps(spec.to_dict(), indent=2)}
{examples_section}
CRITICAL ISSUE TO FIX:
{top_issue}

Available tools: {AVAILABLE_TOOLS}
Available planning strategies: react, plan_and_solve

You MAY change any of:
  - system_prompt (add explicit instructions for tool use, self-checking, citation format)
  - tools (add or remove tools from the available list above)
  - planning_strategy (switch between react and plan_and_solve)
  - stop_condition.must_include_citations (true or false)
  - stop_condition.min_report_length (integer)
  - stop_condition.max_steps (integer, max 12)

Return the COMPLETE modified spec as valid JSON only. Do not include explanation text.
"""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        data["id"]            = spec.id.split("_")[0] + "_m" + str(random.randint(100, 999))
        data["parent_id"]     = spec.id
        data["generation"]    = spec.generation + 1
        data["mutation_type"] = "llm"
        child = AgentSpec.from_dict(data)
        return child if child.validate_spec(AVAILABLE_TOOLS) else None
    except Exception as e:
        print(f"[llm_mutation] Error: {e}")
        return None


# ---------------------------------------------------------------------------
# Random mutation  (small structural tweaks)
# ---------------------------------------------------------------------------

def random_mutation(spec: AgentSpec) -> Optional[AgentSpec]:
    """
    Apply one random micro-mutation to the spec.
    Can mutate: max_steps, min_report_length, or planning_strategy.
    """
    data = spec.to_dict()
    data["id"]            = spec.id.split("_")[0] + "_r" + str(random.randint(100, 999))
    data["parent_id"]     = spec.id
    data["generation"]    = spec.generation + 1
    data["mutation_type"] = "random"

    choice = random.randint(0, 2)

    if choice == 0:
        # Mutate max_steps
        delta = random.choice([-1, 1, 2])
        data["stop_condition"]["max_steps"] = max(
            3, data["stop_condition"]["max_steps"] + delta
        )
    elif choice == 1:
        # Mutate min_report_length
        delta = random.choice([-50, 50, 100])
        data["stop_condition"]["min_report_length"] = max(
            50, data["stop_condition"]["min_report_length"] + delta
        )
    else:
        # Flip planning_strategy
        current = data.get("planning_strategy", "react")
        data["planning_strategy"] = (
            "plan_and_solve" if current == "react" else "react"
        )

    try:
        child = AgentSpec.from_dict(data)
        return child if child.validate_spec(AVAILABLE_TOOLS) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Crossover  (combine two parents)
# ---------------------------------------------------------------------------

def crossover_mutation(spec_a: AgentSpec, spec_b: AgentSpec) -> Optional[AgentSpec]:
    """
    Create a child by taking:
      - system_prompt from parent A
      - tools + planning_strategy from parent B
      - stop_condition averaged between A and B
    """
    try:
        data_a = spec_a.to_dict()
        data_b = spec_b.to_dict()

        sc_a = data_a["stop_condition"]
        sc_b = data_b["stop_condition"]

        child_data = {
            "id":               spec_a.id.split("_")[0] + "_x" + str(random.randint(100, 999)),
            "parent_id":        spec_a.id,   # primary parent is A
            "generation":       max(spec_a.generation, spec_b.generation) + 1,
            "mutation_type":    "crossover",
            "system_prompt":    data_a["system_prompt"],
            "tools":            data_b["tools"],
            "planning_strategy": data_b["planning_strategy"],
            "stop_condition": {
                "min_report_length":     (sc_a["min_report_length"] + sc_b["min_report_length"]) // 2,
                "must_include_citations": sc_a["must_include_citations"] or sc_b["must_include_citations"],
                "max_steps":             (sc_a["max_steps"] + sc_b["max_steps"]) // 2,
            },
            "output_schema": data_a.get("output_schema", "markdown_report_with_citations"),
        }

        child = AgentSpec.from_dict(child_data)
        return child if child.validate_spec(AVAILABLE_TOOLS) else None
    except Exception as e:
        print(f"[crossover_mutation] Error: {e}")
        return None
