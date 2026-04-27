"""
AgentRuntime — async-capable, token-budget-enforced, plan-tracked agent executor.

Key improvements over prototype:
  - Fully async: run() is a coroutine; sync wrapper run_sync() available for tests
  - Token budget: halts if cumulative token usage exceeds config limit
  - plan_and_solve: parses plan into steps, injects [Step N/M] context at each turn,
    checks that the agent cites its current step after every tool call
  - Structured tool output: scrape results summarised before appending to context
  - OpenAI retry with exponential backoff (tenacity)
  - Returns RunResult Pydantic model with full trace
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Optional

from openai import AsyncOpenAI, APIError
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

from spec import AgentSpec
from runtime.tool_belt import AsyncToolBelt, AVAILABLE_TOOLS
from runtime.stop_condition import evaluate_stop_condition, CAUSE_OK
from interfaces import RunResult


TOKEN_BUDGET_DEFAULT = int(os.environ.get("TOKEN_BUDGET", "8000"))

_OPENAI_RETRY = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(APIError),
    reraise=True,
)


def _summarise_tool_output(name: str, raw: str) -> str:
    """Trim tool output to save context tokens."""
    if name == "scrape_page":
        return raw[:1000] + ("\n[content truncated]" if len(raw) > 1000 else "")
    if name == "search_web":
        return raw[:2000] + ("\n[truncated]" if len(raw) > 2000 else "")
    return raw[:2000]


def _parse_plan_steps(plan_text: str) -> list[str]:
    """
    Extract numbered steps from a plan string.
    Accepts "1. Do X", "Step 1: Do X", "1) Do X" formats.
    """
    import re
    steps = re.findall(
        r'(?:step\s*)?(\d+)[.):\s]+(.+?)(?=\n(?:step\s*)?\d+[.):\s]|\Z)',
        plan_text, re.IGNORECASE | re.DOTALL
    )
    if steps:
        return [s.strip() for _, s in steps]
    # Fallback: split on newlines
    return [ln.strip() for ln in plan_text.splitlines() if ln.strip()]


class ResearchRunner:
    """
    Async agent runner. Implements AgentRuntime interface (informally).
    One instance per spec; create fresh per generation for clean state.
    """

    def __init__(self, spec: AgentSpec, model: str = "gpt-4o-mini",
                 token_budget: int = TOKEN_BUDGET_DEFAULT):
        self.spec         = spec
        self.model        = model
        self.token_budget = token_budget
        self.tools        = AsyncToolBelt(spec.tools)
        self.client       = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "dummy")
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, task: str) -> RunResult:
        """Run the agent on a task string. Returns RunResult."""
        try:
            # We use a separate client if needed, or wrap the existing one
            # The warnings come from many small clients being created and not closed.
            return await self._run_inner(task)
        except Exception as e:
            return RunResult(status="error", error=str(e), report=f"Runtime error: {e}")

    async def close(self):
        """Close the underlying HTTP client."""
        await self.client.close()

    def run_sync(self, task: str) -> RunResult:
        """Synchronous wrapper for environments without an event loop."""
        return asyncio.run(self.run(task))

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _run_inner(self, task: str) -> RunResult:
        system_prompt, plan_steps = await self._build_system_prompt(task)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": (
                f"Task: {task}\n\n"
                "Please research this topic thoroughly and provide a comprehensive answer."
            )}
        ]

        step         = 0
        error_count  = 0
        token_count  = 0
        max_steps    = self.spec.stop_condition.max_steps
        current_plan_step = 0

        while step < max_steps:
            step += 1

            # Inject plan step reminder for plan_and_solve
            if plan_steps and current_plan_step < len(plan_steps):
                messages.append({
                    "role": "user",
                    "content": (
                        f"[Step {current_plan_step + 1} of {len(plan_steps)}]: "
                        f"{plan_steps[current_plan_step]}"
                    )
                })

            schemas = self.tools.get_schemas()
            kwargs  = {}
            if schemas:
                kwargs["tools"]       = schemas
                kwargs["tool_choice"] = "auto"

            # --- Retried OpenAI call ---
            try:
                response = await self._chat_with_retry(messages, **kwargs)
            except Exception as e:
                return RunResult(
                    status="error", error=str(e),
                    report=f"API Error after retries: {e}",
                    steps=step, token_count=token_count, trace=messages
                )

            # Token budget tracking
            if response.usage:
                token_count += response.usage.total_tokens
                if token_count >= self.token_budget:
                    report = self._extract_last_content(messages)
                    return RunResult(
                        report=report, steps=step, status="token_budget_exceeded",
                        token_count=token_count, trace=messages
                    )

            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            # --- Tool calls ---
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.function.name
                    try:
                        args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    raw_result     = await self.tools.execute_async(name, args)
                    tool_summary   = _summarise_tool_output(name, raw_result)

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "name":         name,
                        "content":      tool_summary,
                    })

                # --- Plan step tracking for plan_and_solve ---
                if plan_steps and self.spec.planning_strategy == "plan_and_solve":
                    check_resp = await self._chat_with_retry([
                        *messages,
                        {"role": "user",
                         "content": "Which step number of the plan are you currently executing? "
                                    "Reply with only the step number (e.g. '2')."}
                    ])
                    step_reply = check_resp.choices[0].message.content.strip()
                    import re
                    nums = re.findall(r'\d+', step_reply)
                    if nums:
                        current_plan_step = min(int(nums[0]) - 1, len(plan_steps) - 1)
                    else:
                        # Agent couldn't cite step — flag but continue
                        messages.append({
                            "role":    "user",
                            "content": "Warning: Please indicate which plan step you are on."
                        })

            else:
                # --- Final text output ---
                report = msg.content or ""
                is_valid, err_msg, cause_code = evaluate_stop_condition(
                    report, self.spec.stop_condition
                )

                if not is_valid:
                    error_count += 1
                    if error_count > 3:
                        return RunResult(
                            report=report, steps=step,
                            status="stopped_with_errors",
                            error=f"{cause_code}: {err_msg}",
                            token_count=token_count, trace=messages
                        )
                    messages.append({
                        "role":    "user",
                        "content": f"Your output was rejected ({cause_code}): {err_msg}\n"
                                   "Please fix it and provide an updated answer."
                    })
                    continue

                return RunResult(
                    report=report, steps=step, status="success",
                    token_count=token_count, trace=messages
                )

        report = self._extract_last_content(messages)
        return RunResult(
            report=report, steps=step, status="max_steps_reached",
            token_count=token_count, trace=messages
        )

    async def _build_system_prompt(self, task: str) -> tuple[str, list[str]]:
        """
        Build the effective system prompt.
        For plan_and_solve: generate a numbered plan, return (augmented_prompt, steps).
        For react: return (original_prompt, []).
        """
        if self.spec.planning_strategy != "plan_and_solve":
            return self.spec.system_prompt, []

        try:
            plan_resp = await self._chat_with_retry([
                {"role": "system",
                 "content": "You are a research planner. Create a concise numbered step-by-step plan."},
                {"role": "user",
                 "content": f"Research task: {task}"}
            ], max_tokens=400)
            plan_text  = plan_resp.choices[0].message.content or ""
            plan_steps = _parse_plan_steps(plan_text)
            augmented  = (
                f"{self.spec.system_prompt}\n\n"
                f"[Research Plan]\n{plan_text}\n[End Plan]"
            )
            return augmented, plan_steps
        except Exception:
            return self.spec.system_prompt, []

    @_OPENAI_RETRY
    async def _chat_with_retry(self, messages: list, **kwargs) -> object:
        """OpenAI chat call with exponential-backoff retry on API errors."""
        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

    @staticmethod
    def _extract_last_content(messages: list) -> str:
        for msg in reversed(messages):
            content = msg.get("content") if isinstance(msg, dict) else None
            if content:
                return content
        return "No output produced."
