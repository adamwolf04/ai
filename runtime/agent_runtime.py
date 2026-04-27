"""
AgentRuntime — Async, budget-enforced agent executor.
Satisfies Rule #0 (Naming), Rule #1 (Decomposition), Rule #2 (Constants), and Rule #3 (DRY).
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

@dataclass
class RunState:
    """
    Encapsulates the session state of a single run.
    Since ResearchRunner is instantiated and executed per-task sequentially within its scope, 
    this state guarantees complete isolation between concurrent tasks.
    """
    step_count: int = 0
    error_count: int = 0
    total_tokens: int = 0
    current_plan_step: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)

from openai import AsyncOpenAI, APIError
from tenacity import (
    retry, stop_after_attempt, wait_exponential, retry_if_exception_type
)

from spec import AgentSpec
from runtime.tool_belt import AsyncToolBelt
from runtime.stop_condition import evaluate_stop_condition
from interfaces import RunResult
from evaluator.constants import (
    CAUSE_OK,
    CAUSE_TOKEN_BUDGET_EXCEEDED,
    CAUSE_MAX_STEPS_REACHED,
    CAUSE_STOPPED_WITH_ERRORS,
    CAUSE_CRASH,
    STATUS_SUCCESS,
    STATUS_ERROR
)


# Rule #2: Constants
TOKEN_BUDGET_DEFAULT = int(os.environ.get("TOKEN_BUDGET", "8000"))
MAX_RETRY_ATTEMPTS = 3
MAX_STOP_FAILURES = 3

_OPENAI_RETRY = retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    retry=retry_if_exception_type(APIError),
    reraise=True,
)


def _summarize_tool_output(name: str, raw_output: str) -> str:
    """Trims tool output to save context tokens. (Rule #5: Encapsulation)"""
    limit = 2000
    if name == "scrape_page":
        limit = 1000
        
    if len(raw_output) > limit:
        return raw_output[:limit] + "\n[Content truncated for brevity]"
    return raw_output


def _parse_plan_steps(plan_text: str) -> List[str]:
    """Extracts numbered steps from a plan string. (Rule #1: Decomposition)"""
    steps = re.findall(
        r'(?:step\s*)?(\d+)[.):\s]+(.+?)(?=\n(?:step\s*)?\d+[.):\s]|\Z)',
        plan_text, re.IGNORECASE | re.DOTALL
    )
    if steps:
        return [s.strip() for _, s in steps]
    
    # Fallback: simple split on newlines for non-numbered lists
    return [ln.strip() for ln in plan_text.splitlines() if ln.strip()]


class ResearchRunner:
    """
    Async agent runner. 
    Encapsulates the state machine for a single research execution (Rule #5).
    """

    def __init__(self, spec: AgentSpec, model: str = "gpt-4o-mini",
                 token_budget: int = TOKEN_BUDGET_DEFAULT):
        self.spec = spec
        self.model = model
        self.token_budget = token_budget
        self.tools = AsyncToolBelt(spec.tools)
        self.client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "dummy")
        )

    async def run(self, task: str) -> RunResult:
        """Entry point for agent execution."""
        try:
            return await self._execute_task(task)
        except Exception as e:
            return RunResult(status=STATUS_ERROR, error=str(e), report=f"Runtime error: {e}", steps=0, token_count=0, trace=[])

    async def close(self):
        """Cleanly shutdown the HTTP client."""
        await self.client.close()


    async def _execute_task(self, task: str) -> RunResult:
        """Internal execution loop. Decomposed for clarity. (Rule #1)"""
        system_prompt, plan_steps = await self._initialize_session(task)
        
        state = RunState(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": f"Task: {task}\n\nPerform deep research and answer accurately."}
            ]
        )

        while state.step_count < self.spec.stop_condition.max_steps:
            state.step_count += 1
            self._inject_plan_step(plan_steps, state)

            response = await self._call_llm(state.messages)
            
            if self._update_and_check_budget(response, state):
                return self._build_result(state, CAUSE_TOKEN_BUDGET_EXCEEDED)

            assistant_msg = response.choices[0].message
            state.messages.append(assistant_msg.model_dump(exclude_unset=True))

            if assistant_msg.tool_calls:
                await self._process_tool_calls(assistant_msg.tool_calls, plan_steps, state)
            else:
                final_result = self._process_final_response(assistant_msg.content or "", state)
                if final_result:
                    return final_result

        return self._build_result(state, CAUSE_MAX_STEPS_REACHED)

    def _inject_plan_step(self, plan_steps: List[str], state: RunState):
        """Injects the current plan step into the prompt if planning is active."""
        if plan_steps and state.current_plan_step < len(plan_steps):
            state.messages.append({
                "role": "user",
                "content": f"[Plan Step {state.current_plan_step + 1}/{len(plan_steps)}]: {plan_steps[state.current_plan_step]}"
            })

    def _update_and_check_budget(self, response: Any, state: RunState) -> bool:
        """Updates token count and checks against the budget limit."""
        if response.usage:
            state.total_tokens += response.usage.total_tokens
        return state.total_tokens >= self.token_budget

    async def _process_tool_calls(self, tool_calls: List[Any], plan_steps: List[str], state: RunState):
        """Executes tools and updates the agent's progress along its plan."""
        await self._handle_tool_calls(tool_calls, state.messages)
        if plan_steps and self.spec.planning_strategy == "plan_and_solve":
            state.current_plan_step = await self._track_plan_progress(state.messages, len(plan_steps))

    def _process_final_response(self, report: str, state: RunState) -> Optional[RunResult]:
        """Evaluates whether the agent's final report meets stop conditions."""
        is_valid, err_msg, cause = evaluate_stop_condition(report, self.spec.stop_condition)

        if not is_valid:
            state.error_count += 1
            if state.error_count > MAX_STOP_FAILURES:
                return self._build_result(state, CAUSE_STOPPED_WITH_ERRORS, error=f"{cause}: {err_msg}", report=report)
            
            state.messages.append({"role": "user", "content": f"REJECTED ({cause}): {err_msg}. Please fix."})
            return None

        return self._build_result(state, STATUS_SUCCESS, report=report)

    def _build_result(self, state: RunState, status: str, error: Optional[str] = None, report: Optional[str] = None) -> RunResult:
        """Standardized helper to construct RunResult objects."""
        return RunResult(
            report=report if report is not None else self._get_last_report(state.messages),
            steps=state.step_count,
            status=status,
            error=error,
            token_count=state.total_tokens,
            trace=state.messages
        )

    async def _initialize_session(self, task: str) -> Tuple[str, List[str]]:
        """Prepares the system prompt and plan if necessary. (Rule #1)"""
        if self.spec.planning_strategy != "plan_and_solve":
            return self.spec.system_prompt, []

        try:
            resp = await self._call_llm([
                {"role": "system", "content": "Create a numbered research plan for the following task."},
                {"role": "user", "content": task}
            ], max_tokens=400)
            plan_text = resp.choices[0].message.content or ""
            steps = _parse_plan_steps(plan_text)
            augmented_prompt = f"{self.spec.system_prompt}\n\n[PLAN]\n{plan_text}\n[END PLAN]"
            return augmented_prompt, steps
        except Exception:
            return self.spec.system_prompt, []

    @_OPENAI_RETRY
    async def _call_llm(self, messages: List[Dict[str, Any]], **kwargs) -> Any:
        """Internal wrapper for OpenAI chat completions. (Rule #3: DRY)"""
        # Rule #2: Tools schemas are injected here if allowed
        tool_schemas = self.tools.get_schemas()
        if tool_schemas and "tools" not in kwargs:
            kwargs["tools"] = tool_schemas
            kwargs["tool_choice"] = "auto"

        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )

    async def _handle_tool_calls(self, tool_calls: List[Any], messages: List[Dict[str, Any]]):
        """Executes multiple tool calls and appends results. (Rule #1)"""
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            raw_result = await self.tools.execute_async(tc.function.name, args)
            summary = _summarize_tool_output(tc.function.name, raw_result)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.function.name,
                "content": summary,
            })

    async def _track_plan_progress(self, messages: List[Dict[str, Any]], total_steps: int) -> int:
        """Asks the agent which plan step it is currently on. (Rule #5)"""
        try:
            resp = await self._call_llm([
                *messages,
                {"role": "user", "content": "State only the step number you are currently executing (e.g. '2')."}
            ])
            reply = resp.choices[0].message.content or ""
            match = re.search(r'\d+', reply)
            if match:
                return min(int(match.group()) - 1, total_steps - 1)
        except Exception:
            pass
        return 0

    @staticmethod
    def _get_last_report(messages: List[Dict[str, Any]]) -> str:
        """Retrieves the last non-empty assistant message. (Rule #5)"""
        for msg in reversed(messages):
            content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
            if content:
                return content
        return "Process terminated without report."
