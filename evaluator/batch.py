"""
Batch Evaluator — Reusable logic for running a spec against multiple tasks.
Satisfies Rule #3 (DRY) and Rule #1 (Decomposition).
"""
import asyncio
from typing import List, Dict, Any, Optional, Callable
from runtime.agent_runtime import ResearchRunner
from evaluator.verifiable_scorer import score_run_result


async def evaluate_spec_on_tasks(
    spec: Any,
    tasks: List[Dict[str, Any]],
    model: str = "gpt-4o-mini",
    token_budget: int = 8000,
    timeout: int = 120,
    concurrency: int = 8,
    on_result_callback: Optional[Callable[[str, Dict[str, Any], Dict[str, Any], Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Core logic to evaluate an AgentSpec against a collection of tasks.
    Used by both the evolution engine and the benchmarking script.
    """
    runner = ResearchRunner(spec, model=model, token_budget=token_budget)
    sem = asyncio.Semaphore(concurrency)
    
    total_score = 0.0
    matched_count = 0
    per_task_results = []
    all_failures = []

    async def _evaluate_single(task: Dict[str, Any]):
        nonlocal total_score, matched_count
        async with sem:
            try:
                run_result = await asyncio.wait_for(
                    runner.run(task["question"]),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                run_result = {"status": "error", "error": "Timeout", "report": "", "steps": 0, "token_count": 0, "trace": []}
            
            rr_dict = run_result.to_dict() if hasattr(run_result, "to_dict") else run_result
            eval_res = score_run_result(rr_dict, task)
            
            task_id = task.get("id", task["question"][:30])
            total_score += eval_res["score"]
            if eval_res["matched"]:
                matched_count += 1
            
            all_failures.extend(eval_res["failures"])
            
            task_entry = {
                "task_id": task_id,
                "score": eval_res["score"],
                "matched": eval_res["matched"],
                "cause": eval_res["trace"].get("cause_code", "UNKNOWN")
            }
            per_task_results.append(task_entry)
            
            if on_result_callback:
                on_result_callback(task_id, task, eval_res, rr_dict)

    # Run all tasks
    try:
        await asyncio.gather(*[_evaluate_single(t) for t in tasks])
    finally:
        await runner.close()

    count = max(1, len(tasks))
    return {
        "avg_score": total_score / count,
        "match_rate": matched_count / count,
        "per_task": per_task_results,
        "failures": sorted(list(dict.fromkeys(all_failures)))
    }
