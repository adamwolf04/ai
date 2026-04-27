"""
Updated judge — uses deterministic verifiable_scorer for evolution.
LLM judge is preserved as an optional qualitative mode (--qualitative flag).
"""
import os
import json
from openai import OpenAI
from evaluator.verifiable_scorer import score_run_result


def evaluate_run(task: dict, run_result: dict, qualitative: bool = False,
                 model: str = "gpt-4o-mini") -> dict:
    """
    Primary evaluation path: fully deterministic, no LLM.

    Args:
        task         : Task dict with 'question', 'type', 'answers', etc.
        run_result   : RunResult dict from agent_runtime.
        qualitative  : If True, additionally run LLM judge and attach sub-scores.
        model        : LLM model name (used only when qualitative=True).

    Returns:
        {
          "score":    float 0.0-1.0,
          "matched":  bool,
          "failures": List[str],
          "trace": {
            "cause_code":    str,
            "extracted":     str,
            "llm_scores":    dict | None   (populated only if qualitative=True)
          }
        }
    """
    # --- Deterministic scoring (always runs) ---
    result = score_run_result(run_result, task)

    trace = {
        "cause_code": result["cause_code"],
        "extracted":  result["extracted"],
        "llm_scores": None,
    }

    # --- Optional LLM qualitative judge ---
    if qualitative and run_result.get("status") != "error":
        report = run_result.get("report", "")
        question = task.get("question", "")
        llm_scores = _run_llm_judge(question, report, model)
        trace["llm_scores"] = llm_scores

    return {
        "score":    result["score"],
        "matched":  result["matched"],
        "failures": result["failures"],
        "trace":    trace,
    }


def _run_llm_judge(question: str, report: str, model: str) -> dict:
    """
    Optional qualitative LLM judge. Called only in --qualitative mode.
    Never used during evolution.
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "dummy"))
    prompt = (
        f'Evaluate this research report for the task: "{question}"\n\n'
        f"Report:\n{report}\n\n"
        "Rate from 1–5 on: completeness, accuracy, citations, coherence.\n"
        'Output JSON only: {"completeness":4,"accuracy":5,"citations":2,"coherence":4,"feedback":"..."}'
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.2
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {"completeness": 0, "accuracy": 0, "citations": 0,
                "coherence": 0, "feedback": f"Judge error: {e}"}
