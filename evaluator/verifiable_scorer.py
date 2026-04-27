"""
Deterministic, LLM-free verifiable scorer.

Supports task types:
  - factoid         : exact match or numeric tolerance
  - numeric         : number extraction + relative tolerance window
  - set_comparison  : Jaccard over a required answer set
  - list_comparison : intersection/union, order does NOT matter

Returns a result dict:
  score       : float 0.0-1.0
  matched     : bool
  extracted   : str
  cause_code  : str ("OK"|"ANSWER_NOT_FOUND"|"WRONG_VALUE"|"TASK_ERROR"|"UNKNOWN_TYPE")
"""
import re


def _normalize_commas(text: str) -> str:
    """1,500,000 -> 1500000"""
    return re.sub(r'(\d),(\d)', r'\1\2', text)


def _normalize_sci_notation(text: str) -> str:
    """
    Normalize textbook scientific notation to Python e-notation:
      '6.626 x 10^-34'  -> '6.626e-34'
      '6.626 × 10^-34'  -> '6.626e-34'
      '6.626 * 10^-34'  -> '6.626e-34'
      '1.5 x 10^12'     -> '1.5e12'
    """
    pattern = r'([-+]?\d*\.?\d+)\s*[x×\*]\s*10\^([+-]?\d+)'
    return re.sub(pattern, lambda m: f"{m.group(1)}e{m.group(2)}", text)


def _extract_numbers(text: str) -> list:
    text = _normalize_sci_notation(text)
    text = _normalize_commas(text)
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    results = []
    for m in re.findall(pattern, text):
        try:
            results.append(float(m))
        except ValueError:
            pass
    return results


def _within_tolerance(extracted: float, target: float, tol: float) -> bool:
    if target == 0:
        return abs(extracted) <= (tol if tol > 0 else 1e-9)
    return abs(extracted - target) / abs(target) <= tol


def compute_verifiable_score(output: str, task: dict) -> dict:
    """
    Fully deterministic score. No LLM involved.
    """
    task_type = task.get("type", "factoid")
    answers   = task.get("answers", [])
    tol       = task.get("tolerance", 0.05)

    if not answers:
        return {"score": 0.0, "matched": False, "extracted": "", "cause_code": "TASK_ERROR"}

    # ---- factoid / numeric ----
    if task_type in ("factoid", "numeric"):
        primary = answers[0].replace(",", "")
        try:
            true_val = float(primary)
            numbers  = _extract_numbers(output)
            if numbers:
                for num in numbers:
                    if _within_tolerance(num, true_val, tol):
                        return {"score": 1.0, "matched": True, "extracted": str(num), "cause_code": "OK"}
                return {"score": 0.0, "matched": False, "extracted": str(numbers[0]), "cause_code": "WRONG_VALUE"}
        except ValueError:
            pass

        output_lower = output.lower()
        for ans in answers:
            if ans.lower() in output_lower:
                return {"score": 1.0, "matched": True, "extracted": ans, "cause_code": "OK"}
        return {"score": 0.0, "matched": False, "extracted": "", "cause_code": "ANSWER_NOT_FOUND"}

    # ---- set_comparison ----
    elif task_type == "set_comparison":
        al = [a.lower() for a in answers]
        ol = output.lower()
        found = [a for a in al if a in ol]
        jaccard = len(found) / len(al) if al else 0.0
        matched = jaccard >= 0.8
        return {"score": round(jaccard, 4), "matched": matched,
                "extracted": ", ".join(found), "cause_code": "OK" if matched else "WRONG_VALUE"}

    # ---- list_comparison ----
    elif task_type == "list_comparison":
        al = {a.lower() for a in answers}
        ol = output.lower()
        found = {a for a in al if a in ol}
        jaccard = len(found) / len(al) if al else 0.0
        matched = jaccard >= 0.6
        return {"score": round(jaccard, 4), "matched": matched,
                "extracted": ", ".join(sorted(found)), "cause_code": "OK" if matched else "WRONG_VALUE"}

    return {"score": 0.0, "matched": False, "extracted": "", "cause_code": "UNKNOWN_TYPE"}


def score_run_result(run_result: dict, task: dict) -> dict:
    """
    Convenience wrapper: score a RunResult dict against a task.
    Adds 'failures' list for the mutation engine.
    """
    if run_result.get("status") == "error":
        return {"score": 0.0, "matched": False, "extracted": "", "cause_code": "CRASH",
                "failures": ["1. CRITICAL: Agent crashed with runtime exception."]}

    report  = run_result.get("report", "")
    result  = compute_verifiable_score(report, task)
    code    = result["cause_code"]
    failures = []

    if code == "ANSWER_NOT_FOUND":
        failures.append(
            f"2. HIGH: Answer not found for task '{task.get('id','?')}'. "
            f"Expected: {task['answers']}. Required tools: {task.get('required_tools',[])}."
        )
    elif code == "WRONG_VALUE":
        failures.append(
            f"3. MEDIUM: Wrong value ('{result['extracted']}') for task "
            f"'{task.get('id','?')}'. Expected: {task['answers']}."
        )
    elif code == "TASK_ERROR":
        failures.append(f"4. LOW: Task '{task.get('id','?')}' malformed.")

    if run_result.get("status") == "max_steps_reached":
        failures.append("5. LOW: Agent hit max_steps without producing a final answer.")

    result["failures"] = failures
    return result
