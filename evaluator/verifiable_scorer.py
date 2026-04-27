"""
Verifiable Scorer — Deterministic, LLM-free evaluation.
"""
import re
from typing import Dict, Any, List, Optional
from evaluator.constants import (
    CAUSE_OK,
    CAUSE_WRONG_VALUE,
    CAUSE_ANSWER_NOT_FOUND,
    CAUSE_CRASH,
    STATUS_SUCCESS
)


def _normalize_commas(text: str) -> str:
    """Removes commas from numbers: 1,500,000 -> 1500000."""
    return re.sub(r'(\d),(\d)', r'\1\2', text)


def _normalize_sci_notation(text: str) -> str:
    """
    Normalizes textbook scientific notation to Python e-notation:
    '6.626 x 10^-34' -> '6.626e-34'
    """
    pattern = r'([-+]?\d*\.?\d+)\s*[x×\*]\s*10\^([+-]?\d+)'
    return re.sub(pattern, lambda m: f"{m.group(1)}e{m.group(2)}", text)


def _extract_numbers(text: str) -> List[float]:
    """Extracts all floats from text, handling commas and scientific notation."""
    text = _normalize_sci_notation(text)
    text = _normalize_commas(text)
    pattern = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?'
    
    found_numbers = []
    for match in re.findall(pattern, text):
        try:
            found_numbers.append(float(match))
        except ValueError:
            continue
    return found_numbers


def _score_numeric(output: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """Logic for numeric and factoid tasks (which are treated as exact numeric)."""
    target_answers = task["answers"]
    tolerance = task.get("tolerance", 0.0)
    found_values = _extract_numbers(output)

    if not found_values:
        return {
            "score": 0.0,
            "matched": False,
            "extracted": "None",
            "cause_code": CAUSE_ANSWER_NOT_FOUND
        }

    # Design Decision: Use the *last* number mentioned as the final answer.
    # While iterating over all numbers and picking the closest is an option,
    # relying on the last number enforces that the agent's final conclusion
    # must be explicitly correct, rather than coincidentally matching intermediate math.
    final_val = found_values[-1]
    
    for ans_str in target_answers:
        target_val = float(ans_str)
        diff = abs(final_val - target_val)
        
        # Determine match based on tolerance
        is_match = False
        if target_val == 0:
            is_match = diff <= tolerance
        else:
            is_match = (diff / abs(target_val)) <= tolerance

        if is_match:
            return {
                "score": 1.0, 
                "matched": True, 
                "extracted": str(final_val), 
                "cause_code": CAUSE_OK
            }

    return {
        "score": 0.0, 
        "matched": False, 
        "extracted": str(final_val), 
        "cause_code": CAUSE_WRONG_VALUE
    }


def _score_set(output: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Jaccard similarity for unordered sets of answers.
    Note: Requires exact substring match after lowercasing. Partial matches 
    (e.g., 'apples' vs 'apple') will fail. This strictness must be accounted for 
    when curating verifiable task sets.
    """
    target_answers = [a.lower().strip() for a in task["answers"]]
    output_lower = output.lower()
    
    found_count = 0
    matched_elements = []
    for ans in target_answers:
        if ans in output_lower:
            found_count += 1
            matched_elements.append(ans)
    
    if not target_answers:
        return {"score": 1.0, "matched": True, "extracted": "", "cause_code": CAUSE_OK}
        
    score = found_count / len(target_answers)
    return {
        "score": score,
        "matched": score >= 1.0,
        "extracted": ", ".join(matched_elements),
        "cause_code": CAUSE_OK if score >= 1.0 else CAUSE_WRONG_VALUE
    }


def compute_verifiable_score(output: str, task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Primary entry point for deterministic scoring.
    Satisfies Rule #1 (Decomposition) by delegating to specialized scorers.
    """
    task_type = task.get("type", "factoid")
    
    try:
        if task_type in ("numeric", "factoid"):
            return _score_numeric(output, task)
        elif task_type in ("set_comparison", "list_comparison"):
            return _score_set(output, task)
        
        # Fallback for unknown types: simple string matching
        output_lower = output.lower()
        for ans in task["answers"]:
            if ans.lower() in output_lower:
                return {"score": 1.0, "matched": True, "extracted": ans, "cause_code": CAUSE_OK}
        
        return {
            "score": 0.0, 
            "matched": False, 
            "extracted": "None", 
            "cause_code": CAUSE_ANSWER_NOT_FOUND
        }
    except Exception as e:
        return {
            "score": 0.0, 
            "matched": False, 
            "extracted": f"Error: {e}", 
            "cause_code": CAUSE_CRASH
        }


def score_run_result(run_result: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wraps compute_verifiable_score with metadata about the run status.
    Satisfies Rule #3 (DRY) by centralizing run-result scoring logic.
    """
    status = run_result.get("status", "error")
    report = run_result.get("report", "")
    
    if status != STATUS_SUCCESS:
        return {
            "score": 0.0,
            "matched": False,
            "failures": [f"CRITICAL: Run status was {status}. Error: {run_result.get('error', 'None')}"],
            "trace": {"cause_code": CAUSE_CRASH}
        }
    
    eval_data = compute_verifiable_score(report, task)
    
    failures = []
    if not eval_data["matched"]:
        failures.append(f"Verification failed ({eval_data['cause_code']}): Extracted '{eval_data['extracted']}'")
        
    return {
        "score": eval_data["score"],
        "matched": eval_data["matched"],
        "failures": failures,
        "trace": eval_data
    }
