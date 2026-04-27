"""
Stop Condition Evaluation — Logic to determine if an agent should stop researching.
Satisfies Rule #0 (Consistent Naming), Rule #2 (Constants), and Rule #5 (Encapsulation).
"""
from typing import Tuple
from spec import StopConditionSpec
from evaluator.constants import (
    CAUSE_OK,
    CAUSE_REPORT_TOO_SHORT,
    CAUSE_MISSING_CITATION
)


def evaluate_stop_condition(
    report: str, 
    condition: StopConditionSpec
) -> Tuple[bool, str, str]:
    """
    Checks whether a report satisfies the agent's defined stop conditions.

    Returns:
        (is_valid: bool, error_message: str, cause_code: str)
    """
    
    # Rule #1: Decomposition - Check length
    if len(report) < condition.min_report_length:
        msg = (
            f"Report is too short ({len(report)} chars); "
            f"minimum required is {condition.min_report_length}."
        )
        return False, msg, CAUSE_REPORT_TOO_SHORT

    # Rule #1: Decomposition - Check citations
    if condition.must_include_citations:
        if "http://" not in report and "https://" not in report:
            msg = "Report is missing citations (no valid URLs found)."
            return False, msg, CAUSE_MISSING_CITATION

    return True, "", CAUSE_OK
