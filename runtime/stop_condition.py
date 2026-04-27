from spec import StopConditionSpec
from typing import Tuple


CAUSE_OK                = "OK"
CAUSE_REPORT_TOO_SHORT  = "REPORT_TOO_SHORT"
CAUSE_MISSING_CITATION  = "MISSING_CITATION"


def evaluate_stop_condition(
    report: str, condition: StopConditionSpec
) -> Tuple[bool, str, str]:
    """
    Check whether a report satisfies the agent's stop conditions.

    Returns:
        (is_valid: bool, error_message: str, cause_code: str)
    """
    if len(report) < condition.min_report_length:
        msg = (
            f"Report is too short ({len(report)} chars); "
            f"minimum is {condition.min_report_length}."
        )
        return False, msg, CAUSE_REPORT_TOO_SHORT

    if condition.must_include_citations:
        if "http://" not in report and "https://" not in report:
            return False, "Report is missing citations (no URLs found).", CAUSE_MISSING_CITATION

    return True, "", CAUSE_OK
