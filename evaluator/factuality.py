"""
factuality.py — Demoted to optional post-hoc check.

This module is NO LONGER called during evolution.
It is available only for manual investigation or future post-run analysis.

During evolution, the deterministic verifiable_scorer is used instead.
"""
import re
import os
from openai import OpenAI
from runtime.tool_belt import scrape_page


def check_factuality(report: str, model: str = "gpt-4o-mini",
                     bypass: bool = False) -> float:
    """
    Optional LLM-backed factuality check.

    Args:
        report  : Agent's text output.
        model   : LLM model to use.
        bypass  : If True, return 1.0 immediately (used during evolution).

    Returns:
        float 0.0-1.0.
    """
    if bypass:
        return 1.0

    if not os.environ.get("OPENAI_API_KEY"):
        return 1.0  # No key → skip

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    urls   = re.findall(r'(https?://\S+)', report)
    urls   = [u.rstrip(').,"]') for u in urls]

    if not urls:
        return 0.0

    supported = 0
    checks    = min(len(urls), 1)

    for url in urls[:checks]:
        content = scrape_page(url)
        if "Error" in content:
            continue
        prompt = (
            f"Report: {report[:1000]}...\n\n"
            f"Source: {content[:2000]}...\n\n"
            "Does the source support the claims? Answer YES or NO."
        )
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            ans = resp.choices[0].message.content.strip().upper()
            if "YES" in ans:
                supported += 1
        except Exception:
            pass

    return supported / checks if checks > 0 else 0.0
