"""
Main entry point for Stem Agent Evolution.
Satisfies Rule #0 (Naming), Rule #1 (Decomposition), and Rule #3 (DRY).
"""
import argparse
import asyncio
import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv() # Load API keys from .env file
from spec import AgentSpec
from stem_agent import StemAgent
from evaluator.batch import evaluate_spec_on_tasks

# Rule #2: Global Constants
BASELINE_SPEC = AgentSpec(
    id="baseline",
    system_prompt="You are a helpful research assistant. Answer the user's question accurately.",
    tools=[], # Zero-shot baseline
    planning_strategy="react"
)


def _load_tasks(path: str) -> List[Dict[str, Any]]:
    """Rule #3: DRY task loading."""
    if not os.path.exists(path):
        print(f"Warning: Task file not found: {path}")
        return []
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _print_comparison_table(baseline: Dict[str, Any], evolved: Dict[str, Any]):
    """Rule #1: Decomposition of UI logic."""
    print("\n" + "=" * 60)
    print("           BEFORE / AFTER COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Evolved':>10}")
    print("-" * 60)
    
    total = len(baseline["per_task"])
    print(f"{'Test Avg Score':<30} {baseline['avg_score']:>10.3f} {evolved['avg_score']:>10.3f}")
    print(f"{'Exact Match Rate':<30} {baseline['match_rate']:>10.1%} {evolved['match_rate']:>10.1%}")
    print("=" * 60)

    print("\nPer-task breakdown (evolved):")
    print(f"  {'Task ID':<30} {'Score':>6} {'Match':>6} {'Cause'}")
    print(f"  {'-'*55}")
    for r in evolved["per_task"]:
        matched_str = 'Y' if r['matched'] else 'N'
        print(f"  {r['task_id'][:30]:<30} {r['score']:>6.2f} {matched_str:>6}  {r['cause']}")


async def main_async(args):
    """Main async orchestrator (Rule #1)."""
    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # --- Quick mode: override in-memory (Rule #2) ---
    if args.quick:
        cfg["evolution"]["generations"] = 3
        if os.path.exists("tasks/train_fast.jsonl"):
            cfg["data"]["train_tasks"] = "tasks/train_fast.jsonl"
        print("[QUICK MODE] 3 generations, fast task set.\n")

    test_tasks = _load_tasks(cfg["data"]["test_tasks"])
    agent_model = cfg["runtime"]["agent_model"]

    # --- Baseline Evaluation (Rule #3: Using Batch Evaluator) ---
    print("1. Running Baseline Spec on test set...")
    baseline_results = await evaluate_spec_on_tasks(
        BASELINE_SPEC, 
        test_tasks, 
        model=agent_model
    )
    print(f"   Baseline avg score: {baseline_results['avg_score']:.3f}")

    # --- Evolution ---
    print("\n2. Starting Stem Agent Evolution Loop...")
    stem = StemAgent(config_override=cfg)
    best_spec = await stem.run_evolution()

    if not best_spec:
        print("Evolution failed to produce a valid spec.")
        return

    # --- Final Evaluation (Rule #3) ---
    print(f"\n3. Evaluating Evolved Spec ({best_spec.id}) on held-out test set...")
    evolved_results = await evaluate_spec_on_tasks(
        best_spec, 
        test_tasks, 
        model=agent_model
    )

    _print_comparison_table(baseline_results, evolved_results)


def main():
    """CLI entry point (Rule #0)."""
    parser = argparse.ArgumentParser(description="Stem Agent Evolution Launcher")
    parser.add_argument("--fresh", action="store_true", help="Start from a clean slate (wipe DB)")
    parser.add_argument("--resume", action="store_true", help="Resume from previous lineage DB")
    parser.add_argument("--quick", action="store_true", help="Fast run (fewer generations/tasks)")
    
    args = parser.parse_args()

    # --- Fresh mode logic (Rule #1) ---
    if args.fresh:
        db_path = "logs/lineage.db"
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"[FRESH] Wiped lineage DB: {db_path}")
        print("[FRESH] Starting from scratch.\n")

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
