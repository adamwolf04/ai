"""
Stem Agent — main entry point.

Flags:
  --quick        Run only 3 generations on train_fast.jsonl (smoke test)
  --benchmark    Evaluate the best evolved spec on the held-out test set
  --qualitative  Re-run evolved spec on 5 open-ended tasks with LLM judge output
"""
import argparse
import asyncio
import json
import os

from dotenv import load_dotenv
from stem_agent import StemAgent
from spec import AgentSpec, StopConditionSpec
from runtime.agent_runtime import ResearchRunner
from evaluator.judge import evaluate_run

load_dotenv()


# ---------------------------------------------------------------------------
# Baseline spec (no tools, minimal prompt)
# ---------------------------------------------------------------------------

BASELINE_SPEC = AgentSpec(
    id="baseline_v1",
    system_prompt="You are a helpful assistant. Answer the user's question using your own knowledge.",
    tools=[],
    planning_strategy="react",
    stop_condition=StopConditionSpec(
        min_report_length=50,
        must_include_citations=False,
        max_steps=3
    )
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tasks(path: str) -> list:
    if not path or not os.path.exists(path):
        return []
    tasks = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    return tasks


async def run_evaluation(spec: AgentSpec, tasks: list,
                         model: str = "gpt-4o-mini",
                         qualitative: bool = False) -> dict:
    """
    Run spec on all tasks concurrently. Returns {avg_score, per_task}.
    """
    if not tasks:
        return {"avg_score": 0.0, "per_task": []}

    runner  = ResearchRunner(spec, model=model)
    sem     = asyncio.Semaphore(8)
    results = []

    async def run_one(task):
        async with sem:
            try:
                rr = await asyncio.wait_for(runner.run(task["question"]), timeout=120)
            except asyncio.TimeoutError:
                rr_dict = {"status": "error", "report": "", "steps": 0, "token_count": 0}
            else:
                rr_dict = rr.to_dict() if hasattr(rr, "to_dict") else rr

            ev = evaluate_run(task, rr_dict, qualitative=qualitative)
            return {
                "task_id": task.get("id", task["question"][:40]),
                "score":   ev["score"],
                "matched": ev["matched"],
                "cause":   ev["trace"].get("cause_code", ""),
            }

    results = await asyncio.gather(*[run_one(t) for t in tasks])
    avg = sum(r["score"] for r in results) / max(1, len(results))
    return {"avg_score": avg, "per_task": list(results)}


def print_comparison_table(baseline: dict, evolved: dict):
    print("\n" + "=" * 60)
    print("           BEFORE / AFTER COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<30} {'Baseline':>10} {'Evolved':>10}")
    print("-" * 60)
    print(f"{'Test Avg Score':<30} {baseline['avg_score']:>10.3f} {evolved['avg_score']:>10.3f}")
    matched_b = sum(1 for r in baseline["per_task"] if r.get("matched"))
    matched_e = sum(1 for r in evolved["per_task"] if r.get("matched"))
    total     = max(len(baseline["per_task"]), 1)
    print(f"{'Exact Match Rate':<30} {matched_b/total:>10.1%} {matched_e/total:>10.1%}")
    print("=" * 60)

    print("\nPer-task breakdown (evolved):")
    print(f"  {'Task ID':<30} {'Score':>6} {'Match':>6} {'Cause'}")
    print(f"  {'-'*55}")
    for r in evolved["per_task"]:
        matched_str = 'Y' if r['matched'] else 'N'
        print(f"  {r['task_id'][:30]:<30} {r['score']:>6.2f} {matched_str:>6}  {r['cause']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main_async(args):
    print("=== Stem Agent — Deep Research Meta-Optimization ===\n")

    # --- Fresh mode: wipe DB and start clean ---
    if args.fresh:
        import yaml
        with open("config.yaml") as f:
            cfg_fresh = yaml.safe_load(f)
        db_path = cfg_fresh["paths"].get("db_path", "logs/lineage.db")
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"[FRESH] Wiped lineage DB: {db_path}")
        # Also clear saved specs so seeding runs cleanly
        specs_dir = cfg_fresh["paths"].get("specs_dir", "specs")
        if os.path.isdir(specs_dir):
            import glob
            for f_path in glob.glob(os.path.join(specs_dir, "*.json")):
                os.remove(f_path)
        print("[FRESH] Starting from scratch.\n")
    elif args.resume:
        print("[RESUME] Loading existing population from lineage DB...\n")
    else:
        # Default behaviour: resume if DB exists, seed if not
        print("[AUTO] Resume if DB exists, seed if not.\n")

    import yaml
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    # --- Quick mode: in-memory override only, never touch config.yaml ---
    if args.quick:
        cfg["evolution"]["generations"] = 3
        if os.path.exists("tasks/train_fast.jsonl"):
            cfg["data"]["train_tasks"] = "tasks/train_fast.jsonl"
        print("[QUICK MODE] 3 generations, fast task set.\n")

    test_tasks  = load_tasks(cfg["data"]["test_tasks"])
    agent_model = cfg["runtime"]["agent_model"]

    # --- Baseline ---
    print("1. Running Baseline Spec on test set...")
    baseline_results = await run_evaluation(BASELINE_SPEC, test_tasks, model=agent_model)
    print(f"   Baseline avg score: {baseline_results['avg_score']:.3f}")

    # --- Evolution ---
    print("\n2. Starting Stem Agent Evolution Loop...")
    stem      = StemAgent(config_override=cfg)
    best_spec = await stem.run_evolution()

    if not best_spec:
        print("Evolution failed to produce a valid spec.")
        return

    print(f"\n3. Evaluating Evolved Spec ({best_spec.id}) on held-out test set...")
    evolved_results = await run_evaluation(best_spec, test_tasks, model=agent_model)

    # --- Comparison table ---
    print_comparison_table(baseline_results, evolved_results)

    # --- Benchmark mode: full verifiable test set ---
    if args.benchmark:
        print(f"\n4. [BENCHMARK] Running on full test set: {cfg['data']['test_tasks']}")
        bench_results = await run_evaluation(best_spec, test_tasks, model=agent_model)
        print(f"   Benchmark score: {bench_results['avg_score']:.3f}")
        matched = sum(1 for r in bench_results["per_task"] if r.get("matched"))
        print(f"   Exact matches:   {matched}/{len(test_tasks)}")

    # --- Qualitative mode: LLM judge on open-ended tasks ---
    if args.qualitative:
        print("\n5. [QUALITATIVE] Running evolved spec on open-ended tasks with LLM judge...")
        qual_tasks = load_tasks(cfg["evaluator"].get("qualitative_tasks", "tasks/train.jsonl"))[:5]
        for task in qual_tasks:
            runner = ResearchRunner(best_spec, model=agent_model)
            rr     = await asyncio.wait_for(runner.run(task["question"]), timeout=180)
            rr_d   = rr.to_dict() if hasattr(rr, "to_dict") else rr
            ev     = evaluate_run(task, rr_d, qualitative=True,
                                  model=cfg["evaluator"]["judge_model"])
            llm_sc = ev["trace"].get("llm_scores", {})
            print(f"\n  Q: {task['question'][:70]}")
            print(f"  LLM scores: {llm_sc}")
            print(f"  Report (first 300 chars): {rr_d.get('report','')[:300]}")


def main():
    parser = argparse.ArgumentParser(
        description="Stem Agent Evolution Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                        # Default: resume if DB exists, seed if not
  python main.py --resume               # Explicit resume from lineage DB
  python main.py --fresh                # Wipe DB and start a clean-slate evolution
  python main.py --quick                # Smoke test: 3 generations only
  python main.py --benchmark            # Evaluate on held-out test set after evolution
  python main.py --qualitative          # LLM judge on 5 open-ended research tasks
  python main.py --quick --benchmark    # Fast run then benchmark
"""
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Explicitly resume evolution from an existing lineage DB (default behaviour; alias for clarity)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Wipe the lineage DB and saved specs, then start a clean-slate evolution"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick smoke test: cap at 3 generations using the fast task set"
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="After evolution, evaluate the best spec on the held-out test set and print exact-match accuracy"
    )
    parser.add_argument(
        "--qualitative", action="store_true",
        help="After evolution, run the LLM judge on 5 open-ended research tasks and print sub-scores"
    )

    args = parser.parse_args()

    if args.fresh and args.resume:
        print("Error: --fresh and --resume are mutually exclusive.")
        raise SystemExit(1)

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
