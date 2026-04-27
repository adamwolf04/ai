"""
StemAgent — async evolution orchestrator.

Improvements:
  - asyncio.gather evaluates all (spec, task) pairs concurrently
  - asyncio.Semaphore limits concurrency to avoid API rate limits
  - asyncio.wait_for enforces per-task timeout
  - LineageDB stores every evaluation and run trace
  - Graduation threshold: evolution stops when best score >= threshold
  - Crossover operator added to the mutation pipeline
  - Rich mutation prompts include actual failure examples from LineageDB
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import yaml

from evolution.population import PopulationManager
from evolution.operators import (
    seed_population, llm_mutation, random_mutation, crossover_mutation
)
from runtime.agent_runtime import ResearchRunner
from evaluator.judge import evaluate_run
from db.lineage import LineageDB
from spec import AgentSpec


class StemAgent:

    def __init__(self, config_path: str = "config.yaml",
                 config_override: dict | None = None):
        if config_override is not None:
            self.config = config_override
        else:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        self.db = LineageDB(self.config["paths"].get("db_path", "logs/lineage.db"))

        self.pop_manager = PopulationManager(
            specs_dir=self.config["paths"]["specs_dir"],
            db=self.db,
        )

        os.makedirs(self.config["paths"]["logs_dir"], exist_ok=True)

        self.train_tasks      = self._load_tasks(self.config["data"]["train_tasks"])
        self.regression_tasks = self._load_tasks(self.config["data"].get("regression_tasks", ""))

        self.concurrency          = self.config["evolution"].get("concurrency", 8)
        self.task_timeout         = self.config["evolution"].get("task_timeout_s", 120)
        self.graduation_threshold = self.config["evolution"].get("graduation_threshold", 0.85)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_evolution(self) -> AgentSpec | None:
        """Async entry point — await this from an existing event loop."""
        return await self._async_evolution()

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    async def _async_evolution(self) -> AgentSpec | None:
        cfg_ev = self.config["evolution"]

        # Resume from DB if population already exists
        self.pop_manager.restore_from_db()
        start_gen = self.db.get_last_generation()

        if not self.pop_manager.population:
            print("Seeding population...")
            seeds = seed_population(cfg_ev["mutation_model"])
            self.pop_manager.add_specs(seeds)

        best_ever_score  = -1.0
        patience_counter = 0

        for generation in range(start_gen + 1, start_gen + cfg_ev["generations"] + 1):
            print(f"\n--- Generation {generation} ---")

            # --- Evaluate unevaluated specs concurrently ---
            unevaluated = [
                s for s in self.pop_manager.population
                if s.id not in self.pop_manager.scores
            ]
            if unevaluated:
                await self._evaluate_population(unevaluated, generation)

            # --- Check plateau / graduation ---
            current_best = self.pop_manager.get_best()
            if not current_best:
                break

            current_best_score = self.pop_manager.scores.get(current_best.id, 0.0)
            print(f"Best: {current_best.id}  score={current_best_score:.3f}")

            self.db.log_generation(generation, current_best.id, current_best_score)

            if current_best_score >= self.graduation_threshold:
                print(f"Graduated! Score {current_best_score:.3f} >= {self.graduation_threshold}")
                break

            if current_best_score > best_ever_score + 0.05:
                best_ever_score  = current_best_score
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= cfg_ev["patience"]:
                print(f"Early stop: no improvement for {cfg_ev['patience']} generations.")
                break

            # --- Selection & mutation ---
            await self._produce_next_generation(generation, cfg_ev)

        print("\nEvolution complete.")
        self.db.flush()
        return self.pop_manager.get_best()

    # ------------------------------------------------------------------
    # Concurrent evaluation
    # ------------------------------------------------------------------

    async def _evaluate_population(self, specs: list[AgentSpec], generation: int):
        sem = asyncio.Semaphore(self.concurrency)

        async def eval_spec(spec: AgentSpec):
            total_score  = 0.0
            all_failures = []
            runner       = ResearchRunner(
                spec,
                model=self.config["runtime"]["agent_model"],
                token_budget=self.config["runtime"].get("token_budget", 8000)
            )

            print(f"  Evaluating {spec.id}...")
            for task in self.train_tasks:
                async with sem:
                    try:
                        run_result = await asyncio.wait_for(
                            runner.run(task["question"]),
                            timeout=self.task_timeout
                        )
                    except asyncio.TimeoutError:
                        run_result = {"status": "error", "report": "", "steps": 0,
                                      "token_count": 0, "trace": []}

                    # Convert RunResult pydantic → dict if needed
                    rr_dict = (
                        run_result.to_dict()
                        if hasattr(run_result, "to_dict")
                        else run_result
                    )

                    eval_res = evaluate_run(task, rr_dict)
                    total_score  += eval_res["score"]
                    all_failures.extend(eval_res["failures"])

                    # Log to DB
                    self.db.log_evaluation(
                        spec_id=spec.id,
                        task_id=task.get("id", task["question"][:30]),
                        score=eval_res["score"],
                        matched=eval_res["matched"],
                        extracted=eval_res["trace"].get("extracted", ""),
                        cause_code=eval_res["trace"].get("cause_code", ""),
                    )
                    self.db.log_run_trace(
                        spec_id=spec.id,
                        task_id=task.get("id", task["question"][:30]),
                        messages=rr_dict.get("trace", []),
                        token_count=rr_dict.get("token_count", 0),
                        status=rr_dict.get("status", ""),
                    )

            await runner.close()

            avg_score    = total_score / max(1, len(self.train_tasks))
            all_failures = list(dict.fromkeys(all_failures))
            all_failures.sort()

            self.pop_manager.update_score(spec.id, avg_score, all_failures)
            self.pop_manager.save_spec(spec)
            print(f"  {spec.id}: score={avg_score:.3f}")

        await asyncio.gather(*[eval_spec(s) for s in specs])

    # ------------------------------------------------------------------
    # Mutation pipeline
    # ------------------------------------------------------------------

    async def _produce_next_generation(self, generation: int, cfg_ev: dict):
        parents  = self.pop_manager.get_top_k(3)
        children = []
        mut_types = cfg_ev.get("mutation_types", ["llm", "random", "crossover"])

        for parent in parents:
            failures = self.pop_manager.failures.get(parent.id, [])
            examples = self.db.get_failure_examples(parent.id, limit=3)

            # LLM mutation
            if "llm" in mut_types:
                child = await asyncio.to_thread(
                    llm_mutation, parent, failures,
                    cfg_ev["mutation_model"], examples
                )
                if child and child.validate_spec(
                    list(self.config["runtime"].get("available_tools",
                    ["web_search", "scrape_page", "python_repl"]))
                ):
                    children.append(child)
                    self.db.log_mutation(parent.id, child.id, "llm")

            # Random mutation
            if "random" in mut_types and random.random() < cfg_ev.get("random_mutation_rate", 0.5):
                child = await asyncio.to_thread(random_mutation, parent)
                if child and child.validate_spec(
                    ["web_search", "scrape_page", "python_repl"]
                ):
                    children.append(child)
                    self.db.log_mutation(parent.id, child.id, "random")

        # Crossover between top-2
        if "crossover" in mut_types and len(parents) >= 2:
            child = await asyncio.to_thread(crossover_mutation, parents[0], parents[1])
            if child and child.validate_spec(["web_search", "scrape_page", "python_repl"]):
                children.append(child)
                self.db.log_mutation(parents[0].id, child.id, "crossover")

        # Regression check (fast canary before full eval)
        valid_children = []
        for child in children:
            if not self.regression_tasks:
                valid_children.append(child)
                continue
            runner = ResearchRunner(child, self.config["runtime"]["agent_model"])
            passed = 0
            for r_task in self.regression_tasks[:3]:   # sample 3 of 10 for speed
                try:
                    res = await asyncio.wait_for(
                        runner.run(r_task["question"]), timeout=60
                    )
                    rr = res.to_dict() if hasattr(res, "to_dict") else res
                    ev = evaluate_run(r_task, rr)
                    if ev["score"] > 0.0:
                        passed += 1
                except asyncio.TimeoutError:
                    pass
            if passed > 0:
                valid_children.append(child)

        self.pop_manager.add_specs(valid_children)
        self.pop_manager.cull(cfg_ev["population_size"])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_tasks(self, path: str) -> list:
        if not path or not os.path.exists(path):
            return []
        tasks = []
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks
