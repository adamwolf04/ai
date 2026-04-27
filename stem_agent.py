"""
Stem Agent Orchestrator — Meta-optimization loop for agent evolution.
Satisfies all "Unforgivable Curses" (Rule #0 - #5).
"""
from __future__ import annotations

import asyncio
import os
import yaml
import json
from typing import List, Dict, Any, Optional

from spec import AgentSpec
from evaluator.batch import evaluate_spec_on_tasks
from db.lineage import LineageDB
from evolution.population import PopulationManager
from evolution.operators import (
    seed_population,
    llm_mutation,
    random_mutation,
    crossover_mutation
)


class StemAgent:
    """
    Main orchestrator for the Stem Agent evolution loop.
    Encapsulates population management and generation flow (Rule #5).
    """

    def __init__(self, config_path: str = "config.yaml",
                 config_override: Optional[Dict[str, Any]] = None):
        """Rule #1: Decomposition - Constructor handles initialization logic."""
        if config_override is not None:
            self.config = config_override
        else:
            with open(config_path, "r") as f:
                self.config = yaml.safe_load(f)

        self._db = LineageDB(self.config["paths"].get("db_path", "logs/lineage.db"))
        self._pop_manager = PopulationManager(
            specs_dir=self.config["paths"].get("specs_dir", "specs"),
            db=self._db
        )
        
        # Rule #2: Constants loaded from config
        self._train_tasks = self._load_tasks(self.config["data"]["train_tasks"])
        self._task_timeout = self.config["evolution"].get("task_timeout", 120)
        self._concurrency = self.config["evolution"].get("eval_concurrency", 8)
        self._mutation_model = self.config["evolution"]["mutation_model"]
        self._agent_model = self.config["runtime"]["agent_model"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_evolution(self) -> Optional[AgentSpec]:
        """Entry point for the evolutionary process."""
        return await self._async_evolution()

    # ------------------------------------------------------------------
    # Core Loop (Rule #1: Decomposition)
    # ------------------------------------------------------------------

    async def _async_evolution(self) -> Optional[AgentSpec]:
        """The primary generation-based evolution loop."""
        print("Seeding population...")
        if not self._pop_manager.population:
            seeds = seed_population(model=self._mutation_model)
            # Rule #5: Encapsulated DB logging
            for s in seeds:
                self._db.log_spec(s.id, 0, s.to_dict(), mutation_type="seed")
            self._pop_manager.add_specs(seeds)

        num_generations = self.config["evolution"]["generations"]
        
        for gen in range(1, num_generations + 1):
            print(f"\n--- Generation {gen} ---")
            
            # 1. Evaluate current population (Rule #1 / Rule #3)
            await self._evaluate_unevaluated_specs()

            # 2. Selection & Logging
            best_spec, best_score = self._pop_manager.get_best_info()
            if best_spec:
                print(f"Best: {best_spec.id}  score={best_score:.3f}")
                self._db.log_generation(gen, best_spec.id, best_score)
            
            # 3. Produce offspring
            if gen < num_generations:
                await self._evolve_next_generation()

        final_best_spec, _ = self._pop_manager.get_best_info()
        return final_best_spec

    # ------------------------------------------------------------------
    # Evolution Steps (Rule #1 / Rule #5)
    # ------------------------------------------------------------------

    async def _evaluate_unevaluated_specs(self):
        """Identifies and scores specs that lack evaluation data."""
        unevaluated = self._pop_manager.get_unevaluated_specs()
        
        async def eval_spec(spec: AgentSpec):
            print(f"  Evaluating {spec.id}...")
            results = await evaluate_spec_on_tasks(
                spec,
                self._train_tasks,
                model=self._agent_model,
                timeout=self._task_timeout,
                concurrency=self._concurrency,
                on_result_callback=lambda tid, task, eres, rr: self._log_task_result(spec.id, tid, task, eres, rr)
            )
            self._pop_manager.record_score(spec.id, results["avg_score"], results["failures"])

        await asyncio.gather(*(eval_spec(spec) for spec in unevaluated))

    async def _evolve_next_generation(self):
        """Creates mutations and crossovers for the next population (Rule #1)."""
        population = self._pop_manager.get_top_k(self.config["evolution"]["population_size"])
        if not population:
            return

        next_gen_specs = []
        
        # 1. Elitism: Best survives
        best_spec, _ = self._pop_manager.get_best_info()
        if best_spec:
            next_gen_specs.append(best_spec)

        # 2. Evolutionary Operators (Rule #1: Decomposition to evolution/operators.py)
        for spec in population:
            failures = self._pop_manager.failures.get(spec.id, [])
            if failures:
                # Targeted LLM Mutation
                examples = self._db.get_failure_examples(spec.id, limit=3)
                child = llm_mutation(spec, failures, model=self._mutation_model, failure_examples=examples)
                if child:
                    self._db.log_spec(child.id, child.generation, child.to_dict(), 
                                     parent_id=spec.id, mutation_type="llm")
                    next_gen_specs.append(child)

        # 3. Crossover & Random Tweak
        if len(population) >= 2:
            s1, s2 = population[0], population[1]
            child_x = crossover_mutation(s1, s2)
            if child_x:
                self._db.log_spec(child_x.id, child_x.generation, child_x.to_dict(), 
                                 parent_id=s1.id, mutation_type="crossover")
                next_gen_specs.append(child_x)

        self._pop_manager.add_specs(next_gen_specs)
        
        # Rule #1 & Critical Bug Fix: Cull population to prevent unbounded growth
        self._pop_manager.cull(self.config["evolution"]["population_size"])

    # ------------------------------------------------------------------
    # Internal Helpers (Rule #5: Encapsulation)
    # ------------------------------------------------------------------

    def _log_task_result(self, spec_id: str, task_id: str, task: Dict[str, Any], 
                        eval_res: Dict[str, Any], rr_dict: Dict[str, Any]):
        """Internal callback to handle DB persistence (Rule #5)."""
        trace_data = eval_res.get("trace", {})
        self._db.log_evaluation(
            spec_id=spec_id,
            task_id=task_id,
            score=eval_res["score"],
            matched=eval_res["matched"],
            extracted=trace_data.get("extracted", ""),
            cause_code=trace_data.get("cause_code", ""),
        )
        self._db.log_run_trace(
            spec_id=spec_id,
            task_id=task_id,
            messages=rr_dict.get("trace", []),
            token_count=rr_dict.get("token_count", 0),
            status=rr_dict.get("status", ""),
        )

    def _load_tasks(self, path: str) -> List[Dict[str, Any]]:
        """Loads and parses JSONL tasks (Rule #3)."""
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return [json.loads(line) for line in f if line.strip()]
