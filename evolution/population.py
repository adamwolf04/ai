"""
PopulationManager — implements PopulationStore interface with LineageDB wiring.
"""
import os
from typing import Optional
from spec import AgentSpec
from db.lineage import LineageDB
from interfaces import PopulationStore


class PopulationManager(PopulationStore):

    def __init__(self, specs_dir: str = "specs", db: Optional[LineageDB] = None):
        self.specs_dir  = specs_dir
        self.db         = db
        os.makedirs(specs_dir, exist_ok=True)

        self.population: list[AgentSpec] = []
        self.scores:     dict[str, float] = {}
        self.failures:   dict[str, list]  = {}

    # ------------------------------------------------------------------
    # PopulationStore interface
    # ------------------------------------------------------------------

    def add(self, spec: AgentSpec) -> None:
        existing_ids = {s.id for s in self.population}
        if spec.id not in existing_ids:
            self.population.append(spec)
            if self.db:
                self.db.log_spec(
                    spec_id=spec.id,
                    generation=spec.generation,
                    spec_json=spec.to_dict(),
                    parent_id=spec.parent_id,
                    mutation_type=spec.mutation_type,
                )

    def add_specs(self, specs: list[AgentSpec]) -> None:
        """Batch add (deduplication enforced)."""
        for s in specs:
            self.add(s)

    def update_score(self, spec_id: str, score: float,
                     failures: Optional[list] = None) -> None:
        self.scores[spec_id]   = score
        self.failures[spec_id] = failures or []

    def get_top_k(self, k: int) -> list[AgentSpec]:
        return sorted(
            self.population,
            key=lambda s: self.scores.get(s.id, 0.0),
            reverse=True
        )[:k]

    def get_best(self) -> Optional[AgentSpec]:
        if not self.population:
            return None
        return self.get_top_k(1)[0]

    # Alias kept for backward compatibility
    def get_best_spec(self) -> Optional[AgentSpec]:
        return self.get_best()

    def cull(self, max_size: int) -> None:
        sorted_specs   = sorted(
            self.population,
            key=lambda s: self.scores.get(s.id, 0.0),
            reverse=True
        )
        survivors      = sorted_specs[:max_size]
        eliminated     = sorted_specs[max_size:]
        self.population = survivors

        if self.db:
            for spec in eliminated:
                self.db.mark_eliminated(spec.id)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save_spec(self, spec: AgentSpec) -> None:
        path = os.path.join(self.specs_dir, f"{spec.id}.json")
        spec.save(path)

    def restore_from_db(self) -> None:
        """Reload population and scores from the lineage DB (for resume)."""
        if not self.db:
            return
        spec_dicts = self.db.load_population()
        for d in spec_dicts:
            try:
                spec = AgentSpec.from_dict(d)
                existing_ids = {s.id for s in self.population}
                if spec.id not in existing_ids:
                    self.population.append(spec)
            except Exception:
                pass
        self.scores = self.db.load_scores()
