import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class MemorySpec(BaseModel):
    type:           str           = Field(default="scratchpad")
    max_steps:      int           = Field(default=20)
    summary_prompt: Optional[str] = None


class StopConditionSpec(BaseModel):
    min_report_length:     int  = Field(default=500)
    must_include_citations: bool = Field(default=True)
    max_steps:             int  = Field(default=10)


class AgentSpec(BaseModel):
    id:               str
    system_prompt:    str
    tools:            List[str]
    planning_strategy: str               = Field(default="react")
    memory:           MemorySpec         = Field(default_factory=MemorySpec)
    stop_condition:   StopConditionSpec  = Field(default_factory=StopConditionSpec)
    output_schema:    str                = Field(default="markdown_report_with_citations")
    temperature:      float              = Field(default=0.0, ge=0.0, le=2.0)

    # Lineage fields (populated by the orchestrator, not by the LLM)
    parent_id:     Optional[str] = Field(default=None)
    generation:    int           = Field(default=0)
    mutation_type: Optional[str] = Field(default=None)  # "seed"|"llm"|"random"|"crossover"

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentSpec":
        return cls(**data)

    def validate_spec(self, available_tools: List[str]) -> bool:
        """Check that all listed tools exist and stop conditions are well-formed."""
        for tool in self.tools:
            if tool not in available_tools:
                return False
        if self.stop_condition.max_steps <= 0:
            return False
        if self.planning_strategy not in ("react", "plan_and_solve"):
            return False
        return True

    def save(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "AgentSpec":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
