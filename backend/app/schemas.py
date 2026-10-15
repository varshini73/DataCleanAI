from pydantic import BaseModel, Field
from typing import Any, Literal

class Operation(BaseModel):
    name: Literal['missing', 'duplicates', 'outliers', 'convert', 'categorical', 'scale']
    enabled: bool = True
    columns: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ''

class PlanRequest(BaseModel):
    objective: str = ''
    selected_operations: list[Operation] = Field(default_factory=list)
    all_applicable: bool = False

class ExecuteRequest(BaseModel):
    operations: list[Operation]
    confirm_destructive: bool = False

class SimulationRequest(BaseModel):
    operations: list[Operation]

class ApplySimulationRequest(BaseModel):
    candidate_id: str
    confirm_destructive: bool = False
