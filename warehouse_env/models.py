from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Order(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: int
    items: Dict[str, int] = Field(default_factory=dict)
    deadline: int
    priority: float = Field(ge=0)
    processing_time: int = Field(default=1, ge=1)
    status: Literal["pending", "active", "fulfilled", "late", "missed"] = "pending"
    remaining_time: int = 0
    started_at: Optional[int] = None
    completed_at: Optional[int] = None
    late_by: int = 0


class Shipment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item: str
    quantity: int = Field(gt=0)
    arrives_at: int = Field(ge=0)


class WarehouseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["start_order", "expedite_order", "restock_item", "wait"]
    order_id: Optional[int] = None
    item: Optional[str] = None
    quantity: Optional[int] = None


class WarehouseReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    total: float = 0.0
    components: Dict[str, float] = Field(default_factory=dict)


class GraderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completion_weight: float = Field(default=0.45, ge=0)
    on_time_weight: float = Field(default=0.25, ge=0)
    priority_weight: float = Field(default=0.15, ge=0)
    efficiency_weight: float = Field(default=0.10, ge=0)
    reward_weight: float = Field(default=0.05, ge=0)


class TaskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    name: str
    difficulty: Literal["easy", "medium", "hard"]
    max_steps: int = Field(default=12, ge=1)
    initial_inventory: Dict[str, int] = Field(default_factory=dict)
    orders: List[Order] = Field(default_factory=list)
    restock_lead_time: int = Field(default=1, ge=0)
    restock_cost: float = Field(default=0.05, ge=0)
    expedite_cost: float = Field(default=0.10, ge=0)
    wait_cost: float = Field(default=0.02, ge=0)
    late_penalty: float = Field(default=0.10, ge=0)
    invalid_action_penalty: float = Field(default=0.50, ge=0)
    progress_reward: float = Field(default=0.05, ge=0)
    completion_bonus: float = Field(default=1.00, ge=0)
    on_time_bonus: float = Field(default=0.20, ge=0)
    reward_floor: float = -6.0
    reward_ceiling: float = 12.0
    grader: GraderConfig = Field(default_factory=GraderConfig)


class WarehouseObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_name: str
    difficulty: str
    time: int
    max_steps: int
    inventory: Dict[str, int]
    active_order: Optional[Order] = None
    pending_orders: List[Order] = Field(default_factory=list)
    incoming_shipments: List[Shipment] = Field(default_factory=list)
    metrics: Dict[str, float] = Field(default_factory=dict)


class WarehouseState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: TaskConfig
    time: int = 0
    inventory: Dict[str, int] = Field(default_factory=dict)
    orders: List[Order] = Field(default_factory=list)
    incoming_shipments: List[Shipment] = Field(default_factory=list)
    active_order_id: Optional[int] = None
    active_remaining_time: int = 0
    total_reward: float = 0.0
    invalid_actions: int = 0
    wait_actions: int = 0
    restock_actions: int = 0
    expedite_actions: int = 0
    start_actions: int = 0
    completed_orders: int = 0
    fulfilled_on_time: int = 0
    fulfilled_late: int = 0
    total_late_steps: int = 0
    completed_priority: float = 0.0
    total_priority: float = 0.0
    restocked_units: int = 0