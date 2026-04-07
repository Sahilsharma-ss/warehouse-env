from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from warehouse_env.models import (
    Order,
    Shipment,
    TaskConfig,
    WarehouseAction,
    WarehouseObservation,
    WarehouseReward,
    WarehouseState,
)


class WarehouseEnv:
    _RATIO_MIN = 0.1
    _RATIO_MAX = 0.99

    def __init__(self, config: Dict[str, Any] | TaskConfig):
        self.config = TaskConfig.model_validate(config)
        self._initial_state = WarehouseState(
            config=self.config.model_copy(deep=True),
            inventory=dict(self.config.initial_inventory),
            orders=[order.model_copy(deep=True) for order in self.config.orders],
            total_priority=sum(order.priority for order in self.config.orders),
        )
        self._state: Optional[WarehouseState] = None

    def reset(self):
        self._state = self._initial_state.model_copy(deep=True)
        return self._build_observation().model_dump()

    def state(self):
        self._ensure_state()
        return self._state.model_dump()

    def summary(self):
        self._ensure_state()

        total_orders = max(1, len(self._state.orders))
        total_priority = max(1.0, self._state.total_priority)
        action_count = max(
            1,
            self._state.start_actions
            + self._state.restock_actions
            + self._state.expedite_actions
            + self._state.wait_actions
            + self._state.invalid_actions,
        )

        completion_ratio = self._strict_ratio(self._state.completed_orders / total_orders)
        on_time_ratio = self._strict_ratio(self._state.fulfilled_on_time / total_orders)
        priority_completion_ratio = self._strict_ratio(self._state.completed_priority / total_priority)
        invalid_action_rate = self._strict_ratio(self._state.invalid_actions / action_count)

        return {
            "task_id": self._state.config.task_id,
            "task_name": self._state.config.name,
            "difficulty": self._state.config.difficulty,
            "time": self._state.time,
            "max_steps": self._state.config.max_steps,
            "completed_orders": self._state.completed_orders,
            "fulfilled_on_time": self._state.fulfilled_on_time,
            "fulfilled_late": self._state.fulfilled_late,
            "completion_ratio": completion_ratio,
            "on_time_ratio": on_time_ratio,
            "priority_completion_ratio": priority_completion_ratio,
            "invalid_action_rate": invalid_action_rate,
            "wait_actions": self._state.wait_actions,
            "restock_actions": self._state.restock_actions,
            "expedite_actions": self._state.expedite_actions,
            "start_actions": self._state.start_actions,
            "restocked_units": self._state.restocked_units,
            "total_reward": self._state.total_reward,
        }

    def step(self, action):
        self._ensure_state()

        reward = WarehouseReward()
        action_model, action_error = self._parse_action(action)

        if action_error is not None:
            self._state.invalid_actions += 1
            reward.components["invalid_action"] = -self.config.invalid_action_penalty
        else:
            self._apply_action(action_model, reward)

        self._advance_time(reward)

        done = self._is_done()
        if done and self._state.time >= self.config.max_steps:
            self._finalize_unfinished_orders(reward)

        reward.total = round(sum(reward.components.values()), 6)
        self._state.total_reward += reward.total

        observation = self._build_observation()
        info = {
            "reward_breakdown": reward.model_dump(),
            "summary": self.summary(),
            "action_valid": action_error is None,
        }
        return observation.model_dump(), reward.total, done, info

    def _ensure_state(self):
        if self._state is None:
            self.reset()

    def _parse_action(self, action) -> Tuple[Optional[WarehouseAction], Optional[str]]:
        try:
            return WarehouseAction.model_validate(action), None
        except Exception as exc:  # pragma: no cover - defensive path
            return None, str(exc)

    def _apply_action(self, action: WarehouseAction, reward: WarehouseReward):
        if action.type == "wait":
            self._state.wait_actions += 1
            reward.components["wait_cost"] = -self.config.wait_cost
            return

        if action.type == "restock_item":
            self._apply_restock(action, reward)
            return

        if action.type == "start_order":
            self._apply_start_order(action, reward)
            return

        if action.type == "expedite_order":
            self._apply_expedite(action, reward)
            return

        self._state.invalid_actions += 1
        reward.components["invalid_action"] = -self.config.invalid_action_penalty

    def _apply_start_order(self, action: WarehouseAction, reward: WarehouseReward):
        self._state.start_actions += 1

        if self._state.active_order_id is not None or action.order_id is None:
            self._state.invalid_actions += 1
            reward.components["start_order_invalid"] = -self.config.invalid_action_penalty
            return

        order = self._find_order(action.order_id)
        if order is None or order.status in {"fulfilled", "missed", "active"}:
            self._state.invalid_actions += 1
            reward.components["start_order_invalid"] = -self.config.invalid_action_penalty
            return

        if not self._has_inventory(order):
            self._state.invalid_actions += 1
            reward.components["insufficient_inventory"] = -self.config.invalid_action_penalty
            return

        for item, quantity in order.items.items():
            self._state.inventory[item] = self._state.inventory.get(item, 0) - quantity

        order.status = "active"
        order.started_at = self._state.time
        order.remaining_time = order.processing_time
        self._state.active_order_id = order.id
        self._state.active_remaining_time = order.processing_time
        reward.components["start_bonus"] = 0.03 * order.priority

    def _apply_expedite(self, action: WarehouseAction, reward: WarehouseReward):
        self._state.expedite_actions += 1

        active_order = self._get_active_order()
        if active_order is None:
            self._state.invalid_actions += 1
            reward.components["expedite_invalid"] = -self.config.invalid_action_penalty
            return

        if action.order_id is not None and action.order_id != active_order.id:
            self._state.invalid_actions += 1
            reward.components["expedite_invalid"] = -self.config.invalid_action_penalty
            return

        if self._state.active_remaining_time > 0:
            self._state.active_remaining_time = max(0, self._state.active_remaining_time - 1)
            active_order.remaining_time = self._state.active_remaining_time
            reward.components["expedite_cost"] = -self.config.expedite_cost

    def _apply_restock(self, action: WarehouseAction, reward: WarehouseReward):
        self._state.restock_actions += 1

        if action.item is None or action.quantity is None or action.quantity <= 0:
            self._state.invalid_actions += 1
            reward.components["restock_invalid"] = -self.config.invalid_action_penalty
            return

        arrives_at = self._state.time + self.config.restock_lead_time
        self._state.incoming_shipments.append(
            Shipment(item=action.item, quantity=action.quantity, arrives_at=arrives_at)
        )
        self._state.restocked_units += action.quantity
        reward.components["restock_cost"] = -(self.config.restock_cost * action.quantity)

    def _advance_time(self, reward: WarehouseReward):
        self._state.time += 1
        self._deliver_shipments(reward)
        self._progress_active_order(reward)
        self._update_late_orders(reward)

    def _deliver_shipments(self, reward: WarehouseReward):
        arriving = [shipment for shipment in self._state.incoming_shipments if shipment.arrives_at <= self._state.time]
        remaining = [shipment for shipment in self._state.incoming_shipments if shipment.arrives_at > self._state.time]

        for shipment in arriving:
            self._state.inventory[shipment.item] = self._state.inventory.get(shipment.item, 0) + shipment.quantity
            reward.components[f"restock_arrival_{shipment.item}"] = reward.components.get(
                f"restock_arrival_{shipment.item}",
                0.0,
            ) + 0.01 * shipment.quantity

        self._state.incoming_shipments = remaining

    def _progress_active_order(self, reward: WarehouseReward):
        active_order = self._get_active_order()
        if active_order is None:
            return

        self._state.active_remaining_time = max(0, self._state.active_remaining_time - 1)
        active_order.remaining_time = self._state.active_remaining_time
        reward.components["progress"] = reward.components.get("progress", 0.0) + self.config.progress_reward * active_order.priority

        if self._state.active_remaining_time > 0:
            return

        active_order.status = "fulfilled"
        active_order.completed_at = self._state.time
        active_order.late_by = max(0, self._state.time - active_order.deadline)
        self._state.active_order_id = None

        self._state.completed_orders += 1
        self._state.completed_priority += active_order.priority

        if active_order.completed_at <= active_order.deadline:
            self._state.fulfilled_on_time += 1
            reward.components["completion"] = reward.components.get("completion", 0.0) + self.config.completion_bonus * active_order.priority
            reward.components["on_time_bonus"] = reward.components.get("on_time_bonus", 0.0) + self.config.on_time_bonus * active_order.priority
        else:
            self._state.fulfilled_late += 1
            reward.components["completion"] = reward.components.get("completion", 0.0) + 0.5 * self.config.completion_bonus * active_order.priority
            reward.components["late_completion_penalty"] = reward.components.get("late_completion_penalty", 0.0) - self.config.late_penalty * active_order.late_by * active_order.priority

    def _update_late_orders(self, reward: WarehouseReward):
        late_pressure = 0.0
        for order in self._state.orders:
            if order.status in {"fulfilled", "missed"}:
                continue

            if self._state.time > order.deadline:
                order.status = "late"
                order.late_by = self._state.time - order.deadline
                late_pressure -= self.config.late_penalty * min(1.0, 0.25 * order.late_by) * order.priority

        if late_pressure != 0.0:
            reward.components["late_pressure"] = reward.components.get("late_pressure", 0.0) + late_pressure

    def _finalize_unfinished_orders(self, reward: WarehouseReward):
        for order in self._state.orders:
            if order.status in {"fulfilled", "missed"}:
                continue

            order.status = "missed"
            order.completed_at = None
            reward.components[f"missed_order_{order.id}"] = -self.config.invalid_action_penalty * 0.5

    def _build_observation(self) -> WarehouseObservation:
        active_order = self._get_active_order()
        pending_orders = [
            order.model_copy(deep=True)
            for order in self._state.orders
            if order.status in {"pending", "late"}
        ]

        completion_ratio = self._strict_ratio(self._state.completed_orders / max(1, len(self._state.orders)))
        on_time_ratio = self._strict_ratio(self._state.fulfilled_on_time / max(1, len(self._state.orders)))
        priority_completion_ratio = self._strict_ratio(
            self._state.completed_priority / max(1.0, self._state.total_priority)
        )

        return WarehouseObservation(
            task_id=self._state.config.task_id,
            task_name=self._state.config.name,
            difficulty=self._state.config.difficulty,
            time=self._state.time,
            max_steps=self._state.config.max_steps,
            inventory=dict(self._state.inventory),
            active_order=active_order.model_copy(deep=True) if active_order is not None else None,
            pending_orders=pending_orders,
            incoming_shipments=[shipment.model_copy(deep=True) for shipment in self._state.incoming_shipments],
            metrics={
                "completion_ratio": completion_ratio,
                "on_time_ratio": on_time_ratio,
                "priority_completion_ratio": priority_completion_ratio,
                "total_reward": self._state.total_reward,
            },
        )

    @classmethod
    def _strict_ratio(cls, value: float) -> float:
        clamped = max(cls._RATIO_MIN, min(cls._RATIO_MAX, value))
        return round(clamped, 4)

    def _find_order(self, order_id: int) -> Optional[Order]:
        return next((order for order in self._state.orders if order.id == order_id), None)

    def _get_active_order(self) -> Optional[Order]:
        if self._state.active_order_id is None:
            return None
        return self._find_order(self._state.active_order_id)

    def _has_inventory(self, order: Order) -> bool:
        return all(self._state.inventory.get(item, 0) >= quantity for item, quantity in order.items.items())

    def _is_done(self) -> bool:
        if self._state.time >= self.config.max_steps:
            return True
        return all(order.status in {"fulfilled", "missed"} for order in self._state.orders)