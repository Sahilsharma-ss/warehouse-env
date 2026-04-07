from __future__ import annotations

import json
from typing import Iterable, Sequence

from warehouse_env.models import TaskConfig


class TaskGrader:
    _EPSILON = 1e-4

    def __init__(self, max_steps: int | None = None):
        self.max_steps = max_steps

    def evaluate(self, env, agent):
        observation = env.reset()
        total_reward = 0.0
        step_limit = self.max_steps or env.config.max_steps

        for _ in range(step_limit):
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            total_reward += reward
            if done:
                break

        return self.score(env, total_reward)

    def score(self, env, total_reward: float):
        summary = env.summary()
        config = env.config
        weights = config.grader

        completion = summary["completion_ratio"]
        on_time = summary["on_time_ratio"]
        priority = summary["priority_completion_ratio"]
        efficiency = max(0.0, 1.0 - summary["invalid_action_rate"])
        reward_component = self._normalize_reward(total_reward, config.reward_floor, config.reward_ceiling)

        weighted_sum = (
            weights.completion_weight * completion
            + weights.on_time_weight * on_time
            + weights.priority_weight * priority
            + weights.efficiency_weight * efficiency
            + weights.reward_weight * reward_component
        )

        total_weight = (
            weights.completion_weight
            + weights.on_time_weight
            + weights.priority_weight
            + weights.efficiency_weight
            + weights.reward_weight
        )

        if total_weight <= 0:
            return self._strict_unit_interval(0.0)

        normalized_score = max(0.0, min(1.0, weighted_sum / total_weight))
        return self._strict_unit_interval(normalized_score)

    def evaluate_all(self, env_class, agent, task_files: Sequence[str]):
        scores = []

        for task_file in task_files:
            with open(task_file, "r", encoding="utf-8") as handle:
                config = TaskConfig.model_validate(json.load(handle))

            env = env_class(config)
            score = self.evaluate(env, agent)
            scores.append(score)

        if not scores:
            return self._strict_unit_interval(0.0)

        return self._strict_unit_interval(sum(scores) / len(scores))

    @classmethod
    def _strict_unit_interval(cls, value: float) -> float:
        clamped = max(cls._EPSILON, min(1.0 - cls._EPSILON, value))
        return round(clamped, 4)

    @classmethod
    def _normalize_reward(cls, reward: float, floor: float, ceiling: float) -> float:
        if ceiling <= floor:
            return cls._EPSILON

        normalized = (reward - floor) / (ceiling - floor)
        clamped = max(0.0, min(1.0, normalized))
        return cls._strict_unit_interval(clamped)


class Grader(TaskGrader):
    pass
