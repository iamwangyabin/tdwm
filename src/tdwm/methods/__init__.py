"""Research method components implemented by TDWM."""

from .goal_tail import (
    GoalTailValue,
    discounted_goal_tail_target,
    goal_cost,
    goal_tail_loss,
    soft_update,
)

__all__ = [
    "GoalTailValue",
    "discounted_goal_tail_target",
    "goal_cost",
    "goal_tail_loss",
    "soft_update",
]
