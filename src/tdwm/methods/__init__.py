"""Research method components implemented by TDWM."""

from .goal_tail import (
    GoalTailTDOutput,
    GoalTailValue,
    discounted_goal_tail_target,
    ema_update,
    future_goal_td_objective,
    goal_cost,
    goal_tail_loss,
)

__all__ = [
    "GoalTailTDOutput",
    "GoalTailValue",
    "discounted_goal_tail_target",
    "ema_update",
    "future_goal_td_objective",
    "goal_cost",
    "goal_tail_loss",
]
