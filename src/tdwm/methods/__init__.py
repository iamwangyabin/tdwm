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
from .local_successor import (
    GoalConditionedPolicy,
    LocalSuccessorHeads,
    SuccessorPredictor,
    SuccessorTDOutput,
    future_goal_successor_objective,
    goal_cost_weights,
    latent_goal_cost,
    successor_feature_basis,
    successor_goal_cost,
    successor_td_target,
)

__all__ = [
    "GoalTailTDOutput",
    "GoalTailValue",
    "discounted_goal_tail_target",
    "ema_update",
    "future_goal_td_objective",
    "goal_cost",
    "goal_tail_loss",
    "GoalConditionedPolicy",
    "LocalSuccessorHeads",
    "SuccessorPredictor",
    "SuccessorTDOutput",
    "future_goal_successor_objective",
    "goal_cost_weights",
    "latent_goal_cost",
    "successor_feature_basis",
    "successor_goal_cost",
    "successor_td_target",
]
