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
    successor_td_target,
)
from .rf_successor_lewm import (
    ActionPrefixSuccessorHead,
    MultiHorizonSuccessorOutput,
    finite_horizon_successor_targets,
    multi_horizon_successor_objective,
    successor_recurrence_residual,
)
from .successor_geometry import (
    goal_cost_weights,
    latent_goal_cost,
    successor_feature_basis,
    successor_goal_cost,
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
    "ActionPrefixSuccessorHead",
    "MultiHorizonSuccessorOutput",
    "finite_horizon_successor_targets",
    "multi_horizon_successor_objective",
    "successor_recurrence_residual",
]
