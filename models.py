# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Data-Centric RL Environment.

Action  → plain text command string (like DataWranglerEnv)
Observation → rich structured observation with accuracy, quality, budget info
State   → episode metadata
"""

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DataCentricAction(Action):
    """Action for the Data-Centric environment — a text command string.

    The agent sends natural-language-style commands to inspect the dataset,
    query specialist sub-agents, apply their recommendations, and ultimately
    submit the cleaned dataset for scoring.

    Examples:
        - "inspect_dataset"
        - "inspect_model"
        - "query_cleaner"
        - "query_augmenter class_1"
        - "query_balancer"
        - "query_validator"
        - "apply 1"
        - "reject 2"
        - "validate"
        - "submit"
    """

    message: str = Field(..., description="Text command to execute in the environment")


class DataCentricObservation(Observation):
    """Observation returned after each action in the Data-Centric environment.

    Provides the agent with rich feedback about the current episode state,
    including dataset health, model accuracy, budget, and specialist session info.
    """

    response: str = Field(
        default="",
        description="Text result of the executed command",
    )
    current_accuracy: float = Field(
        default=0.0,
        description="Last validated model accuracy (or baseline if not yet validated)",
    )
    baseline_accuracy: float = Field(
        default=0.0,
        description="Accuracy at episode start — never changes",
    )
    target_accuracy: float = Field(
        default=0.0,
        description="Accuracy threshold the agent must exceed to hit target",
    )
    estimated_quality: float = Field(
        default=0.0,
        description="Lightweight quality score without sklearn retraining (0.0-1.0)",
    )
    dataset_shape: str = Field(
        default="",
        description="Current dataset dimensions, e.g. '200 rows × 5 columns'",
    )
    rows_preserved_pct: float = Field(
        default=1.0,
        description="Fraction of original rows still present (1.0 = no data loss)",
    )
    budget_remaining: int = Field(
        default=0,
        description="Steps remaining before forced submit",
    )
    step_number: int = Field(
        default=0,
        description="Current step number in the episode",
    )
    max_steps: int = Field(
        default=30,
        description="Maximum steps allowed for this task",
    )
    active_session: str = Field(
        default="none",
        description="Which specialist agent was queried last (cleaner/augmenter/balancer/none)",
    )
    validate_calls_remaining: int = Field(
        default=3,
        description="How many more free validates remain before reward turns negative",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has ended",
    )
    reward: float = Field(
        default=0.0,
        description="Reward for this step",
    )
