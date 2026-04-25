"""Data-Centric Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import DataCentricAction, DataCentricObservation
except ImportError:
    from models import DataCentricAction, DataCentricObservation


class DataCentricEnv(EnvClient[DataCentricAction, DataCentricObservation, State]):
    """
    Client for the Data-Centric RL Environment.

    Connects over WebSocket for efficient multi-step interactions.

    Example:
        >>> with DataCentricEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task="task_0_tutorial")
        ...     result = client.step(DataCentricAction(message="inspect_dataset"))
        ...     print(result.observation.response)

    Docker example:
        >>> client = DataCentricEnv.from_docker_image("data_centric_env:latest")
        >>> result = client.reset(task="task_1_easy")
    """

    def _step_payload(self, action: DataCentricAction) -> Dict:
        return {"message": action.message}

    def _parse_result(self, payload: Dict) -> StepResult[DataCentricObservation]:
        obs_data = payload.get("observation", {})
        observation = DataCentricObservation(
            response=obs_data.get("response", ""),
            current_accuracy=obs_data.get("current_accuracy", 0.0),
            baseline_accuracy=obs_data.get("baseline_accuracy", 0.0),
            target_accuracy=obs_data.get("target_accuracy", 0.0),
            estimated_quality=obs_data.get("estimated_quality", 0.0),
            dataset_shape=obs_data.get("dataset_shape", ""),
            rows_preserved_pct=obs_data.get("rows_preserved_pct", 1.0),
            budget_remaining=obs_data.get("budget_remaining", 0),
            step_number=obs_data.get("step_number", 0),
            max_steps=obs_data.get("max_steps", 30),
            active_session=obs_data.get("active_session", "none"),
            validate_calls_remaining=obs_data.get("validate_calls_remaining", 3),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
