# Training Logs

This directory contains output from GRPO training runs against the live DataCentricEnvironment.

## Files

### `training.jsonl`
Per-episode reward log. Each line is one training episode:

```json
{
  "episode": 5,
  "task": "task_1_easy",
  "level": 1,
  "reward": 0.312,
  "accuracy_gain": 0.091,
  "steps_used": 11,
  "success": true,
  "curriculum_stage": "easy"
}
```

| Field | Description |
|---|---|
| `episode` | Global episode counter across the training run |
| `task` | Which curriculum task was run (`task_0_tutorial` … `task_3_hard`) |
| `level` | Curriculum level (0=tutorial, 1=easy, 2=medium, 3=hard) |
| `reward` | Total episode reward from the composable rubric system [-1.0, 1.0] |
| `accuracy_gain` | Raw accuracy improvement above the episode baseline |
| `steps_used` | Number of actions taken before submit |
| `success` | Whether the agent hit the target accuracy threshold |
| `curriculum_stage` | Human-readable level label |

### `grpo/` and `sft/`
TensorBoard event files. View with:
```bash
tensorboard --logdir logs/
```

## Generating Real Logs

Run the training notebook:
```
train_colab.ipynb  →  Step 7 (GRPO Training)
```

The log is written incrementally — one line per episode — by `log_episode_jsonl()` in `train_data_centric.py`. After training, commit the full `logs/training.jsonl` to replace this sample file.

> **Note:** The `training.jsonl` in this directory is a **sample** showing the log format and expected learning trajectory. Replace it with your actual run output after training completes.
