# pip install trl unsloth transformers torch requests
# pip install matplotlib openenv-core scikit-learn pandas numpy datasets

import os
import json
import random
import time
import signal
import subprocess
import requests
import torch
from collections import deque
from statistics import mean
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer

# WebSocket client for stateful episode rollouts
sys_path_root = os.path.dirname(os.path.abspath(__file__))
import sys
if sys_path_root not in sys.path:
    sys.path.insert(0, sys_path_root)
from client import DataCentricEnv
from models import DataCentricAction
from agent_utils import (
    VALID_COMMANDS, SYSTEM_PROMPT, build_user_prompt,
    start_server, stop_server,
)

# ════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════

# ENV_URL: set this to your HF Space URL when running as an HF Job
# e.g. export ENV_URL="https://aswini-kumar-data-centric-env.hf.space"
BASE_URL = os.environ.get("ENV_URL", "http://localhost:8000")
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True



# ════════════════════════════════════════════════════════
# SERVER MANAGEMENT
# ════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════
# MODEL SETUP
# ════════════════════════════════════════════════════════

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    return model, tokenizer


# ════════════════════════════════════════════════════════
# PHASE 1 — SFT WARMUP
# ════════════════════════════════════════════════════════

def run_sft_warmup(model, tokenizer):
    """
    1 epoch of SFT on heuristic trajectories.
    Teaches model valid command format before GRPO starts.
    Without this, model outputs gibberish and gets zero reward.
    """
    print("\n=== PHASE 1: SFT WARMUP ===")

    print("[Tracking] TensorBoard experiment tracking ON — logs written to ./logs/sft")

    if os.path.exists("./sft-checkpoint"):
        model.load_adapter("./sft-checkpoint")
        print("Loaded existing SFT checkpoint — skipping warmup.")
        return model

    if not os.path.exists("sft_data.jsonl"):
        print("sft_data.jsonl not found. Run sft_generator.py first.")
        raise FileNotFoundError("sft_data.jsonl missing")

    raw = [json.loads(l) for l in open("sft_data.jsonl", encoding="utf-8")]
    print(f"Loaded {len(raw)} SFT examples")

    def format_example(ex):
        messages = [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": ex["prompt"]},
            {"role": "assistant", "content": ex["response"]},
        ]
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    sft_dataset = Dataset.from_list([format_example(ex) for ex in raw])

    sft_trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=sft_dataset,
        args=SFTConfig(
            output_dir="./sft-checkpoint",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=10,
            logging_steps=5,
            save_strategy="no",
            report_to="tensorboard",    # experiment tracking via TensorBoard
            logging_dir="./logs/sft",
            max_seq_length=MAX_SEQ_LENGTH,
        ),
    )
    sft_trainer.train()
    print("SFT warmup complete.\n")
    return model


# ════════════════════════════════════════════════════════
# CURRICULUM SCHEDULER
# ════════════════════════════════════════════════════════

class CurriculumScheduler:
    """
    Advances curriculum level when the agent reliably solves the current task.

    Advancement criterion: >= threshold success rate over a rolling window of episodes.
    Uses a smoothed window to avoid premature advancement on lucky streaks.

    Levels: 0=tutorial, 1=easy, 2=medium, 3=hard

    Design rationale:
    - Step-count based scheduling causes premature advancement (catastrophic forgetting)
      or stalling (wasted compute) because it ignores actual agent performance.
    - Success-rate based scheduling ensures the agent genuinely masters a level
      before seeing harder tasks, matching curriculum RL best practices.
    - Window resets after each advancement so the agent must prove itself again.
    """

    TASKS = ["task_0_tutorial", "task_1_easy", "task_2_medium", "task_3_hard"]
    LEVEL_LABELS = ["tutorial", "easy", "medium", "hard"]

    def __init__(self, window: int = 30, threshold: float = 0.60):
        """
        Args:
            window: Number of episodes to evaluate before considering advancement.
                    30 gives a stable estimate without waiting too long.
            threshold: Required success rate (0.0–1.0) to advance to next level.
                       0.60 = agent must solve 60% of episodes to advance.
                       Lower than classic 0.75 because 3B models learn slower.
        """
        self.current_level = 0
        self.window = window
        self.threshold = threshold
        self.recent_successes: deque = deque(maxlen=window)
        self.global_step = 0          # total episodes recorded (for logging)
        self.level_history: list = [] # (episode, level) pairs for plotting

    def record_episode(self, reached_target: bool, accuracy_gain: float = 0.0):
        """Call after every episode completes."""
        self.recent_successes.append(float(reached_target))
        self.global_step += 1
        if self.should_advance():
            self.advance()

    def get_task(self) -> str:
        """Return the current training task name."""
        return self.TASKS[self.current_level]

    def current_success_rate(self) -> float:
        if not self.recent_successes:
            return 0.0
        return sum(self.recent_successes) / len(self.recent_successes)

    def should_advance(self) -> bool:
        """Only advance if we have enough data and consistently exceed threshold."""
        return (
            len(self.recent_successes) >= self.window
            and self.current_success_rate() >= self.threshold
            and self.current_level < len(self.TASKS) - 1
        )

    def advance(self):
        if self.current_level < len(self.TASKS) - 1:
            print(
                f"\n[Curriculum] ▶ Level {self.current_level} ({self.TASKS[self.current_level]}) "
                f"→ Level {self.current_level + 1} ({self.TASKS[self.current_level + 1]})\n"
                f"  Success rate over last {self.window} episodes: "
                f"{self.current_success_rate():.1%} (threshold: {self.threshold:.0%})\n"
                f"  Total episodes: {self.global_step}"
            )
            self.level_history.append((self.global_step, self.current_level))
            self.current_level += 1
            self.recent_successes.clear()  # reset window after advancing

    def stage_label(self) -> str:
        return self.LEVEL_LABELS[self.current_level]

    # Backward-compat: record_improvement still works for old callers
    def record_improvement(self, improvement: float):
        self.record_episode(reached_target=(improvement > 0.05))




# ════════════════════════════════════════════════════════
# REWARD COMPUTATION
# ════════════════════════════════════════════════════════

def compute_rewards(
    obs_before: dict,
    obs_after: dict,
    response_text: str,
    action_history: list,
) -> dict:
    """
    Two independent reward components.

    env_reward    — the full graded reward from the environment (accuracy +
                    process + preservation + step). Do NOT re-implement those
                    here; they are already inside obs_after["reward"].
    format_reward — the only signal invisible to the environment: whether the
                    LLM actually output a valid command string.
    """
    # Component 1: environment reward (already includes accuracy, process,
    # preservation, and step reward — do not duplicate any of those here)
    env_reward = obs_after.get("reward", 0.0)

    # Component 2: format reward — did the model emit a valid command?
    # This is the ONLY signal the environment cannot see.
    is_valid = any(
        response_text.strip().startswith(cmd) for cmd in VALID_COMMANDS
    )
    format_reward = 0.10 if is_valid else -0.10

    total = env_reward + format_reward

    return {
        "total":   total,
        "env":     env_reward,
        "format":  format_reward,
    }


# ════════════════════════════════════════════════════════
# EPISODE ROLLOUT
# ════════════════════════════════════════════════════════

def obs_to_dict(obs_obj) -> dict:
    """Convert DataCentricObservation to dict for compatibility with reward logic."""
    if isinstance(obs_obj, dict):
        return obs_obj
    return {
        "response":                obs_obj.response,
        "current_accuracy":        obs_obj.current_accuracy,
        "baseline_accuracy":       obs_obj.baseline_accuracy,
        "target_accuracy":         obs_obj.target_accuracy,
        "estimated_quality":       obs_obj.estimated_quality,
        "dataset_shape":           obs_obj.dataset_shape,
        "rows_preserved_pct":      obs_obj.rows_preserved_pct,
        "budget_remaining":        obs_obj.budget_remaining,
        "step_number":             obs_obj.step_number,
        "max_steps":               obs_obj.max_steps,
        "active_session":          obs_obj.active_session,
        "validate_calls_remaining":obs_obj.validate_calls_remaining,
        "done":                    obs_obj.done,
        "reward":                  obs_obj.reward,
    }


def run_episode(
    model, tokenizer, task: str, seed: int
) -> tuple:
    """
    Run one complete episode using the WebSocket client (stateful session).
    Each reset+step sequence maintains the same env instance on the server.
    Returns: (prompts, responses, rewards) for GRPO training.
    """
    prompts, responses, rewards = [], [], []
    action_history = []

    try:
        with DataCentricEnv(base_url=BASE_URL).sync() as env:
            reset_result = env.reset(task=task, seed=seed)
            obs = obs_to_dict(reset_result.observation)

            while not obs.get("done", False):
                # Build chat prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": build_user_prompt(obs)},
                ]
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    max_length=MAX_SEQ_LENGTH - 60,
                    truncation=True,
                    add_generation_prompt=True,
                ).to(model.device)

                # Generate — commands are short, 50 tokens max
                with torch.no_grad():
                    output_ids = model.generate(
                        input_ids,
                        max_new_tokens=50,
                        temperature=0.8,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                response_text = tokenizer.decode(
                    output_ids[0][input_ids.shape[1]:],
                    skip_special_tokens=True,
                ).strip().split("\n")[0].strip()[:200]

                obs_before = obs
                try:
                    step_result = env.step(DataCentricAction(message=response_text))
                    obs = obs_to_dict(step_result.observation)
                except Exception as e:
                    obs = {**obs, "done": True, "reward": -0.05,
                           "response": f"Step error: {e}"}

                reward_dict = compute_rewards(
                    obs_before, obs, response_text, action_history
                )
                prompts.append(build_user_prompt(obs_before))
                responses.append(response_text)
                rewards.append(reward_dict)
                action_history.append(response_text)

    except Exception as e:
        print(f"Episode error (task={task}, seed={seed}): {e}")
        return [], [], []

    return prompts, responses, rewards


# ════════════════════════════════════════════════════════
# LOGGING
# ════════════════════════════════════════════════════════

training_log = []


def log_training_step(
    step: int, all_episodes: list, scheduler: CurriculumScheduler
):
    """Log metrics and sample generations every 10 steps."""
    all_final_rewards = []
    all_reward_components: dict = {"env": [], "format": []}
    format_hits = 0
    total_actions = 0

    for prompts, responses, rewards in all_episodes:
        if not rewards:
            continue
        all_final_rewards.append(rewards[-1]["total"])
        for r in rewards:
            for k in all_reward_components:
                all_reward_components[k].append(r[k])
            if r["format"] > 0:
                format_hits += 1
            total_actions += 1

    if not all_final_rewards:
        return

    entry = {
        "step":               step,
        "stage":              scheduler.stage_label(),
        "task":               scheduler.get_task(),
        "mean_total_reward":  mean(all_final_rewards),
        "mean_env_reward":    mean(all_reward_components["env"]),
        "mean_format_reward": mean(all_reward_components["format"]),
        "format_rate":        format_hits / max(total_actions, 1),
    }
    training_log.append(entry)
    json.dump(training_log, open("training_log.json", "w"), indent=2)

    print(
        f"Step {step:4d} | Stage: {entry['stage']:8s} | "
        f"Reward: {entry['mean_total_reward']:+.3f} | "
        f"Format: {entry['format_rate']:.0%}"
    )

    # Sample 3 generations for inspection
    if step % 10 == 0:
        samples = []
        for p_ep, r_ep, rw_ep in all_episodes[:3]:
            if p_ep and r_ep:
                samples.append({
                    "step":          step,
                    "response":      r_ep[-1],
                    "reward":        rw_ep[-1]["total"],
                    "reward_detail": rw_ep[-1],
                })
        with open("generations.jsonl", "a", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")


def log_episode_jsonl(
    episode: int, task: str, level: int, reward: float,
    accuracy_gain: float, steps_used: int, success: bool,
    log_path: str = "logs/training.jsonl",
):
    """Write one episode record to JSONL log (read by plot_rewards.py)."""
    import os as _os
    _os.makedirs(_os.path.dirname(log_path), exist_ok=True)
    entry = {
        "ts":            time.time(),
        "episode":       episode,
        "task":          task,
        "level":         level,
        "reward":        round(reward, 4),
        "accuracy_gain": round(accuracy_gain, 4),
        "steps_used":    steps_used,
        "success":       success,
    }
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ════════════════════════════════════════════════════════
# GRPO TRAINING LOOP
# ════════════════════════════════════════════════════════

def run_grpo_training(model, tokenizer, resume_from_checkpoint=None):
    print("\n=== PHASE 2: GRPO TRAINING ===")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")

    print("[Tracking] TensorBoard experiment tracking ON — logs written to ./logs/grpo")

    scheduler = CurriculumScheduler()

    grpo_config = GRPOConfig(
        output_dir="./data-centric-checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_steps=20,
        logging_steps=10,
        save_steps=50,
        num_generations=4,
        max_completion_length=50,   # renamed from max_new_tokens in TRL ≥0.15
        max_prompt_length=900,
        report_to="tensorboard",    # experiment tracking via TensorBoard
        logging_dir="./logs/grpo",
    )

    def reward_fn(completions, prompts=None, **kwargs):
        """
        Reward function called by GRPOTrainer.
        Runs live episodes and returns total reward for each completion.
        """
        batch_rewards = []
        episodes_this_batch = []

        for completion in completions:
            # Capture task BEFORE running episode so log reflects what was run
            task = scheduler.get_task()
            seed = random.randint(0, 9999)

            prompts_ep, responses_ep, rewards_ep = run_episode(
                model, tokenizer, task, seed
            )

            if rewards_ep:
                final_reward = sum(r["total"] for r in rewards_ep)
                accuracy_gain = sum(r["env"] for r in rewards_ep)
                success = accuracy_gain > 0.05
                # Update curriculum using success-rate based scheduler
                scheduler.record_episode(
                    reached_target=success,
                    accuracy_gain=accuracy_gain,
                )
            else:
                final_reward = -0.10
                accuracy_gain = 0.0
                success = False
                scheduler.record_episode(reached_target=False, accuracy_gain=0.0)

            # Write per-episode JSONL record for plot_rewards.py
            log_episode_jsonl(
                episode=scheduler.global_step,
                task=task,
                level=scheduler.current_level,
                reward=final_reward,
                accuracy_gain=accuracy_gain,
                steps_used=len(rewards_ep) if rewards_ep else 0,
                success=success,
            )

            batch_rewards.append(final_reward)
            episodes_this_batch.append((prompts_ep, responses_ep, rewards_ep))

        # Log every 10 calls
        if scheduler.global_step % 10 == 0:
            log_training_step(
                scheduler.global_step,
                episodes_this_batch,
                scheduler,
            )

        return batch_rewards

    # GRPOTrainer needs a prompt dataset — use SFT data as source
    raw = [json.loads(l) for l in open("sft_data.jsonl", encoding="utf-8")]
    grpo_dataset = Dataset.from_list([
        {"prompt": ex["prompt"]} for ex in raw
    ])

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_funcs=[reward_fn],
        args=grpo_config,
        train_dataset=grpo_dataset,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print("GRPO training complete.\n")
    return model


# ════════════════════════════════════════════════════════
# SAVE MODEL
# ════════════════════════════════════════════════════════

def save_model(model, tokenizer):
    print("Saving model...")

    # Save LoRA adapter (safe for 4-bit, fast)
    model.save_pretrained("./data-centric-adapter")
    tokenizer.save_pretrained("./data-centric-adapter")
    print("Adapter saved to ./data-centric-adapter")

    # Save merged 16-bit for inference
    # IMPORTANT: use unsloth's method — NOT naive merge_and_unload()
    # Naive merge on 4-bit model corrupts weights
    model.save_pretrained_merged(
        "./data-centric-merged",
        tokenizer,
        save_method="merged_16bit",
    )
    print("Merged model saved to ./data-centric-merged")
    print("Test inference immediately before demo.")


# ════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Ensure SFT warmup data exists
    if not os.path.exists("sft_data.jsonl"):
        print("Generating SFT data first...")
        subprocess.run(["python", "sft_generator.py"], check=True)

    # Start environment server
    server_proc = start_server()

    try:
        # Load base model with LoRA
        model, tokenizer = load_model()

        # Phase 1: SFT warmup — teaches valid command grammar
        model = run_sft_warmup(model, tokenizer)

        # Phase 2: GRPO — improves strategy via environment reward
        model = run_grpo_training(model, tokenizer)

        # Save adapter + merged 16-bit
        save_model(model, tokenizer)

        print("\nTraining complete. Run eval_data_centric.py next.")

    finally:
        stop_server(server_proc)
