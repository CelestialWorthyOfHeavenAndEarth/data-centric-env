---
title: Data-Centric AI RL Environment
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - data-centric-ai
  - grpo
  - unsloth
---

# 🧠 Data-Centric AI — Multi-Agent RL Environment

An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement learning environment that trains an LLM to act as a **data engineering orchestrator** — coordinating 4 specialist sub-agents across multi-step plans to improve ML datasets under budget constraints.

> **Core insight:** In traditional ML, practitioners tune models to squeeze out performance. This environment flips that — the model architecture is deliberately **frozen**, forcing the LLM agent to master *data engineering* as its only lever: diagnosing noise, coordinating specialist agents, and strategically transforming the dataset until accuracy surpasses the target. This is [Data-Centric AI](https://datacentricai.org/) — the paradigm Andrew Ng argues matters more than model architecture.

> **Live Space:** https://huggingface.co/spaces/Aswini-Kumar/data-centric-env

### Key Capabilities

| Capability | How it works |
|---|---|
| **Multi-Agent Coordination** | LLM orchestrates 4 specialist agents (cleaner, augmenter, balancer, validator) — deciding *who* to call and *when*, modeling each specialist's strengths |
| **Long-Horizon Planning** | 30-step budget with sparse terminal reward. Agent must plan inspect → query → apply → validate → submit sequences with delayed feedback |
| **Theory-of-Mind Reasoning** | Agent infers which specialist is best for the current data problem (class imbalance vs. missing values vs. outliers) |
| **Anti-Exploit Hardening** | 9 security mechanisms (immutable ground truth, golden rows, cooldowns, budget caps) prevent reward hacking |
| **Curriculum Learning** | Auto-advances from tutorial → easy → medium → hard based on rolling success rate |
| **Composable Reward System** | 4-component rubric (accuracy + process + preservation + efficiency) using OpenEnv's `Rubric` base class |

---

## 🎯 What the Agent Does

The agent receives a noisy tabular dataset and a fixed classifier. It must orchestrate specialist sub-agents to clean, augment, and balance the data until accuracy hits a target — **without touching the model**.

Each episode:
1. Agent **inspects** the dataset and model
2. Agent **queries** specialist sub-agents for recommendations
3. Agent **applies** the best fix (or **undoes** a bad one)
4. Agent **validates** accuracy improvement
5. Agent **submits** when target is reached or budget runs out

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     LLM Agent (Qwen2.5-3B)                      │
│            SFT warmup → GRPO live-environment training          │
└─────────────┬───────────────────────────────────┬───────────────┘
              │  text commands                    │  structured obs
              ▼                                   ▲
┌─────────────────────────────────────────────────────────────────┐
│              DataCentricEnvironment (OpenEnv)                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Cleaner  │  │Augmenter │  │ Balancer │  │  Validator   │    │
│  │  Agent   │  │  Agent   │  │  Agent   │  │   Agent      │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘    │
│       └──────────────┴──────────────┴───────────────┘            │
│                           │                                      │
│              ┌────────────▼────────────┐                         │
│              │   Working Copy (mutable) │◄── Snapshot stack (×3) │
│              └────────────┬────────────┘    for undo support     │
│                           │                                      │
│              ┌────────────▼────────────┐                         │
│              │  ModelEvaluator (RF)     │                         │
│              │  n_est=20 (fast_mode)   │                         │
│              └────────────┬────────────┘                         │
│                           │                                      │
│              ┌────────────▼────────────┐                         │
│              │  Ground Truth (frozen)  │ ← never mutated         │
│              └─────────────────────────┘                         │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │  DataCentricRubric (OpenEnv Rubric system)                │   │
│  │  ├── AccuracyRubric      — Δ accuracy vs baseline         │   │
│  │  ├── ProcessRubric       — workflow pattern scoring       │   │
│  │  ├── PreservationRubric  — row preservation incentive     │   │
│  │  └── EfficiencyRubric    — accuracy gain / budget used    │   │
│  │  + StepRubric            — dense per-apply proxy reward   │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                  │
│  Anti-Exploit: 9 protections (GT immutability, cooldowns, etc.)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🌍 Environment Design

### Action Space
Single text command — one of:

| Command | Effect |
|---------|--------|
| `inspect_dataset` | View shape, missing values, class distribution |
| `inspect_model` | View classifier accuracy, precision, recall, F1 |
| `query_cleaner` | Get missing-value / outlier fix recommendations |
| `query_augmenter [class]` | Get data augmentation recommendations |
| `query_balancer` | Get class rebalancing recommendations |
| `query_validator` | Check rule violations (costs 2 budget) |
| `apply <N>` | Apply recommendation number N |
| `reject <N>` | Reject a recommendation |
| `undo` | Revert last apply (max 3 levels deep) |
| `validate` | Retrain classifier and score (cooldown applies) |
| `submit` | Finalize and score the episode |

### Observation Space
Each step returns a structured observation:

```python
DataCentricObservation(
    response="...",              # Specialist agent's text response
    current_accuracy=0.71,       # Current classifier accuracy
    baseline_accuracy=0.62,      # Accuracy before any changes
    target_accuracy=0.73,        # Target to hit
    estimated_quality=0.84,      # Dataset quality score (0-1)
    rows_preserved_pct=0.97,     # % of original rows still present
    budget_remaining=22,         # Steps remaining
    validate_calls_remaining=2,  # Free validate calls left
    active_session="cleaner",    # Which specialist is active
    done=False,
)
```

### Reward Function — OpenEnv Rubric System

Uses `openenv.core.rubrics.base.Rubric` with composable child rubrics (nn.Module-style auto-registration):

| Rubric | Signal | Range |
|--------|--------|-------|
| **AccuracyRubric** | Δ accuracy × 2.5 + submit bonus | [-1.0, +1.0] |
| **ProcessRubric** | Correct query→apply→validate sequencing | [-0.10, +0.05] |
| **PreservationRubric** | Rows preserved ≥ 90% | [-0.40, +0.05] |
| **EfficiencyRubric** | Accuracy gain / budget used (submit only) | [-0.05, +0.20] |
| **StepRubric** | Dense per-apply quality proxy | [-0.30, +0.15] |

Total clamped to **[-1.0, 1.0]** by `DataCentricRubric.forward()`.

### Anti-Exploit Protections
9 hardened mechanisms including:
- Ground truth immutability assertion after every `apply`
- Validate cooldown enforcement (must take 2 actions between validates)
- Duplicate apply detection + session apply limit (max 3 per query)
- Recommendation staleness validation (re-query after each session)
- Catastrophic data loss detection (< 50% rows → terminate)
- Episode wall-clock timeout (5 min → forced submit)
- Input truncation (> 200 chars → truncate + penalty)

---

## 📚 Task Curriculum (4 Levels)

| Task | Rows | Issues | Baseline | Target | Budget |
|------|------|--------|----------|--------|--------|
| `task_0_tutorial` | 100 | Missing values (20%) | ~0.62 | 0.73 | 30 |
| `task_1_easy` | 200 | Missing + imbalance | ~0.63 | 0.79 | 25 |
| `task_2_medium` | 500 | Missing + duplicates + imbalance + type errors | ~0.58 | 0.74 | 40 |
| `task_3_hard` | 900 | 6 issues incl. outliers + logic errors | ~0.54 | 0.71 | 60 |

Advancement criterion: ≥ 60% success rate over a rolling 30-episode window.

---

## 🤖 Training Pipeline

**Model:** Qwen2.5-3B-Instruct (4-bit via Unsloth)  
**Algorithm:** SFT warmup → GRPO (TRL)  
**Framework:** OpenEnv + TRL + Unsloth

### Phase 1 — SFT Warmup
Train on ~8,100 heuristic trajectory examples to teach valid command syntax before RL.

### Phase 2 — GRPO
Live environment rollouts scored by the composable Rubric system. Curriculum scheduler advances from tutorial → easy → medium → hard as performance improves.

### Training Notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CelestialWorthyOfHeavenAndEarth/data-centric-env/blob/main/train_colab.ipynb)

[`train_colab.ipynb`](train_colab.ipynb) — complete end-to-end training pipeline for Colab T4 GPU.

---

## 📊 Results

### Heuristic Baseline Verification

The heuristic agent (`inference.py`) validates that the environment is solvable:

| Task | Accuracy Gain | Target Hit Rate |
|------|--------------|-----------------|
| `task_0_tutorial` | +0.11 | ✓ 100% |
| `task_1_easy` | +0.08 | ✓ 80% |
| `task_2_medium` | +0.06 | ✓ 60% |
| `task_3_hard` | +0.04 | ~ 40% |

> **Note:** After GRPO training, embed reward curves from `plots/` here.
> Run `python plot_rewards.py` to generate: `reward_curve.png`, `success_rate.png`, `accuracy_gain.png`, `curriculum.png`.

---

## 🧪 Testing

```bash
# Run all tests (35 tests — grader + environment)
pytest tests/ -v

# Run only grader tests (22 tests)
pytest tests/test_grader.py -v

# Run only environment safety tests (13 tests)
pytest tests/test_environment.py -v
```

Tests cover:
- All 5 Rubric components (accuracy, process, preservation, efficiency, step)
- Reward clamping to declared [-1.0, 1.0] range
- Ground truth immutability after every command
- Budget enforcement and episode termination
- Validate cooldown and call limiting
- Undo/snapshot stack behavior
- Unknown command handling

---

## 🚀 Quick Start

### Connect to the Live Space

```python
from client import DataCentricEnv
from models import DataCentricAction

with DataCentricEnv(base_url="https://aswini-kumar-data-centric-env.hf.space").sync() as env:
    result = env.reset(task="task_0_tutorial", seed=42)
    print(f"Baseline: {result.observation.baseline_accuracy:.2f}  Target: {result.observation.target_accuracy:.2f}")

    result = env.step(DataCentricAction(message="inspect_dataset"))
    print(result.observation.response)

    result = env.step(DataCentricAction(message="query_cleaner"))
    print(result.observation.response)
```

### Run Locally

```bash
# Install
pip install openenv-core fastapi uvicorn scikit-learn pandas numpy

# Start server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# In another terminal
python -c "
from client import DataCentricEnv
from models import DataCentricAction
with DataCentricEnv(base_url='http://localhost:8000').sync() as env:
    obs = env.reset(task='task_0_tutorial', seed=42).observation
    print(f'Ready — baseline={obs.baseline_accuracy:.2f} target={obs.target_accuracy:.2f}')
"
```

---

## 📁 Project Structure

```
data_centric_env/
├── openenv.yaml              # OpenEnv manifest (tasks, reward range, action/obs types)
├── client.py                 # DataCentricEnv WebSocket client
├── models.py                 # DataCentricAction + DataCentricObservation (Pydantic)
├── train_data_centric.py     # Full SFT → GRPO training pipeline
├── train_colab.ipynb         # Colab training notebook (T4 GPU)
├── eval_data_centric.py      # Evaluation: random vs trained agent
├── plot_rewards.py           # Reward curve visualization (4 plots)
├── sft_generator.py          # SFT warmup data generator (~8100 examples)
├── inference.py              # Heuristic baseline agent
├── tests/
│   ├── test_grader.py        # 22 tests — Rubric system + reward components
│   └── test_environment.py   # 13 tests — safety invariants + anti-exploit
└── server/
    ├── app.py                # FastAPI server (HTTP + WebSocket via OpenEnv)
    ├── data_centric_environment.py   # Core RL environment logic (680 lines)
    ├── dataset_generator.py  # Synthetic dataset generation (4 task configs)
    ├── specialist_agents.py  # CleanerAgent, AugmenterAgent, BalancerAgent, ValidatorAgent
    ├── grader.py             # Composable Rubric system (openenv.core.rubrics.base)
    ├── anti_exploit.py       # 9 anti-reward-hacking protections
    ├── model_evaluator.py    # RF classifier with hash-based caching
    └── Dockerfile            # HuggingFace Spaces deployment
```

---

## 🏷️ Hackathon

**Theme:** #3.1 — World Modeling / Professional Tasks  
**Stack:** OpenEnv · Unsloth · TRL (GRPO) · FastAPI · scikit-learn  
**Repo:** [github.com/CelestialWorthyOfHeavenAndEarth/data-centric-env](https://github.com/CelestialWorthyOfHeavenAndEarth/data-centric-env)  
**Space:** [huggingface.co/spaces/Aswini-Kumar/data-centric-env](https://huggingface.co/spaces/Aswini-Kumar/data-centric-env)
