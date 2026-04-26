"""Full connectivity audit — run after any code changes."""
import ast, pathlib, sys

PASS = []
FAIL = []

def ok(msg):  PASS.append(msg); print(f"  PASS  {msg}")
def fail(msg, err): FAIL.append(msg); print(f"  FAIL  {msg}: {err}")

# 1. agent_utils
try:
    from agent_utils import VALID_COMMANDS, SYSTEM_PROMPT, build_user_prompt, start_server, stop_server
    ok(f"agent_utils — {len(VALID_COMMANDS)} commands, {len(SYSTEM_PROMPT)} char prompt")
except Exception as e:
    fail("agent_utils import", e)

# 2. Server core
try:
    from server.data_centric_environment import DataCentricEnvironment
    from server.grader import compute_total_reward
    from server.anti_exploit import validate_calls_remaining, AntiExploitState
    from server.model_evaluator import ModelEvaluator
    from server.specialist_agents import CleanerAgent, AnalystAgent, BalancerAgent, AugmenterAgent
    ok("server.* all imports")
except Exception as e:
    fail("server imports", e)

# 3. Client + models
try:
    from client import DataCentricEnv
    from models import DataCentricAction, DataCentricObservation
    ok("client + models")
except Exception as e:
    fail("client/models", e)

# 4. Script parse check (no heavy deps loaded)
for script in ["train_data_centric.py", "eval_data_centric.py", "sft_generator.py",
               "inference.py", "plot_rewards.py", "hf_job_train.py", "submit_job.py"]:
    try:
        ast.parse(pathlib.Path(script).read_text(encoding="utf-8"))
        ok(f"{script} syntax OK")
    except Exception as e:
        fail(f"{script} syntax", e)

# 5. Live environment cycle
try:
    from models import DataCentricAction
    env = DataCentricEnvironment()
    obs = env.reset(task="task_0_tutorial", seed=42)
    assert obs.validate_calls_remaining == 3, f"expected 3 got {obs.validate_calls_remaining}"
    assert obs.baseline_accuracy > 0
    ok(f"env.reset() — baseline={obs.baseline_accuracy:.4f}, vcr={obs.validate_calls_remaining}")
except Exception as e:
    fail("env.reset()", e)

try:
    obs = env.step(DataCentricAction(message="inspect_dataset"))
    ok(f"inspect_dataset — reward={obs.reward:.4f}")
except Exception as e:
    fail("step inspect_dataset", e)

try:
    obs = env.step(DataCentricAction(message="query_analyst"))
    ok(f"query_analyst — reward={obs.reward:.4f}")
except Exception as e:
    fail("step query_analyst", e)

try:
    obs = env.step(DataCentricAction(message="query_cleaner"))
    ok(f"query_cleaner — reward={obs.reward:.4f}")
except Exception as e:
    fail("step query_cleaner", e)

try:
    obs = env.step(DataCentricAction(message="apply 1"))
    ok(f"apply 1 — accuracy={obs.current_accuracy:.4f}")
except Exception as e:
    fail("step apply 1", e)

try:
    obs = env.step(DataCentricAction(message="validate"))
    ok(f"validate — accuracy={obs.current_accuracy:.4f}, vcr={obs.validate_calls_remaining}")
except Exception as e:
    fail("step validate", e)

try:
    obs = env.step(DataCentricAction(message="submit"))
    ok(f"submit — final reward={obs.reward:.4f}, done={obs.done}")
except Exception as e:
    fail("step submit", e)

# 6. agent_utils.build_user_prompt
try:
    obs2 = env.reset(task="task_1_easy", seed=0)
    prompt = build_user_prompt(obs2.__dict__)
    assert "Budget remaining" in prompt
    ok("build_user_prompt output valid")
except Exception as e:
    fail("build_user_prompt", e)

# Summary
print()
print("=" * 60)
print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print("FAILURES:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("ALL CHECKS PASSED")
