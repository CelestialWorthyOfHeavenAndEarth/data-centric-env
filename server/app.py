"""FastAPI application for the Data Centric Env Environment."""

import sys
import os

# Ensure the project root is on the path regardless of how the server is launched
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core is required. Install with: pip install openenv-core"
    ) from e

try:
    from ..models import DataCentricAction, DataCentricObservation
    from .data_centric_environment import DataCentricEnvironment
except (ImportError, ModuleNotFoundError):
    from models import DataCentricAction, DataCentricObservation
    from server.data_centric_environment import DataCentricEnvironment

from fastapi.responses import HTMLResponse

# max_concurrent_envs=1: avoids concurrency safety check that instantiates the env
# at startup (which would load sklearn and pandas, slowing HF health check).
# Increase if running on a paid Space with more RAM.
app = create_app(
    DataCentricEnvironment,
    DataCentricAction,
    DataCentricObservation,
    env_name="data_centric_env",
    max_concurrent_envs=1,
)

_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Data-Centric AI RL Environment</title>
<style>
  body { font-family: system-ui, sans-serif; background: #0f1117; color: #e0e0e0;
         display: flex; justify-content: center; padding: 60px 20px; margin: 0; }
  .card { max-width: 700px; width: 100%; }
  h1 { font-size: 2rem; margin-bottom: 4px; color: #fff; }
  .badge { display:inline-block; background:#238636; color:#fff; border-radius:12px;
           padding:2px 10px; font-size:0.8rem; margin-bottom:24px; }
  p { color: #aaa; line-height: 1.6; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 28px 0; }
  .endpoint { background: #1c1f26; border: 1px solid #30363d; border-radius: 8px;
              padding: 14px 18px; }
  .endpoint code { color: #58a6ff; font-size: 0.9rem; }
  .endpoint small { color: #666; display:block; margin-top:4px; }
  a { color: #58a6ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .footer { margin-top: 32px; font-size: 0.8rem; color: #555; border-top: 1px solid #21262d; padding-top: 16px; }
</style>
</head>
<body>
<div class="card">
  <h1>&#x1F9E0; Data-Centric AI Environment</h1>
  <span class="badge">&#x2022; Running</span>
  <p>An <a href="https://github.com/meta-pytorch/OpenEnv" target="_blank">OpenEnv</a>-compliant
  RL environment where a language model learns to orchestrate specialist data-repair agents
  — imputing, rebalancing, and augmenting corrupted pipelines to rehabilitate an
  <strong>inference-locked production model</strong>, purely through data-centric interventions.</p>

  <div class="grid">
    <div class="endpoint">
      <code>GET /health</code>
      <small>Server health check</small>
    </div>
    <div class="endpoint">
      <code>GET /docs</code>
      <small>Interactive API docs (Swagger)</small>
    </div>
    <div class="endpoint">
      <code>POST /reset</code>
      <small>Start a new episode</small>
    </div>
    <div class="endpoint">
      <code>POST /step</code>
      <small>Execute an action</small>
    </div>
    <div class="endpoint">
      <code>WS /ws</code>
      <small>WebSocket (stateful session)</small>
    </div>
    <div class="endpoint">
      <code>GET /state</code>
      <small>Current episode state</small>
    </div>
  </div>

  <p><strong>Quick start:</strong></p>
  <pre style="background:#161b22;padding:14px;border-radius:8px;font-size:0.85rem;overflow:auto">
pip install git+https://huggingface.co/spaces/Aswini-Kumar/data-centric-env

from data_centric_env import DataCentricEnv, DataCentricAction
with DataCentricEnv(base_url="https://aswini-kumar-data-centric-env.hf.space").sync() as env:
    obs = env.reset(task="task_0_tutorial", seed=42)
    result = env.step(DataCentricAction(message="query_analyst"))
    print(result.observation.response)</pre>

  <div class="footer">
    <a href="/docs">API Docs</a> &nbsp;|&nbsp;
    <a href="/health">Health</a> &nbsp;|&nbsp;
    <a href="https://github.com/CelestialWorthyOfHeavenAndEarth/data-centric-env" target="_blank">GitHub</a> &nbsp;|&nbsp;
    <a href="https://huggingface.co/spaces/Aswini-Kumar/data-centric-env" target="_blank">HF Space</a>
  </div>
</div>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
async def landing():
    """Human-readable landing page for the HF Space App tab."""
    return HTMLResponse(content=_LANDING_HTML)


def main(host: str = "0.0.0.0", port: int = 7860):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(host=args.host, port=args.port)
