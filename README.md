# ML Training Inspector
### Watch the learning, not just the loading bar.

A **PyTorch training dashboard** with a FastAPI backend, WebSocket telemetry, and React charts.

[Watch the walkthrough](https://youtu.be/x7-KYXCESMw) · [Training loop](backend/trainer.py) · [Anomaly rules](backend/anomaly.py) · [Tests](backend/tests)

## What you can observe

| Signal | What the dashboard shows |
| --- | --- |
| Learning | Training/evaluation loss and accuracy |
| Gradients | Per-layer norms and threshold alerts |
| Class behavior | Per-class CIFAR-10 accuracy |
| Progress | Batch/epoch updates streamed over WebSockets |
| Run control | Manual stop, checkpoint saving, recent-run history |

Choose **SimpleCNN or ResNet9**. The trainer runs in a background thread and broadcasts updates to browser clients. Anomaly detection uses explicit thresholds for gradient extremes, overfitting signals, and plateaus; it is not a learned detector. Failed or stopped runs do not emit a successful-completion event.

```text
PyTorch training thread → FastAPI event loop → WebSocket → React charts
          ↓                      ↓
     checkpoints            run history
```

## Run locally

Native frontend development requires Node.js 22.12 or newer.

Requires Docker Compose. The first training run downloads CIFAR-10; CPU training can be slow. CUDA is selected when available in the runtime, but the default Compose setup does not configure GPU passthrough.

```sh
git clone https://github.com/azim-haffar/ml-training-inspector.git
cd ml-training-inspector
docker compose up --build
```

Dashboard: `http://localhost:3000` · API docs: `http://localhost:8000/docs`.

For native development, install `backend/requirements.txt` in a Python 3.11 virtual environment and run `uvicorn main:app --reload` from `backend`. In `frontend`, copy `.env.example` to `.env`, run `npm ci`, then `npm run dev`.

Run settings are validated: 1–100 epochs, batch size 1–1024, finite learning rate greater than 0 and at most 1, and one of the two supported models. These limits reject invalid inputs; they do not guarantee a run fits your hardware.

## Verify without downloading a dataset

```sh
# From backend
python -m pip install 'pydantic>=2,<3' 'fastapi>=0.111,<1' 'httpx>=0.27,<1'
python -m unittest discover -s tests -v

# From frontend
npm ci
npm run build
```

CI checks configuration validation, anomaly-rule boundaries, Python syntax, and the frontend build. These are not full training or model-quality benchmarks.

## Honest boundaries

- The UI calls evaluation metrics “validation,” but the trainer currently evaluates the official CIFAR-10 **test split after each epoch**. There is no separate held-out validation split; these metrics should not be reported as an unbiased final benchmark.
- No reproducible accuracy claim is made. Seeds and a benchmark protocol are not configured.
- Stop control is manual, not automatic early stopping. Checkpoints are saved; resume training is not exposed by the API.
- Training state is process-local: use one backend worker. There is no authentication, run isolation, or multi-user production hardening. Compose binds to localhost.
- Downloaded dependencies, Python caches, datasets, and generated checkpoints stay outside source control. Existing history is preserved; fresh clones no longer check out generated files.
