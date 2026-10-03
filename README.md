# ML Training Inspector

A PyTorch training dashboard with FastAPI, WebSockets, React and Recharts. Observe training and evaluation loss, accuracy, sampled gradients, per-class accuracy and the learning rate used in each epoch. Choose SimpleCNN or ResNet9, stop a run manually and save a model snapshot.

[Existing walkthrough](https://youtu.be/x7-KYXCESMw) · [Verification record](ENGINEERING_NOTES.md)

## Quick reproducible CPU demo

Use Python 3.12 and Node.js 22.12 or newer. From the repository root:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r backend/requirements-demo.txt
python scripts/demo.py
python -m pytest -q
```

The script trains SimpleCNN for two epochs with seed 42, batch size 32, 160 synthetic training images and 80 validation images. It writes actual events to `demo-output/metrics.json` and a snapshot to `demo-output/checkpoints/`. Synthetic class stripes are deliberately easy to learn; this demonstrates application plumbing, not a real-world quality benchmark. Timing excludes Python startup and varies by machine. Reproducibility is verified on the tested CPU environment; equality across hardware/library versions is not promised.

## Interactive demo

```sh
# Terminal 1, repository root, activated environment
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000 --workers 1

# Terminal 2
cd frontend
npm ci
npm test
npm run dev -- --host 127.0.0.1
```

Open http://localhost:5173. The dashboard defaults to the two-epoch synthetic CPU demo. Start training, observe the charts and expand the received-metrics table. Open a second tab to observe the same shared run. For stop testing, select 50 epochs and press Stop. Start again to verify resources are released. Export JSON after completion to inspect the chart inputs.

Choose CIFAR-10 for the public dataset (about 170 MB on first use). The dataset is cached under `backend/data`. CPU runs on the full dataset take substantially longer. No paid compute is required.

Alternatively, `docker compose up --build` starts the dashboard at http://localhost:3000 and the API at http://localhost:8000. Compose binds only to loopback and uses one backend worker. Container startup was not verified in this environment.

## Metric meaning

- Loss is mean cross-entropy weighted by example count, including an uneven final batch. Training accuracy counts correct predictions over all processed training examples.
- Training metrics accumulate during optimizer updates with dropout and augmentation active. Evaluation uses `eval()` and no gradients; those are different measurement conditions.
- The live chart displays individual sampled batch losses, every batch in the synthetic demo and every twentieth/final batch for CIFAR-10. Its last 120 points are retained. `running_loss` is a separate cumulative field.
- The LR chart displays the rate used during that epoch, before the scheduler steps.
- Gradients are parameter L2 norms sampled from the last backward pass, not epoch averages. Threshold signals are heuristics, with no measured detection accuracy. Near-zero biases before BatchNorm can trigger benign alerts.
- Per-class accuracy is evaluation accuracy; classes without examples have null accuracy, not zero.

## Checkpoints and lifecycle

New snapshots use unique run IDs and atomic replacement. They contain model and optimizer state, model/dataset/seed metadata, attempted epoch and fully completed epoch count. Saving and loading state are tested. **Resume is not implemented**: scheduler, RNG and sampler position are absent, so a snapshot is explicitly marked `resumable: false`. An interrupted epoch must not be presented as a completed epoch.

One blocking PyTorch training thread sends metrics to the API event loop. Each client has a bounded queue of 256 messages. Disconnects cancel sender/receiver tasks and remove the queue; a slow subscriber is closed with code 1013. Reconnection receives current status, but earlier chart points are not replayed. Shutdown signals cancellation and joins the thread. Cancellation is checked between batches, including evaluation; dataset downloads and in-flight tensor operations are not immediately interruptible.

## Boundaries

- One worker, one shared run, CPU only through the API. Process-local state is not synchronized across workers. Every connected client observes the same run and can stop it.
- Local demonstration only: no authentication, authorization, per-user isolation or durable full event log. Do not expose the API publicly as-is.
- CIFAR-10 uses the official **test split** for per-epoch evaluation, despite dashboard labels saying validation. There is no separate unbiased final benchmark. Full CIFAR-10 training and model-quality claims are unverified.
- Fixed anomaly rules do not establish overfitting or gradient failure. Manual stopping is not automatic early stopping.
- Run summaries persist locally; full chart history is held in the browser and exported explicitly. Reconnect notices disclose incomplete histories.

## Verified demo

The two-epoch synthetic CPU demo was run locally and its dashboard values matched the emitted training metrics. Reloading the saved model reproduced the reported validation metrics. Backend tests cover uneven batches, cancellation, shutdown, reconnects, slow subscribers and two-client delivery; frontend tests check chart mappings and terminal states. See [the verification record](ENGINEERING_NOTES.md) for environment details and limitations.

## Checks

`python -m pytest -q` verifies metrics, seeded real CPU training, checkpoint loading, stopping, two-client WebSocket delivery, disconnect cleanup, queue limits, API errors, reconnect status, shutdown, configuration and heuristic boundaries. `npm test` checks chart value mappings and terminal states. `npm run build` verifies the frontend production build. CI runs these checks and the CPU demo; the remote CI result is not claimed until published and executed.
