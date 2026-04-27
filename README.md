# ML Training Inspector

## Motivation

I kept running into the same annoying problem: kick off a training run, go do something else for an hour, come back and have no idea what actually happened. Loss went down — great — but *when* did it plateau? Were gradients healthy the whole time, or did they quietly die in epoch 3?

I looked at TensorBoard but it felt like overkill for what I needed. I wanted something I could spin up instantly that shows me the stuff I actually care about while a run is happening, not after it finishes. So I built this.

It streams training metrics over a WebSocket in real time, flags when something looks off, and lets you stop a run early and save a checkpoint if you've already seen what you needed.

## What it does

- **Live loss and accuracy charts** — batch-level stream and epoch-level train/val curves
- **Per-layer gradient norms** — bar chart that turns orange/red when gradients look suspiciously small
- **Anomaly detection** — flags vanishing/exploding gradients, overfitting, and loss/accuracy plateaus as they happen
- **Stop training** — interrupt a run at any point; automatically saves a checkpoint with model + optimizer state
- **Auto-checkpoint** — saves `model_epoch_N.pth` on completion or early stop

The model is a simple 3-layer CNN trained on CIFAR-10. Gets to ~75–78% val accuracy in 10 epochs, which is good enough for interesting training dynamics to observe.

## Tech stack

| Layer    | What                              |
| -------- | --------------------------------- |
| Training | PyTorch, CIFAR-10 via torchvision |
| API      | FastAPI, WebSocket streaming      |
| Frontend | React + Vite, Recharts            |
| Infra    | Docker Compose                    |

The training loop runs in a background thread. Metrics get broadcast to all connected WebSocket clients via per-client asyncio queues, which means multiple browser tabs all get the same stream without stealing from each other.

## How to run

```bash
docker-compose up
```

Open [http://localhost:3000](http://localhost:3000), configure the run, and click **Start Training**.

The first run downloads CIFAR-10 (~170 MB) — takes a minute to start. The dataset is cached in a Docker volume so subsequent runs start immediately.

**Without Docker:**

```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (separate terminal)
cd frontend
cp .env.example .env
npm install
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173).

Checkpoints are saved to `backend/checkpoints/model_epoch_N.pth`.

## Project structure

```text
├── backend/
│   ├── main.py        # FastAPI server — WebSocket, /start, /stop endpoints
│   ├── trainer.py     # CNN model + training loop with stop_event support
│   ├── anomaly.py     # Threshold-based anomaly checks
│   ├── .env.example
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── App.jsx
│       ├── components/
│       │   ├── LossChart.jsx        # Epoch train/val loss
│       │   ├── BatchLossChart.jsx   # Live scrolling batch loss
│       │   ├── AccuracyChart.jsx
│       │   ├── GradientChart.jsx    # Per-layer gradient norms (colour-coded)
│       │   ├── AnomalyPanel.jsx     # Dismissible alert cards
│       │   └── ProgressBar.jsx
│       └── hooks/
│           └── useTrainingSocket.js # WebSocket hook with auto-reconnect
├── docker-compose.yml
└── README.md
```

## What I learned / what's rough

The async + threading interaction was trickier than expected. PyTorch's training loop is blocking, so it runs in a thread, and metrics need to be pushed back to asyncio's event loop safely. `call_soon_threadsafe` solves it but it took a while to figure out why naive queue usage was dropping messages.

The anomaly detection is purely threshold-based — fixed cutoffs for gradient norms, loss gaps, etc. Something smarter would track the *rate of change* and use rolling statistics, but that starts getting into signal processing territory that's beyond what I know right now.

Hot reload in Docker on Windows requires Vite's `usePolling: true` because the filesystem events don't propagate through the virtualization layer. Spent longer than I'd like to admit debugging that.
