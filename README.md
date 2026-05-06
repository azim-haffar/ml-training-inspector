# ML Training Inspector

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?style=flat-square&logo=react&logoColor=black)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=flat-square&logo=docker&logoColor=white)

> **Real-time ML training dashboard — live metrics, anomaly detection, and early stopping**

## Demo

[Watch demo on YouTube](https://youtu.be/x7-KYXCESMw)

## Motivation

I kept running into the same problem: kick off a training run, come back an hour later, and have no idea what actually happened. Loss went down — great — but *when* did it plateau? Were gradients healthy the whole time, or did they quietly die in epoch 3?

TensorBoard felt like overkill for what I needed. I wanted something I could spin up instantly that shows me the stuff I actually care about while a run is happening, not after it finishes. So I built this.

It streams training metrics over a WebSocket in real time, flags when something looks off, and lets you stop a run early and save a checkpoint if you've already seen what you needed.

## What it does

- **Live loss and accuracy charts** — batch-level stream and epoch-level train/val curves
- **Per-layer gradient norms** — bar chart that turns orange/red when gradients look suspiciously small
- **Anomaly detection** — flags vanishing/exploding gradients, overfitting, and loss/accuracy plateaus as they happen
- **Stop training** — interrupt a run at any point; automatically saves a checkpoint with model + optimizer state
- **Auto-checkpoint** — saves `model_epoch_N.pth` on completion or early stop

The model is a 3-layer CNN trained on CIFAR-10, reaching ~75–78% val accuracy in 10 epochs — enough for interesting training dynamics to observe.

## Tech Stack

| Layer    | Technology                        |
| -------- | --------------------------------- |
| Training | PyTorch, CIFAR-10 via torchvision |
| API      | FastAPI, WebSocket streaming      |
| Frontend | React + Vite, Recharts            |
| Infra    | Docker Compose                    |

The training loop runs in a background thread. Metrics get broadcast to all connected WebSocket clients via per-client asyncio queues — multiple browser tabs all get the same stream without contention.

## How to Run

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

## Project Structure

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

## Licence

MIT © 2026 Azim Haffar
