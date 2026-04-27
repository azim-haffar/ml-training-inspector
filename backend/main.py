import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Training Inspector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global state ---
# Each WebSocket client gets its own queue so all clients receive every message.
_client_queues: list[asyncio.Queue] = []
_event_loop: Optional[asyncio.AbstractEventLoop] = None
_training_active = False
# Note: _stop_event is process-local. This works for single-worker dev use,
# but would need Redis/shared state for multi-worker deployments.
_stop_event = threading.Event()

HISTORY_PATH = "./data/history.json"


class TrainingConfig(BaseModel):
    epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 0.001
    model: str = "simple_cnn"  # "simple_cnn" | "resnet9"


@app.on_event("startup")
async def startup():
    global _event_loop
    _event_loop = asyncio.get_event_loop()


def _broadcast(data: dict):
    """Push a message to every connected client (safe to call from a thread)."""
    if _event_loop is None:
        return
    for q in list(_client_queues):
        _event_loop.call_soon_threadsafe(q.put_nowait, data)


def _save_history(run_info: dict, config: TrainingConfig):
    """Append a run summary to history.json, keeping the last 20 entries."""
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    last = run_info.get("last_epoch") or {}
    duration = round(time.time() - run_info["start_time"])

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": config.model,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.learning_rate,
        "final_train_acc": last.get("train_acc"),
        "final_val_acc": last.get("val_acc"),
        "final_loss": last.get("train_loss"),
        "anomaly_count": run_info["total_anomalies"],
        "duration_seconds": duration,
    }
    history.append(entry)
    history = history[-20:]  # keep at most 20, endpoint returns last 5

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"History saved ({len(history)} entries)")


@app.post("/api/start")
async def start_training(config: TrainingConfig):
    global _training_active
    if _training_active:
        return {"error": "Training is already running"}

    _training_active = True
    _stop_event.clear()

    def run():
        global _training_active
        run_info = {"start_time": time.time(), "total_anomalies": 0, "last_epoch": None}

        # Wrap broadcast so we can track run metrics for history
        def tracking_callback(data: dict):
            if data.get("type") == "epoch":
                run_info["last_epoch"] = data
                run_info["total_anomalies"] += len(data.get("anomalies", []))
            _broadcast(data)

        try:
            trainer = Trainer(
                epochs=config.epochs,
                batch_size=config.batch_size,
                lr=config.learning_rate,
                model_name=config.model,
                stop_event=_stop_event,
                checkpoint_dir="./checkpoints",
            )
            trainer.train(callback=tracking_callback)
        except Exception as e:
            logger.error(f"Training crashed: {e}", exc_info=True)
            _broadcast({"type": "error", "message": str(e)})
        finally:
            _training_active = False
            # Only save history if we got at least one epoch
            if run_info["last_epoch"] is not None:
                try:
                    _save_history(run_info, config)
                except Exception as e:
                    logger.warning(f"Could not save history: {e}")
            _broadcast({"type": "done"})
            logger.info("Training thread finished")

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    logger.info(f"Training started — model={config.model} epochs={config.epochs}")

    return {"status": "started"}


@app.post("/api/stop")
async def stop_training():
    if not _training_active:
        return {"error": "No training is currently running"}
    _stop_event.set()
    logger.info("Stop signal sent")
    return {"status": "stopping"}


@app.get("/api/status")
async def get_status():
    return {"is_running": _training_active}


@app.get("/api/history")
async def get_history():
    try:
        with open(HISTORY_PATH) as f:
            history = json.load(f)
        return {"history": history[-5:]}
    except FileNotFoundError:
        return {"history": []}
    except json.JSONDecodeError:
        return {"history": []}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue()
    _client_queues.append(queue)
    logger.info(f"Client connected — {len(_client_queues)} client(s) active")

    try:
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=10.0)
                await websocket.send_json(data)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat"})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        _client_queues.remove(queue)
        logger.info(f"Client disconnected — {len(_client_queues)} client(s) remaining")
