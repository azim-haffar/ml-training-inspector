import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config import TrainingConfig

from trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    global _event_loop
    _event_loop = asyncio.get_running_loop()
    try:
        yield
    finally:
        _stop_event.set()
        if _training_thread is not None:
            await asyncio.to_thread(_training_thread.join)
        _event_loop = None

app = FastAPI(title="ML Training Inspector API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
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

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = str(BASE_DIR / "data" / "history.json")
_training_thread = None
_run_status = "idle"
_latest_start = None
_latest_epoch = None
_latest_checkpoint = None
_latest_error = None


def _broadcast(data: dict):
    """Mutate client queues only on the event loop; evict slow subscribers."""
    def deliver():
        for queue in list(_client_queues):
            if queue.full():
                # Closing the subscriber forces status recovery instead of silently losing metrics.
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "overflow"})
                _client_queues.remove(queue)
            else:
                queue.put_nowait(data)
    if _event_loop is not None and not _event_loop.is_closed():
        _event_loop.call_soon_threadsafe(deliver)


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
        "completed_epochs": last.get("epoch", 0),
        "dataset": config.dataset,
        "seed": config.seed,
        "status": _run_status,
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

    with open(HISTORY_PATH + ".tmp", "w") as f:
        json.dump(history, f, indent=2)
    os.replace(HISTORY_PATH + ".tmp", HISTORY_PATH)

    logger.info(f"History saved ({len(history)} entries)")


@app.post("/api/start")
async def start_training(config: TrainingConfig):
    global _training_active, _training_thread, _run_status, _latest_start, _latest_epoch, _latest_checkpoint, _latest_error
    if _training_active:
        raise HTTPException(409, "Training is already running")

    _run_status = "starting"
    _latest_start = _latest_epoch = _latest_checkpoint = _latest_error = None
    _training_active = True
    _stop_event.clear()

    def run():
        global _training_active, _run_status, _latest_start, _latest_epoch, _latest_checkpoint, _latest_error
        run_info = {"start_time": time.time(), "total_anomalies": 0, "last_epoch": None}

        # Wrap broadcast so we can track run metrics for history
        def tracking_callback(data: dict):
            global _run_status, _latest_start, _latest_epoch, _latest_checkpoint
            if data["type"] == "training_start":
                _latest_start = data
                _run_status = "training"
            elif data["type"] == "stopped":
                _run_status = "stopped"
            if data.get("checkpoint"):
                _latest_checkpoint = data["checkpoint"]
            if data.get("type") == "epoch":
                _latest_epoch = data
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
                checkpoint_dir=str(BASE_DIR / "checkpoints"),
                dataset=config.dataset, seed=config.seed, device=config.device,
                train_samples=config.train_samples, val_samples=config.val_samples,
            )
            trainer.train(callback=tracking_callback)
        except Exception as e:
            logger.error(f"Training crashed: {e}", exc_info=True)
            _run_status = "error"
            _latest_error = str(e)
            _broadcast({"type": "error", "message": str(e)})
        finally:
            if _run_status not in ("stopped", "error"):
                _run_status = "done"
            # Only save history if we got at least one epoch
            if run_info["last_epoch"] is not None:
                try:
                    _save_history(run_info, config)
                except Exception as e:
                    logger.warning(f"Could not save history: {e}")
            _training_active = False
            _broadcast({"type": "done", "status": _run_status})
            logger.info("Training thread finished")

    _training_thread = threading.Thread(target=run, name="training", daemon=False)
    _training_thread.start()
    logger.info(f"Training started — model={config.model} epochs={config.epochs}")

    return {"status": "started"}


@app.post("/api/stop")
async def stop_training():
    if not _training_active:
        raise HTTPException(409, "No training is currently running")
    global _run_status
    _run_status = "stopping"
    _stop_event.set()
    logger.info("Stop signal sent")
    return {"status": "stopping"}


@app.get("/api/status")
async def get_status():
    return {"is_running": _training_active, "status": _run_status,
            "training_start": _latest_start, "last_epoch": _latest_epoch,
            "checkpoint": _latest_checkpoint, "error": _latest_error,
            "workers": 1, "history_replay": False}


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
    queue = asyncio.Queue(maxsize=256)
    _client_queues.append(queue)
    async def send():
        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=10)
            except asyncio.TimeoutError:
                data = {"type": "heartbeat"}
            if data["type"] == "overflow":
                await websocket.close(code=1013, reason="Slow subscriber; reconnect for status")
                return
            await websocket.send_json(data)
    async def receive():
        while True:
            await websocket.receive_text()
    tasks = [asyncio.create_task(send()), asyncio.create_task(receive())]
    try:
        await websocket.send_json({"type": "status", **await get_status()})
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        for task in tasks:
            task.cancel()
        if queue in _client_queues:
            _client_queues.remove(queue)
        await asyncio.gather(*tasks, return_exceptions=True)
