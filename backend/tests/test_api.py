"""Exercise API orchestration with a fake trainer; no Torch or dataset needed."""
import asyncio
import importlib
import sys
import types
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

trainer_stub = types.ModuleType("trainer")
trainer_stub.Trainer = object
with patch.dict(sys.modules, {"trainer": trainer_stub}):
    api = importlib.import_module("main")


class ImmediateThread:
    def __init__(self, target, **kwargs):
        self.target = target

    def start(self):
        self.target()


class FakeTrainer:
    mode = "complete"

    def __init__(self, **kwargs):
        self.stop = kwargs["stop_event"]

    def train(self, callback):
        if self.mode == "fail":
            raise RuntimeError("Synthetic training failure")
        if self.mode == "stop":
            self.stop.set()
            callback({"type": "stopped"})


class APITests(unittest.TestCase):
    def test_invalid_request_rejected_before_trainer_runs(self):
        with TestClient(api.app) as client, patch.object(api, "Trainer") as trainer:
            response = client.post("/api/start", json={"epochs": 0})
            self.assertEqual(response.status_code, 422)
            trainer.assert_not_called()

    def test_completion_event_only_for_successful_run(self):
        for mode, expected in [("complete", ["done"]), ("stop", ["stopped"]), ("fail", ["error"])]:
            with self.subTest(mode=mode):
                FakeTrainer.mode = mode
                api._training_active = False
                events = []
                with patch.object(api, "Trainer", FakeTrainer), patch.object(api.threading, "Thread", ImmediateThread), patch.object(api, "_broadcast", events.append):
                    asyncio.run(api.start_training(api.TrainingConfig()))
                self.assertEqual([event["type"] for event in events], expected)
                self.assertFalse(api._training_active)


if __name__ == "__main__":
    unittest.main()
