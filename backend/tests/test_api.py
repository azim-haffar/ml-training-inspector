"""Exercise API orchestration with a fake trainer; no Torch or dataset needed."""
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
import main as api


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

    def test_completion_event_preserves_outcome(self):
        for mode, expected in [("complete", "done"), ("stop", "stopped"), ("fail", "error")]:
            with self.subTest(mode=mode):
                FakeTrainer.mode = mode
                api._training_active = False
                with TestClient(api.app) as client, patch.object(api, "Trainer", FakeTrainer):
                    response = client.post("/api/start", json={})
                    self.assertEqual(response.status_code, 200)
                    api._training_thread.join(2)
                    self.assertEqual(client.get("/api/status").json()["status"], expected)
                    self.assertFalse(api._training_active)


if __name__ == "__main__":
    unittest.main()
