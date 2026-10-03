import unittest
from pydantic import ValidationError
from config import TrainingConfig
from anomaly import check_anomalies


class TrainingTests(unittest.TestCase):
    def test_supported_models_and_defaults(self):
        self.assertEqual(TrainingConfig().model, "simple_cnn")
        self.assertEqual(TrainingConfig(model="resnet9").epochs, 10)

    def test_invalid_run_configuration_is_rejected(self):
        for invalid in ({"epochs": 0}, {"epochs": 101}, {"batch_size": 0},
                        {"batch_size": 1025}, {"learning_rate": -1},
                        {"learning_rate": float("nan")}, {"learning_rate": float("inf")},
                        {"model": "unknown"}):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                TrainingConfig(**invalid)

    def test_gradient_thresholds_without_epoch_history(self):
        alerts = check_anomalies([], {"tiny": 1e-8, "huge": 101, "normal": 1})
        self.assertEqual({a["type"] for a in alerts}, {"vanishing_gradient", "exploding_gradient"})
        self.assertEqual(check_anomalies([], {"lower": 1e-7, "upper": 100, "zero": 0}), [])

    def test_overfitting_and_plateau_windows(self):
        epoch = {"train_loss": 1.0, "val_loss": 1.6, "val_acc": 50.0}
        self.assertEqual(check_anomalies([epoch], {}), [])
        alerts = {a["type"] for a in check_anomalies([epoch] * 5, {})}
        self.assertEqual(alerts, {"overfitting", "loss_plateau", "accuracy_plateau"})


if __name__ == "__main__":
    unittest.main()
