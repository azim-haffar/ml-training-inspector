"""
Simple anomaly detection — just threshold checks for now.
Nothing ML-based, just the obvious stuff that shows up in debugging sessions.
"""


def check_anomalies(epoch_history: list, grad_norms: dict) -> list:
    anomalies = []

    # --- Gradient checks (current batch) ---
    for layer, norm in grad_norms.items():
        if norm < 1e-7 and norm > 0:
            anomalies.append({
                "type": "vanishing_gradient",
                "severity": "warning",
                "message": f"Near-zero gradient in '{layer}' (norm={norm:.2e})",
            })
        elif norm > 100:
            anomalies.append({
                "type": "exploding_gradient",
                "severity": "error",
                "message": f"Very large gradient in '{layer}' (norm={norm:.2f})",
            })

    if len(epoch_history) < 2:
        return anomalies

    latest = epoch_history[-1]

    # --- High loss (possible exploding gradients or bad LR) ---
    if latest["train_loss"] > 10.0:
        anomalies.append({
            "type": "high_loss",
            "severity": "error",
            "message": f"Training loss is unusually high ({latest['train_loss']:.2f}) — check learning rate",
        })

    # --- Overfitting: val loss diverging from train loss ---
    if "val_loss" in latest and "train_loss" in latest:
        gap = latest["val_loss"] - latest["train_loss"]
        if gap > 0.4 and len(epoch_history) >= 3:
            anomalies.append({
                "type": "overfitting",
                "severity": "warning",
                "message": f"Train/val loss gap is {gap:.3f} — model may be overfitting",
            })

    # --- Loss plateau: hasn't moved in 5 epochs ---
    if len(epoch_history) >= 5:
        recent = [e["train_loss"] for e in epoch_history[-5:]]
        delta = max(recent) - min(recent)
        if delta < 0.005:
            anomalies.append({
                "type": "loss_plateau",
                "severity": "info",
                "message": f"Loss moved less than {delta:.5f} over the last 5 epochs",
            })

    # --- Accuracy not improving after several epochs ---
    if len(epoch_history) >= 4:
        recent_acc = [e["val_acc"] for e in epoch_history[-4:]]
        if max(recent_acc) - min(recent_acc) < 0.5:
            anomalies.append({
                "type": "accuracy_plateau",
                "severity": "info",
                "message": f"Val accuracy stuck around {latest['val_acc']:.1f}% for 4 epochs",
            })

    return anomalies
