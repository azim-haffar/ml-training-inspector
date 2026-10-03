# Engineering verification record — 4 October 2026

## Source and preservation

The initial local checkout was `e16d31f`; public main was inspected at `1afbdff`. The verified changes were rebased onto public main before publication, retaining upstream frontend dependency upgrades, localhost bindings, configuration validation and generated-file cleanup. Upstream API tests were adapted to the explicit terminal-status envelope; the original configuration and heuristic tests remain.

Existing checkpoints remain on disk and outside source control. No portfolio files were edited. Changes use the configured repository Git identity.

## Changes and decisions

- Added a seeded, download-free CPU dataset/demo. Two CPU threads keep small runs fast without using GPU or paid compute.
- Corrected cross-entropy aggregation to weight examples rather than batches. Batch charts now use sampled individual losses; LR reflects the rate actually used during the epoch.
- Kept process-local one-run architecture, enforced through documented one-worker deployment and HTTP 409 conflicts. No shared-state infrastructure was added.
- Added bounded per-client queues, slow-client close/reconnect behavior, receiver-driven disconnect detection, task cancellation, status snapshots and shutdown joining.
- Preserved stopped/error terminal outcomes. The final envelope can carry `status: stopped` or `status: error`; only `status: done` means successful completion.
- Added atomic unique checkpoint filenames and completed-epoch metadata. Explicitly declined to claim resume: scheduler/RNG/sampler state is missing. Weights and optimizer state load successfully.
- Improved loading/error messages, reconnect notices, dataset selection, phone layouts, focus outlines, theme-button labeling, progress semantics, full class labels and an accessible epoch table.
- Added CPU/lifecycle/metric tests, frontend mapping tests and a CI definition. The definition has not yet run remotely.

## Observed verification

Windows, Python 3.12.14, torch 2.5.1+cpu, torchvision 0.20.1+cpu, FastAPI 0.115.6, Node 24.19.0, Vite 8.3.2 after upstream integration.

`python scripts/demo.py`: seed 42, SimpleCNN, 2 epochs, batch size 32, 160 train / 80 validation synthetic examples. Training-loop elapsed time: 0.689 seconds, excluding interpreter/import startup. Epoch 1: train loss 1.17077, validation loss 2.18514, train accuracy 63.75%, validation accuracy 20%. Epoch 2: train loss 0.06278, validation loss 2.07824, train accuracy 100%, validation accuracy 10%. These are pipeline observations, not quality claims. Short-run BatchNorm evaluation differs considerably from training behavior.

15 backend tests passed (integrated run: 4.60 seconds): uneven final batch weighting, per-class count math, exact repeatability on this environment, model/optimizer checkpoint loading with reproduced validation metrics and atomic filenames, partial stop checkpoints, two-client delivery, idle disconnect cleanup, start conflict, shutdown, error outcome, bounded slow-client queue, cancellation during evaluation, reconnect state, configuration and heuristic thresholds.

2 frontend tests passed. Production build passed; Vite reports an approximately 554 KB uncompressed main bundle, above its 500 KB advisory threshold. One third-party Starlette/AnyIO deprecation warning remains.

Browser verification: completed real CPU runs, rendered charts and epoch table. Table values exactly matched the demo metrics. At 390-pixel viewport width, document width was 375 pixels; at 1280 pixels it was 1265. No page overflow was observed. All ten class labels are present. Screenshot: `demo-output/dashboard.jpg`.

Initial sandbox runs failed on temporary-directory permissions or stalled on restricted event-loop/network access. Completed verification used an isolated project environment and approved local execution; stalled processes were cleaned up.

## Remaining limits

- Resume absent; full CIFAR-10 training, ResNet9 training, GPU, container startup and cross-platform repeatability not verified.
- CIFAR-10 evaluation uses its official test split each epoch. No unbiased final quality benchmark exists.
- Reconnect returns status and latest epoch metadata, not missed chart history; charts disclose that gap. Exports retain only the latest 120 sampled batch points.
- Cancellation waits for in-flight batch operations and dataset download. Shutdown may wait on those operations.
- No authentication or per-user run ownership. Loopback demo only, one worker.
- Gradient/anomaly thresholds have no detection-quality evaluation; near-zero BatchNorm-adjacent biases can be benign.
- Remote CI results must be checked after publication; local success does not establish container or cross-platform behavior.
