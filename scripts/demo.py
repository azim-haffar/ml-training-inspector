"""Reproducible CPU smoke run. Run from repository root."""
import json
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
from trainer import Trainer

output = Path('demo-output')
output.mkdir(exist_ok=True)
events = []
started = time.perf_counter()
Trainer(epochs=2, batch_size=32, dataset='synthetic', seed=42,
        checkpoint_dir=str(output / 'checkpoints')).train(events.append)
result = {'elapsed_seconds': round(time.perf_counter() - started, 3), 'events': events}
(output / 'metrics.json').write_text(json.dumps(result, indent=2))
print(json.dumps({'elapsed_seconds': result['elapsed_seconds'],
                  'epochs': [e for e in events if e['type'] == 'epoch']}, indent=2))
