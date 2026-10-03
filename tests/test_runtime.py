import asyncio
import json
import threading
import time
from pathlib import Path
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from fastapi.testclient import TestClient
import main
from trainer import Trainer, CIFAR10_CLASSES


def test_metrics_are_sample_weighted():
    logits = torch.tensor([[3.,0.], [0.,3.], [0.,3.]])
    labels = torch.tensor([0,1,0])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)
    trainer = Trainer()
    loss, accuracy, classes = trainer._evaluate(nn.Identity(), loader, nn.CrossEntropyLoss())
    assert loss == pytest.approx(nn.CrossEntropyLoss()(logits, labels).item())
    assert accuracy == pytest.approx(200/3)
    assert classes[CIFAR10_CLASSES[0]] == 50
    assert classes[CIFAR10_CLASSES[1]] == 100
    assert classes[CIFAR10_CLASSES[2]] is None


def test_real_training_repeatability_checkpoint_and_lr(tmp_path):
    runs = []
    for _ in range(2):
        events = []
        trainer = Trainer(epochs=2, batch_size=16, dataset='synthetic', train_samples=35,
                          val_samples=20, checkpoint_dir=str(tmp_path))
        trainer.train(events.append)
        epochs = [e for e in events if e['type'] == 'epoch']
        batches = [e for e in events if e['type'] == 'batch']
        assert len(batches) == 6
        for epoch in epochs:
            points = [e for e in batches if e['epoch'] == epoch['epoch']]
            expected = sum(e['loss'] * size for e, size in zip(points, [16,16,3])) / 35
            assert epoch['train_loss'] == pytest.approx(expected, abs=1e-5)
        assert [e['lr'] for e in epochs] == [0.001, 0.0005]
        checkpoint = torch.load(events[-1]['checkpoint'], weights_only=True)
        model = trainer._build_model()
        model.load_state_dict(checkpoint['model_state_dict'])
        restored_optimizer = torch.optim.Adam(model.parameters())
        restored_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        _, validation = trainer._load_data()
        loss, accuracy, classes = trainer._evaluate(model, validation, nn.CrossEntropyLoss())
        assert loss == pytest.approx(epochs[-1]['val_loss'], abs=1e-5)
        assert accuracy == pytest.approx(epochs[-1]['val_acc'], abs=.005)
        assert classes == epochs[-1]['class_accuracies']
        assert checkpoint['completed_epochs'] == 2
        assert checkpoint['optimizer_state_dict']['state']
        assert checkpoint['resumable'] is False
        runs.append(epochs)
    assert runs[0] == runs[1]
    assert len(list(tmp_path.glob('*.pth'))) == 2
    assert not list(tmp_path.glob('*.tmp'))


def test_stop_during_training_saves_partial_checkpoint(tmp_path):
    stop = threading.Event()
    events = []
    def callback(data):
        events.append(data)
        if data['type'] == 'batch': stop.set()
    Trainer(epochs=3, batch_size=16, dataset='synthetic', stop_event=stop,
            checkpoint_dir=str(tmp_path)).train(callback)
    assert events[-1]['type'] == 'stopped'
    checkpoint = torch.load(events[-1]['checkpoint'], weights_only=True)
    assert checkpoint['completed_epochs'] == 0
    assert not [e for e in events if e['type'] == 'epoch']


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setattr(main, 'HISTORY_PATH', str(tmp_path / 'history.json'))
    monkeypatch.setattr(main, 'BASE_DIR', tmp_path)
    main._run_status = 'idle'
    main._latest_start = main._latest_epoch = main._latest_checkpoint = main._latest_error = None
    with TestClient(main.app) as client:
        yield client
    assert not main._client_queues
    assert not main._training_active
    if main._training_thread: assert not main._training_thread.is_alive()


def test_two_clients_real_stream_and_disconnect(api):
    with api.websocket_connect('/ws') as a, api.websocket_connect('/ws') as b:
        assert a.receive_json()['type'] == b.receive_json()['type'] == 'status'
        assert api.post('/api/start', json={'dataset':'synthetic', 'epochs':1, 'batch_size':32}).status_code == 200
        first, second = [], []
        while True:
            x, y = a.receive_json(), b.receive_json()
            first.append(x); second.append(y)
            if x['type'] == 'done': break
        assert first == second
        assert [e for e in first if e['type'] == 'epoch']
        assert first[-1]['status'] == 'done'
    for _ in range(100):
        if not main._client_queues: break
        time.sleep(.01)
    assert not main._client_queues
    history = api.get('/api/history').json()['history']
    assert history[-1]['completed_epochs'] == 1
    assert history[-1]['status'] == 'done'


def test_conflict_stop_and_shutdown(api, monkeypatch):
    class WaitingTrainer:
        def __init__(self, stop_event, **kwargs): self.stop = stop_event
        def train(self, callback):
            self.stop.wait(5)
            callback({'type':'stopped'})
    monkeypatch.setattr(main, 'Trainer', WaitingTrainer)
    assert api.post('/api/start', json={}).status_code == 200
    assert api.post('/api/start', json={}).status_code == 409
    assert api.post('/api/stop').status_code == 200
    main._training_thread.join(2)
    assert api.get('/api/status').json()['status'] == 'stopped'
    assert api.post('/api/stop').status_code == 409
    assert api.post('/api/start', json={}).status_code == 200
    # Fixture shutdown must signal and join the still-running thread.


def test_errors_and_invalid_config(api, monkeypatch):
    for config in [{'epochs':0}, {'model':'unknown'}, {'learning_rate':-1}, {'device':'cuda'}]:
        assert api.post('/api/start', json=config).status_code == 422
    class BrokenTrainer:
        def __init__(self, **kwargs): pass
        def train(self, callback): raise ValueError('deliberate test failure')
    monkeypatch.setattr(main, 'Trainer', BrokenTrainer)
    with api.websocket_connect('/ws') as ws:
        ws.receive_json()
        api.post('/api/start', json={})
        assert ws.receive_json()['type'] == 'error'
        assert ws.receive_json() == {'type':'done', 'status':'error'}
    assert api.get('/api/status').json()['status'] == 'error'


def test_slow_subscriber_is_bounded():
    async def check():
        main._event_loop = asyncio.get_running_loop()
        queue = asyncio.Queue(maxsize=2)
        main._client_queues.append(queue)
        for i in range(3): main._broadcast({'type':'batch', 'batch':i})
        await asyncio.sleep(0)
        assert queue.qsize() == 1
        assert queue.get_nowait()['type'] == 'overflow'
        assert queue not in main._client_queues
        main._event_loop = None
    asyncio.run(check())


def test_cancellation_during_evaluation(tmp_path):
    trainer = Trainer(dataset='synthetic', checkpoint_dir=str(tmp_path))
    trainer.stop_event.set()
    loader = DataLoader(TensorDataset(torch.randn(2, 2), torch.tensor([0,1])), batch_size=1)
    with pytest.raises(InterruptedError):
        trainer._evaluate(nn.Identity(), loader, nn.CrossEntropyLoss())


def test_reconnect_receives_current_status(api):
    api.post('/api/start', json={'dataset':'synthetic', 'epochs':1})
    main._training_thread.join(5)
    with api.websocket_connect('/ws') as ws:
        snapshot = ws.receive_json()
        assert snapshot['status'] == 'done'
        assert snapshot['training_start']['run_id']
        assert snapshot['last_epoch']['epoch'] == 1
        assert snapshot['checkpoint']
        assert snapshot['history_replay'] is False
