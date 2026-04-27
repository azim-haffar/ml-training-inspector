import os
import threading
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import logging

from anomaly import check_anomalies

logger = logging.getLogger(__name__)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# Simple 3-layer CNN — not trying to beat SOTA, just enough to have
# interesting training dynamics to observe
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ResNet-9: a fast 9-layer network that gets ~90%+ on CIFAR-10 with the
# right LR schedule. Uses two residual blocks to avoid vanishing gradients.
class ResNet9(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

        self.prep   = conv_block(3, 64)
        self.layer1 = nn.Sequential(conv_block(64, 128), nn.MaxPool2d(2))
        self.res1   = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.layer2 = nn.Sequential(conv_block(128, 256), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(conv_block(256, 512), nn.MaxPool2d(2))
        self.res3   = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = x + self.res1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x + self.res3(x)
        return self.classifier(x)


class Trainer:
    def __init__(self, epochs=10, batch_size=64, lr=0.001,
                 model_name: str = "simple_cnn",
                 stop_event: threading.Event = None,
                 checkpoint_dir: str = "./checkpoints"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epoch_history = []
        self.stop_event = stop_event or threading.Event()
        self.checkpoint_dir = checkpoint_dir

    def _build_model(self):
        if self.model_name == "resnet9":
            return ResNet9().to(self.device)
        return SimpleCNN().to(self.device)

    def _load_data(self):
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2023, 0.1994, 0.2010)

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        trainset = torchvision.datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_transform)
        valset   = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)

        # num_workers=0 avoids shared memory issues inside Docker containers
        train_loader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True,  num_workers=0)
        val_loader   = DataLoader(valset,   batch_size=self.batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader

    def _get_grad_norms(self, model):
        norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                norms[name] = round(param.grad.norm().item(), 8)
        return norms

    def _evaluate(self, model, loader, criterion):
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        class_correct = [0] * 10
        class_total   = [0] * 10

        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                for c in range(10):
                    mask = (labels == c)
                    class_correct[c] += (predicted[mask] == c).sum().item()
                    class_total[c]   += mask.sum().item()

        class_acc = {
            CIFAR10_CLASSES[i]: round(100.0 * class_correct[i] / class_total[i], 1)
            if class_total[i] > 0 else 0.0
            for i in range(10)
        }

        return total_loss / len(loader), 100.0 * correct / total, class_acc

    def _save_checkpoint(self, model, optimizer, epoch):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_name": self.model_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        logger.info(f"Checkpoint saved → {path}")
        return path

    def train(self, callback):
        logger.info(f"Starting training — model={self.model_name} device={self.device}")
        model     = self._build_model()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        # CosineAnnealingLR gives a smooth LR decay that's nice to visualise;
        # the LR drops from lr to ~0 over T_max epochs then can restart.
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        train_loader, val_loader = self._load_data()
        total_batches = len(train_loader)

        callback({
            "type": "training_start",
            "model": self.model_name,
            "total_epochs": self.epochs,
            "total_batches": total_batches,
            "device": str(self.device),
        })

        last_epoch = 0

        for epoch in range(1, self.epochs + 1):
            last_epoch = epoch
            model.train()
            running_loss, correct, total = 0.0, 0, 0

            for batch_idx, (inputs, labels) in enumerate(train_loader):
                if self.stop_event.is_set():
                    checkpoint_path = self._save_checkpoint(model, optimizer, epoch)
                    callback({"type": "stopped", "epoch": epoch, "checkpoint": checkpoint_path})
                    return

                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted  = outputs.max(1)
                total   += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if (batch_idx + 1) % 20 == 0:
                    callback({
                        "type": "batch",
                        "model": self.model_name,
                        "epoch": epoch,
                        "batch": batch_idx + 1,
                        "total_batches": total_batches,
                        "loss": round(running_loss / (batch_idx + 1), 5),
                        "accuracy": round(100.0 * correct / total, 2),
                        "grad_norms": self._get_grad_norms(model),
                    })

            val_loss, val_acc, class_acc = self._evaluate(model, val_loader, criterion)
            train_loss = running_loss / total_batches
            train_acc  = 100.0 * correct / total

            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]

            epoch_entry = {
                "epoch": epoch,
                "train_loss": round(train_loss, 5),
                "val_loss":   round(val_loss, 5),
                "train_acc":  round(train_acc, 2),
                "val_acc":    round(val_acc, 2),
            }
            self.epoch_history.append(epoch_entry)

            anomalies = check_anomalies(self.epoch_history, self._get_grad_norms(model))

            callback({
                "type": "epoch",
                "model": self.model_name,
                **epoch_entry,
                "class_accuracies": class_acc,
                "anomalies": anomalies,
                "lr": round(current_lr, 8),
            })

            logger.info(
                f"Epoch {epoch}/{self.epochs} — "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.1f}%"
            )

        checkpoint_path = self._save_checkpoint(model, optimizer, last_epoch)
        callback({"type": "checkpoint_saved", "checkpoint": checkpoint_path, "epoch": last_epoch})
