from typing import Literal
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=64, ge=1, le=1024)
    learning_rate: float = Field(default=0.001, gt=0, le=1, allow_inf_nan=False)
    model: Literal["simple_cnn", "resnet9"] = "simple_cnn"
    dataset: Literal["cifar10", "synthetic"] = "cifar10"
    seed: int = Field(default=42, ge=0, le=2147483647)
    device: Literal["cpu"] = "cpu"
    train_samples: int = Field(default=160, ge=10, le=1000)
    val_samples: int = Field(default=80, ge=10, le=1000)
