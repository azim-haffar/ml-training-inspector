from typing import Literal
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    epochs: int = Field(default=10, ge=1, le=100)
    batch_size: int = Field(default=64, ge=1, le=1024)
    learning_rate: float = Field(default=0.001, gt=0, le=1, allow_inf_nan=False)
    model: Literal["simple_cnn", "resnet9"] = "simple_cnn"
