from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
from dataclasses import field


@dataclass
class TrainerConfig:
    """
    Configuration for model training and evaluation.
    """
    # Fixed or set parameters
    eval_every_steps: int = 1024
    max_steps: int = 300 * 1024
    batch_size: int = 1
    normalize: bool = True
    pretrained_model_path: Optional[Path] = None
    random_init: bool = False
    freeze_encoder: bool = False  # If True, freeze Orthrus encoder and only train regression head
    Orthrus_MLM: bool = False
    
    # Model and optimization parameters
    accumulation_steps: int = 256
    learning_rate: float = 0.001
    max_norm: float = 3.0
    weight_decay: float = 0.0
    orthrus_weight_decay: float = 0.001
    lr_gamma: float = 0.99
    warmup_steps: int = 10000
    
    # Loss function parameters
    loss_type: str = "huber"  # Options: "mse", "huber"
    huber_delta: float = 0.5  # Delta parameter for Huber loss
    
    # DNNHead parameters
    dnn_hidden_dims: List[int] = field(default_factory=lambda: [256, 64])
    dnn_dropout: float = 0.0
    activation_function: str = "gelu"  # Options: "relu", "gelu"
    use_layer_norm: bool = True  # Whether to use layer normalization in DNNHead
    
    # Early stopping parameters
    early_stopping_patience: int = 50  # Number of evaluation steps without improvement before stopping
    


