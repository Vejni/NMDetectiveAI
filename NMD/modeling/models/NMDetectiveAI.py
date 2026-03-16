"""
NMDetective-AI: Deep learning model for NMD efficiency prediction.

This model combines the Orthrus genomic foundation model with a deep neural network
regression head for sequence-based NMD efficiency prediction.
"""

import torch
import torch.nn as nn
from typing import List, Optional
from transformers import AutoModel, AutoConfig


class DNNHead(nn.Module):
    """
    Deep Neural Network regression head for Orthrus encoder.
    
    Args:
        input_dim: Input feature dimension from encoder
        hidden_dims: List of hidden layer sizes
        dropout: Dropout probability
        output_dim: Output dimension (default 1 for regression)
        use_layer_norm: Whether to use layer normalization
    """

    def __init__(
        self, 
        input_dim: int = 512, 
        hidden_dims: List[int] = [256, 64], 
        dropout: float = 0.0, 
        output_dim: int = 1,
        use_layer_norm: bool = False,
        activation_function: str = "relu",
    ):
        super().__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_layer_norm and i == 0:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if activation_function == "relu":
                layers.append(nn.ReLU())
            elif activation_function == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unsupported activation function: {activation_function}")
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(dims[-1], output_dim))
        self.fc = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.fc(x)


class NMDetectiveAI(nn.Module):
    """
    NMDetective-AI: Deep learning model combining Orthrus encoder and regression head.
    
    This model uses a pretrained Orthrus genomic foundation model as an encoder
    and a deep neural network head for NMD efficiency prediction.
    
    Args:
        hidden_dims: List of hidden layer sizes for DNN head (default: [256, 128, 64])
        dropout: Dropout probability for DNN head (default: 0.2)
        random_init: Whether to use random initialization for encoder (default: False)
        use_mlm: Whether to use MLM pretrained Orthrus (default: False)
    """

    def __init__(
        self, 
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
        random_init: bool = False,
        use_mlm: bool = False,
        activation_function: str = "relu",
        use_layer_norm: bool = False,
    ):
        super().__init__()
        
        # Load Orthrus encoder
        if random_init:
            config_orthrus = AutoConfig.from_pretrained(
                "antichronology/orthrus-6-track", trust_remote_code=True
            )
            self.encoder = AutoModel.from_config(config_orthrus, trust_remote_code=True)
        else:
            if use_mlm:
                self.encoder = AutoModel.from_pretrained(
                    "antichronology/orthrus-6-track", trust_remote_code=True
                )
            else:
                self.encoder = AutoModel.from_pretrained(
                    "quietflamingo/orthrus-large-6-track", trust_remote_code=True
                )
        
        # Create DNN regression head (fixed: 512 input dim, 1 output, no layer norm)
        self.head = DNNHead(
            input_dim=512,
            hidden_dims=hidden_dims,
            dropout=dropout,
            output_dim=1,
            use_layer_norm=use_layer_norm,
            activation_function=activation_function,
        )

    def forward(self, x, lengths: Optional[torch.Tensor] = None, embed: bool = False):
        """
        Forward pass through the model.
        
        Args:
            x: Input sequences with shape (B, L, 6) where B=batch, L=length, 6=channels
            lengths: Sequence lengths for mean pooling
            embed: If True, return embeddings instead of predictions
            
        Returns:
            Predictions (or embeddings if embed=True)
        """
        # Get mean-pooled representation from encoder
        # Orthrus expects (B, L, 6) format with channel_last=True
        x = self.encoder.representation(x, lengths, channel_last=True)
        
        if embed:
            return x
        else:
            x = self.head(x)
            return x
