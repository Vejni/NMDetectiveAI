from pathlib import Path
from typing import Tuple, Optional
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score
import numpy as np
import wandb
from tqdm import tqdm
from loguru import logger

from NMD.modeling.models.NMDetectiveAI import NMDetectiveAI
from NMD.modeling.TrainerConfig import TrainerConfig
from NMD.utils import load_model, relative_squared_error
from NMD.config import OUT_DIR

class Trainer:
    """Base class for training NMD prediction models."""
    
    def __init__(self, config: TrainerConfig, train_type: str, mode: str):
        self.config = config
        self.train_type = train_type  # e.g., "PTC", "DMS_SP", "DMS_LE", "DMS_PE"
        self.mode = mode  # e.g., "CV", "minigene_CV_normalize"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup output directory structure: OUT_DIR / train_type / wandb_id
        self.wandb_id = wandb.run.id if wandb.run else "no_wandb"
        self.output_dir = self._setup_output_directory()
        
        # Setup model components
        self.model, self.criterion, self.optimizer, self.scheduler = self._setup_model()
        
        # Load pretrained weights if specified
        if config.pretrained_model_path:
            load_model(self.model, config.pretrained_model_path, device=self.device)
            logger.info(f"Loaded pretrained weights from {config.pretrained_model_path}")
        
        # Save training parameters
        self._save_training_parameters()
    
    def _setup_model(self) -> Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler]:
        """Setup model, criterion, optimizer and scheduler."""
        # Create NMDetectiveAI model with all configuration
        if self.config.random_init:
            logger.warning("Using random initialization for Orthrus model.")
        
        model = NMDetectiveAI(
            hidden_dims=self.config.dnn_hidden_dims,
            dropout=self.config.dnn_dropout,
            random_init=self.config.random_init,
            use_mlm=self.config.Orthrus_MLM,
            activation_function=self.config.activation_function,
            use_layer_norm=self.config.use_layer_norm,
        ).to(self.device)
        
        logger.info("Initialized NMDetectiveAI model")
        
        # Freeze encoder if specified
        if self.config.freeze_encoder:
            for param in model.encoder.parameters():
                param.requires_grad = False
            logger.info("Frozen Orthrus encoder - only training regression head")
        
        # Setup optimizer with different weight decay for encoder vs head
        # Only include trainable parameters
        optimizer_params = []
        if not self.config.freeze_encoder:
            optimizer_params.append({
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "weight_decay": self.config.orthrus_weight_decay
            })
        optimizer_params.append({
            "params": model.head.parameters(),
            "weight_decay": self.config.weight_decay
        })
        
        optimizer = optim.AdamW(optimizer_params, lr=self.config.learning_rate)
        
        # Setup scheduler and criterion
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.config.lr_gamma)
        
        # Setup loss function based on config
        if self.config.loss_type == "huber":
            criterion = nn.HuberLoss(delta=self.config.huber_delta)
            logger.info(f"Using Huber loss (delta={self.config.huber_delta})")
        elif self.config.loss_type == "mse":
            criterion = nn.MSELoss()
            logger.info("Using MSE loss")
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}. Options: 'mse', 'huber'")
        
        return model, criterion, optimizer, scheduler
    
    def _setup_output_directory(self) -> Path:
        """Setup output directory with structure: OUT_DIR / train_type / wandb_id"""
       
        if "CV" in self.mode:
            # For cross-validation, include mode in path
            output_dir = OUT_DIR / self.train_type / self.mode / self.wandb_id
        else:
            # include in last dir the mode (e.g. minigene, normalize) if not empty
            if self.mode:
                output_dir = OUT_DIR / self.train_type / (self.mode + "_" + self.wandb_id)
            else:
                output_dir = OUT_DIR / self.train_type / self.wandb_id
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")
        return output_dir
    
    def _save_training_parameters(self):
        """Save training parameters to a text file."""
        params_file = self.output_dir / "training_parameters.txt"
        
        with open(params_file, 'w') as f:
            f.write("=== NMD Training Parameters ===\n\n")
            
            # Basic info
            f.write(f"Train Type: {self.train_type}\n")
            f.write(f"Wandb ID: {self.wandb_id}\n")
            f.write(f"Wandb Project: NMD\n")
            f.write(f"Device: {self.device}\n\n")
            
            # Configuration parameters
            f.write("=== Training Configuration ===\n")
            for key, value in self.config.__dict__.items():
                f.write(f"{key}: {value}\n")
            
            f.write("=== Model Information ===\n")
            f.write(f"Model: NMDetective-AI\n")
            f.write(f"Encoder: Orthrus (quietflamingo/orthrus-large-6-track)\n")
            f.write(f"Random Init: {self.config.random_init}\n")
            f.write(f"Freeze Encoder: {self.config.freeze_encoder}\n")
            f.write(f"Regression Head: DNN\n")
            f.write(f"DNN Input Dim: 512 (fixed)\n")
            f.write(f"DNN Hidden Dims: {self.config.dnn_hidden_dims}\n")
            f.write(f"DNN Dropout: {self.config.dnn_dropout}\n")
            f.write(f"DNN Output Dim: 1 (fixed)\n")
            f.write(f"Loss Function: MSE (fixed)\n")
            if self.config.pretrained_model_path:
                f.write(f"Pretrained Model: {self.config.pretrained_model_path}\n")
            else:
                f.write("Pretrained Model: None (fresh training)\n")
            
            f.write(f"\n=== Output Directory ===\n")
            f.write(f"Output Directory: {self.output_dir}\n")
        
        logger.info(f"Training parameters saved to {params_file}")
    
    def train_step_based(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None
    ) -> Path:
        """Execute step-based training with periodic validation and early stopping."""
        self.model.train()
        train_iterator = iter(train_loader)
        running_loss = 0.0
        epochs_completed = 0
        batches_per_epoch = len(train_loader)
        steps_in_current_epoch = 0
        best_val_loss = float("inf")
        best_val_corr = float("-inf")
        best_val_r2 = float("-inf")
        best_step = 0
        best_model_state = None
        early_stopping_counter = 0
        
        self.optimizer.zero_grad()
        
        for step in tqdm(range(self.config.max_steps), desc="Training Steps"):
            # Get next batch (reset iterator if needed)
            try:
                batch_sequences, batch_lengths, batch_labels = next(train_iterator)
                steps_in_current_epoch += 1
            except StopIteration:
                train_iterator = iter(train_loader)
                batch_sequences, batch_lengths, batch_labels = next(train_iterator)
                epochs_completed += 1
                steps_in_current_epoch = 1  # Reset to 1 for the new epoch
                logger.info(f"Completed epoch {epochs_completed} with {step + 1} steps")
            
            # Move to device
            batch_sequences, batch_lengths, batch_labels = [
                x.to(self.device) for x in (batch_sequences, batch_lengths, batch_labels)
            ]
            
            # Forward pass
            predictions = self.model(batch_sequences, batch_lengths).view(-1)
            batch_labels = batch_labels.view(-1)
            loss = self.criterion(predictions, batch_labels) / self.config.accumulation_steps
            loss.backward()
            running_loss += loss.item() * self.config.accumulation_steps
            
            # Gradient accumulation and optimization
            if (step + 1) % self.config.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Validation and checkpointing
            if (step + 1) % self.config.eval_every_steps == 0:
                if val_loader is not None:
                    val_metrics = self._validate(val_loader)
                    self.model.train()  # Return to training mode
                    
                    # Log metrics
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    # Calculate fractional epoch progress
                    fractional_epoch = epochs_completed + (steps_in_current_epoch / batches_per_epoch)
                    
                    logger.info(
                        f"Step {step + 1}: Val Loss: {val_metrics['loss']:.4f}, "
                        f"Val R2: {val_metrics['r2']:.4f}, Val Corr: {val_metrics['correlation']:.4f}"
                    )
                    
                    wandb.log({
                        "step": step + 1,
                        "epoch": fractional_epoch,
                        "train_loss": running_loss / self.config.eval_every_steps,
                        "learning_rate": current_lr,
                        "early_stopping_counter": early_stopping_counter,
                        **{f"val_{k}": v for k, v in val_metrics.items()}
                    })
                    
                    # Save best model and check early stopping
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        best_val_corr = val_metrics['correlation']
                        best_val_r2 = val_metrics['r2']
                        best_step = step + 1
                        best_model_state = deepcopy(self.model.state_dict())
                        early_stopping_counter = 0  # Reset counter on improvement
                        logger.info(f"New best model at step {step + 1} with val_loss: {best_val_loss:.4f}")
                    else:
                        early_stopping_counter += 1
                        logger.info(f"No improvement for {early_stopping_counter} evaluations")
                    
                    # Check early stopping
                    if early_stopping_counter >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {early_stopping_counter} evaluations without improvement")
                        break
                    
                    # Step scheduler after warmup
                    if step > (self.config.warmup_steps * self.config.eval_every_steps):
                        self.scheduler.step()
                
                running_loss = 0.0
        
        # Log training completion status
        if early_stopping_counter >= self.config.early_stopping_patience:
            logger.info(f"Training stopped early at step {step + 1} due to no improvement in validation loss")
        else:
            logger.info(f"Training completed all {self.config.max_steps} steps")
        
        # Save last step model state
        last_step_save_path = self._get_last_step_model_save_path()
        torch.save({"model_state_dict": self.model.state_dict()}, last_step_save_path)
        logger.info(f"Saved last step model to {last_step_save_path}")
        
        # Save best model as the final model
        save_path = self._get_model_save_path()
        final_state = best_model_state if best_model_state is not None else self.model.state_dict()
        torch.save({"model_state_dict": final_state}, save_path)
        logger.info(f"Saved best model to {save_path}")

        if wandb.run and val_loader is not None:
            wandb.log({
                "Best Val Loss": best_val_loss, 
                "Best Val Correlation": best_val_corr,
                "Best Val R2": best_val_r2,
                "Best Step": best_step,
                "Final Step": step + 1,
                "Early Stopped": early_stopping_counter >= self.config.early_stopping_patience,
                "Last Step Model Path": str(last_step_save_path),
                "Best Model Path": str(save_path)
            })

        # update self.model
        self.model.load_state_dict(final_state)

        return save_path
    
    def _validate(self, val_loader: DataLoader) -> dict:
        """Execute validation pass."""
        self.model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_sequences, batch_lengths, batch_labels in tqdm(val_loader, desc="Validation"):
                batch_sequences, batch_lengths, batch_labels = [
                    x.to(self.device) for x in (batch_sequences, batch_lengths, batch_labels)
                ]
                
                predictions = self.model(batch_sequences, batch_lengths).view(-1)
                batch_labels = batch_labels.view(-1)
                
                loss = self.criterion(predictions, batch_labels)
                val_loss += loss.item()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        avg_val_loss = val_loss / len(val_loader)
        val_r2 = r2_score(all_labels, all_preds)
        val_corr = np.corrcoef(all_labels, all_preds)[0, 1]
        val_rse = relative_squared_error(all_labels, all_preds)
        
        return {
            "loss": avg_val_loss,
            "r2": val_r2,
            "correlation": val_corr,
            "rse": val_rse
        }
    
    def get_predictions(self, data_loader: DataLoader) -> tuple[list, list]:
        """Get model predictions for a dataset."""
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for batch_sequences, batch_lengths, batch_labels in tqdm(data_loader, desc="Getting predictions"):
                batch_sequences, batch_lengths, batch_labels = [
                    x.to(self.device) for x in (batch_sequences, batch_lengths, batch_labels)
                ]
                
                predictions = self.model(batch_sequences, batch_lengths).view(-1)
                batch_labels = batch_labels.view(-1)
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        return all_preds, all_labels
    
    def _get_model_save_path(self) -> Path:
        """Generate standardized model save path with wandb ID."""
        return self.output_dir / f"{self.wandb_id}.pt"
    
    def _get_last_step_model_save_path(self) -> Path:
        """Generate standardized last step model save path with wandb ID."""
        return self.output_dir / f"{self.wandb_id}_last_step.pt"
    