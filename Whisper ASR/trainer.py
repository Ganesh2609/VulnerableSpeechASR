"""
Modular trainer module for training neural networks with configurable components.

This module provides a flexible training framework that can work with any PyTorch model,
supporting features like checkpointing, logging, visualization, and metric tracking.
It's specifically designed for sequence-to-sequence models like Whisper.
"""

import os
import torch 
from torch import nn
import matplotlib.pyplot as plt
from typing import Optional
from logger import TrainingLogger
from transformers import WhisperTokenizer
from tqdm import tqdm
from torchmetrics.text import WordErrorRate, CharErrorRate


class ModularTrainer:
    """
    A modular trainer for PyTorch models with comprehensive training utilities.
    
    This trainer provides a complete training pipeline including:
    - Training and validation loops
    - Checkpointing and resumption
    - Loss and metric tracking (including WER and CER for speech recognition)
    - Real-time visualization of training progress
    - Comprehensive logging
    
    The trainer is designed to be modular, allowing users to plug in their own
    models, optimizers, schedulers, and loss functions. It includes specific
    support for speech recognition tasks with Word Error Rate (WER) and
    Character Error Rate (CER) metrics.
    
    Attributes:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        test_loader (Optional[DataLoader]): DataLoader for validation/test data.
        loss_fn (torch.nn.Module): Loss function for training.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (Optional[_LRScheduler]): Learning rate scheduler.
        tokenizer: Tokenizer for decoding predictions.
        logger (TrainingLogger): Logger instance for training progress.
        device (torch.device): Device for computation (CPU/GPU).
        num_epochs (int): Total number of training epochs.
        checkpoint_path (str): Directory for saving checkpoints.
        graph_path (str): Path for saving training plots.
        verbose (bool): Whether to print verbose training information.
        current_epoch (int): Current training epoch.
        current_step (int): Current global training step.
        best_metric (float): Best validation metric achieved.
        history (dict): Training and validation metrics history.
        step_history (dict): Step-wise metrics history for plotting.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: torch.utils.data.DataLoader, 
                 test_loader: Optional[torch.utils.data.DataLoader] = None,
                 loss_fn: Optional[torch.nn.Module] = None,
                 optimizer: Optional[torch.optim.Optimizer] = None,
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 tokenizer = WhisperTokenizer,
                 log_path: Optional[str] = './logs/training.log',
                 num_epochs: Optional[int] = 16,
                 checkpoint_path: Optional[str] = './checkpoints',
                 graph_path: Optional[str] = './graphs/model_loss.png',
                 verbose: Optional[bool] = True,
                 device: Optional[torch.device] = None) -> None:
        """
        Initialize the ModularTrainer with all necessary components.
        
        Sets up directories, initializes training components, and prepares
        the model for training. If no device is specified, automatically
        selects GPU if available, otherwise CPU.
        
        Args:
            model (torch.nn.Module): The neural network model to train.
            train_loader (DataLoader): DataLoader providing training batches.
            test_loader (Optional[DataLoader]): DataLoader for validation/testing.
                If None, no validation will be performed.
            loss_fn (Optional[torch.nn.Module]): Loss function to use.
                Defaults to CrossEntropyLoss if not provided.
            optimizer (Optional[Optimizer]): Optimizer for parameter updates.
                Defaults to Adam with lr=1e-3 if not provided.
            scheduler (Optional[_LRScheduler]): Learning rate scheduler.
                If None, no learning rate scheduling is performed.
            tokenizer: Tokenizer for converting predictions to text.
                Defaults to WhisperTokenizer.
            log_path (Optional[str]): Path for saving log files.
                Default is './logs/training.log'.
            num_epochs (Optional[int]): Number of epochs to train.
                Default is 16.
            checkpoint_path (Optional[str]): Directory for saving model checkpoints.
                Default is './checkpoints'.
            graph_path (Optional[str]): Path for saving training plots.
                Default is './graphs/model_loss.png'.
            verbose (Optional[bool]): Enable verbose logging.
                Default is True.
            device (Optional[torch.device]): Device for computation.
                If None, automatically selects GPU if available.
        """
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(checkpoint_path, exist_ok=True)
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        
        self.logger = TrainingLogger(log_path=log_path)

        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.optimizer = optimizer or torch.optim.Adam(params=self.model.parameters(), lr=1e-3)
        self.scheduler = scheduler

        self.num_epochs = num_epochs
        self.checkpoint_path = checkpoint_path
        self.graph_path = graph_path
        self.verbose = verbose
        self.loss_update_step = 15

        self.current_epoch = 1
        self.current_step = 1
        self.best_metric = float('inf')

        self.tokenizer = tokenizer
        self.wer_metric = WordErrorRate()
        self.cer_metric = CharErrorRate()

        self.history = {
            'Training Loss': [],
            'Training Word Error Rate' : [],
            'Training Character Error Rate' : [],
            'Testing Loss': [],
            'Testing Word Error Rate' : [],
            'Testing Character Error Rate' : []
        }

        self.step_history = {
            'Training Loss': [],
            'Training Word Error Rate' : [],
            'Training Character Error Rate' : [],
            'Testing Loss': [],
            'Testing Word Error Rate' : [],
            'Testing Character Error Rate' : []
        }

    def update_plot(self) -> None:
        """
        Update training progress plots with current metrics.
        
        Creates a 2x3 grid of plots showing:
        - Training Loss, WER, and CER (top row)
        - Testing Loss, WER, and CER (bottom row)
        
        Plots are saved to the path specified by self.graph_path.
        Each plot shows the progression of metrics over training steps,
        sampled at intervals defined by self.loss_update_step.
        """
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        axs[0, 0].plot(self.step_history['Training Loss'], color='blue', label='Training Loss')
        axs[0, 0].set_title('Training Loss')
        axs[0, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 0].set_ylabel('Loss')
        axs[0, 0].legend()

        axs[0, 1].plot(self.step_history['Training Word Error Rate'], color='purple', label='Training WER')
        axs[0, 1].set_title('Training Word Error Rate')
        axs[0, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 1].set_ylabel('Error Rate')
        axs[0, 1].legend()

        axs[0, 2].plot(self.step_history['Training Character Error Rate'], color='brown', label='Training CER')
        axs[0, 2].set_title('Training Character Error Rate')
        axs[0, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[0, 2].set_ylabel('Error Rate')
        axs[0, 2].legend()

        axs[1, 0].plot(self.step_history['Testing Loss'], color='green', label='Testing Loss')
        axs[1, 0].set_title('Testing Loss')
        axs[1, 0].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].legend()

        axs[1, 1].plot(self.step_history['Testing Word Error Rate'], color='red', label='Testing WER')
        axs[1, 1].set_title('Testing Word Error Rate')
        axs[1, 1].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 1].set_ylabel('Error Rate')
        axs[1, 1].legend()

        axs[1, 2].plot(self.step_history['Testing Character Error Rate'], color='orange', label='Testing CER')
        axs[1, 2].set_title('Testing Character Error Rate')
        axs[1, 2].set_xlabel(f'Steps [every {self.loss_update_step}]')
        axs[1, 2].set_ylabel('Error Rate')
        axs[1, 2].legend()

        plt.tight_layout()
        plt.savefig(self.graph_path)
        plt.close(fig)

        return

    def train_epoch(self) -> None:
        """
        Execute one training epoch.
        
        Performs forward pass, loss computation, backward pass, and parameter
        updates for all batches in the training loader. Tracks and logs:
        - Batch-wise and epoch-wise loss
        - Word Error Rate (WER)
        - Character Error Rate (CER)
        
        Updates step history periodically for real-time plotting and logs
        epoch summary statistics.
        """
        self.model.train()
        total_loss = 0.0
        total_wer = 0.0
        total_cer = 0.0

        with tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Training)') as t:
            
            for i, batch in t:
                
                input_features = batch['Audio'].to(self.device)
                transcription_text = batch['Text']
                decoder_input_ids = batch['Transcript'].to(self.device)
                decoder_attention_mask = batch['Transcript mask'].to(self.device)

                logits = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask).logits
                loss = self.loss_fn(logits[:, 1:-1, :].reshape(-1,  51865), decoder_input_ids[:, 2:].reshape(-1))
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                predicted_ids = torch.argmax(logits, dim=-1)
                prediction = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                wer = self.wer_metric(prediction, transcription_text)
                cer = self.cer_metric(prediction, transcription_text)

                total_loss += loss.item()
                total_wer += wer.item() * 100
                total_cer += cer.item() * 100

                self.current_step += 1

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Train Loss' : total_loss/(i+1),
                    'Word Error Rate' : total_wer/(i+1),
                    'Character Error Rate' : total_cer/(i+1)
                })

                if i % self.loss_update_step == 0 and i != 0:
                    self.step_history['Training Loss'].append(total_loss / (i+1))
                    self.step_history['Training Word Error Rate'].append(total_wer / (i+1))
                    self.step_history['Training Character Error Rate'].append(total_cer / (i+1))
                    self.update_plot()

        train_loss = total_loss / len(self.train_loader)
        train_wer = total_wer / len(self.train_loader)
        train_cer = total_cer / len(self.train_loader)
        self.history['Training Loss'].append(train_loss)
        self.history['Training Word Error Rate'].append(train_wer)
        self.history['Training Character Error Rate'].append(train_cer)
        
        self.logger.info(f"Training loss for epoch {self.current_epoch}: {train_loss}")
        self.logger.info(f"Training word error rate for epoch {self.current_epoch}: {train_wer}\n")
        self.logger.info(f"Training character error rate for epoch {self.current_epoch}: {train_cer}\n")

        return

    def test_epoch(self) -> None:
        """
        Execute one validation/test epoch.
        
        Evaluates the model on the test set without gradient computation.
        Tracks the same metrics as training but without parameter updates.
        Updates the learning rate scheduler if provided and saves the best
        model based on test loss.
        
        Also periodically writes sample predictions to 'output.txt' for
        qualitative evaluation.
        """
        self.model.eval()
        total_loss = 0.0
        total_wer = 0.0
        total_cer = 0.0

        with tqdm(enumerate(self.test_loader), total=len(self.test_loader), desc=f'Epoch [{self.current_epoch}/{self.num_epochs}] (Testing)') as t:
            
            for i, batch in t:
                
                input_features = batch['Audio'].to(self.device)
                transcription_text = batch['Text']
                decoder_input_ids = batch['Transcript'].to(self.device)
                decoder_attention_mask = batch['Transcript mask'].to(self.device)
                
                with torch.no_grad():
                    logits = self.model(input_features=input_features, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask).logits
                    loss = self.loss_fn(logits[:, 1:-1, :].reshape(-1,  51865), decoder_input_ids[:, 2:].reshape(-1))
                    predicted_ids = torch.argmax(logits, dim=-1)
                
                prediction = self.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)
                wer = self.wer_metric(prediction, transcription_text).item() * 100
                cer = self.cer_metric(prediction, transcription_text).item() * 100

                total_loss += loss.item()
                total_wer += wer
                total_cer += cer

                t.set_postfix({
                    'Batch Loss' : loss.item(),
                    'Test Loss' : total_loss/(i+1),
                    'Word Error Rate' : total_wer/(i+1),
                    'Character Error Rate' : total_cer/(i+1)
                })

                if i % self.loss_update_step == 0 and i != 0:

                    self.step_history['Testing Loss'].append(total_loss / (i+1))
                    self.step_history['Testing Word Error Rate'].append(total_wer / (i+1))
                    self.step_history['Testing Character Error Rate'].append(total_cer / (i+1))
                    self.update_plot()

                    with open('output.txt', 'a', encoding="utf-8") as f:
                        f.write(transcription_text[0] + '\n' + prediction[0] + '\n' + self.tokenizer.decode(predicted_ids[0], skip_special_tokens=False) + '\n\n\n')

        test_loss = total_loss / len(self.test_loader)
        test_wer = total_wer / len(self.test_loader)
        test_cer = total_cer / len(self.test_loader)
        self.history['Testing Loss'].append(test_loss)
        self.history['Testing Word Error Rate'].append(test_wer)
        self.history['Testing Character Error Rate'].append(test_cer)

        if self.scheduler:
            self.scheduler.step(test_loss)

        if test_loss < self.best_metric:
            self.best_metric = test_loss
            self.save_checkpoint(is_best=True)

        self.logger.info(f"Testing loss for epoch {self.current_epoch}: {test_loss}")
        self.logger.info(f"Testing word error rate for epoch {self.current_epoch}: {test_wer}\n")
        self.logger.info(f"Testing character error rate for epoch {self.current_epoch}: {test_cer}\n")
        if self.scheduler:
            self.logger.info(f"Current Learning rate: {self.scheduler.get_last_lr()}")

        return

    def train(self, resume_from: Optional[str]=None) -> None:
        """
        Main training loop.
        
        Executes the complete training process for the specified number of epochs.
        Supports resuming from a checkpoint, performing both training and validation
        for each epoch, and saving checkpoints after each epoch.
        
        Args:
            resume_from (Optional[str]): Path to checkpoint file to resume from.
                If provided, training continues from the saved state.
                If None, training starts from scratch.
        """
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {self.current_epoch}")
            self.logger.log_training_resume(
                epoch=self.current_epoch, 
                global_step=self.current_step, 
                total_epochs=self.num_epochs
            )
        else:
            self.logger.info(f"Starting training for {self.num_epochs} epochs from scratch")
    
        print(f"Starting training from epoch {self.current_epoch} to {self.num_epochs}")
        
        for epoch in range(self.current_epoch, self.num_epochs + 1):

            self.current_epoch = epoch
            self.train_epoch()
            
            if self.test_loader:
                self.test_epoch()
    
            self.save_checkpoint()
        
        return

    def save_checkpoint(self, is_best:Optional[bool]=False):
        """
        Save current training state to a checkpoint file.
        
        Saves model parameters, optimizer state, scheduler state (if available),
        training history, and all relevant metadata needed to resume training.
        
        Args:
            is_best (Optional[bool]): If True, saves as 'best_model.pth'.
                Otherwise, saves with epoch number in filename.
                Default is False.
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'current_step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_history' : self.step_history,
            'history': self.history,
            'best_metric': self.best_metric
        }

        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()

        if is_best:
            path = os.path.join(self.checkpoint_path, 'best_model.pth')
        else:
            path = os.path.join(
                self.checkpoint_path, 
                f'model_epoch_{self.current_epoch}.pth'
            )

        torch.save(checkpoint, path)
        
        if self.verbose:
            save_type = "Best model" if is_best else "Checkpoint"
            self.logger.info(f"{save_type} saved to {path}")

    def load_checkpoint(self, checkpoint:Optional[str]=None, resume_from_best:Optional[bool]=False):
        """
        Load training state from a checkpoint file.
        
        Restores model parameters, optimizer state, scheduler state (if available),
        training history, and all metadata to resume training from a saved state.
        
        Args:
            checkpoint (Optional[str]): Path to checkpoint file to load.
                Required if resume_from_best is False.
            resume_from_best (Optional[bool]): If True, loads from 'best_model.pth'
                in the checkpoint directory. Default is False.
        """
        if resume_from_best:
            checkpoint_path = os.path.join(self.checkpoint_path, 'best_model.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        else:
            checkpoint = torch.load(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.current_epoch = checkpoint.get('epoch') + 1
        self.current_step = checkpoint.get('current_step')
        self.best_metric = checkpoint.get('best_metric')
        
        loaded_history = checkpoint.get('history')
        for key in self.history:
            self.history[key] = loaded_history.get(key, self.history[key])

        loaded_step_history = checkpoint.get('step_history')
        for key in self.step_history:
            self.step_history[key] = loaded_step_history.get(key, self.step_history[key])

        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return