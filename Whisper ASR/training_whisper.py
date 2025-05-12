"""
Training script for Whisper model fine-tuning.

This script sets up and executes the training pipeline for fine-tuning a Whisper
model on custom audio transcription data. It configures the model, data loaders,
optimizer, scheduler, and trainer components to perform supervised training.

The script specifically uses the Tamil Whisper model and trains it on custom
data for improved transcription performance.
"""

# import warnings
# warnings.filterwarnings("ignore")

import torch
from torch import nn
from trainer import ModularTrainer
from dataset import get_data_loaders
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration


def main():
    """
    Main training function for Whisper model fine-tuning.
    
    This function orchestrates the complete training pipeline:
    1. Sets up the computational device (GPU/CPU)
    2. Loads the Whisper processor, tokenizer, and pre-trained model
    3. Creates data loaders for training and testing
    4. Configures optimizer and learning rate scheduler
    5. Initializes the trainer with all components
    6. Executes the training process
    
    The function uses specific paths for data, logs, checkpoints, and graphs,
    and is configured for a Tamil Whisper medium model. Training parameters
    include AdamW optimizer with weight decay and ReduceLROnPlateau scheduler.
    
    Notes:
        - Automatically detects and uses GPU if available, falls back to CPU
        - Uses Tamil-specific Whisper processor for transcription task
        - Includes commented option to resume training from checkpoint
        - Batch size is set to 4 for memory efficiency
        - Learning rate starts at 1e-5 with automatic reduction on plateau
    """
    # Determine device for computation
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Load Whisper model components
    # Using Tamil-specific processor for transcription task
    processor = WhisperProcessor.from_pretrained("vasista22/whisper-tamil-medium", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained("vasista22/whisper-tamil-medium", task="transcribe")
    # Load pre-trained model checkpoint
    model = WhisperForConditionalGeneration.from_pretrained("JairamKanna/pretrainedwhisper-medium-native-v2")

    # Set data path
    root_path = "Dataset/Training"

    # Create data loaders with batch size of 4
    train_loader, test_loader = get_data_loaders(root=root_path, processor=processor, tokenizer=tokenizer, batch_size=4)

    # Configure training hyperparameters
    learning_rate = 1e-5
    weight_decay = 1e-2
    
    # Set up loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Configure optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Set up learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, 
        mode="min",          # Monitor minimum loss
        factor=0.5,          # Reduce LR by half
        patience=2,          # Wait 2 epochs before reducing
        min_lr=1e-7,         # Minimum learning rate
        verbose=False
    )

    # Initialize trainer with all components
    trainer = ModularTrainer(
        model=model,
        train_loader=train_loader,  
        test_loader=test_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        tokenizer=tokenizer,
        log_path='Codebase/Whisper/Train Data/Logs/whisper_medium_1.log',
        num_epochs=1,
        checkpoint_path='Codebase/Whisper/Train Data/Checkpoints/whisper medium 1',
        graph_path='Codebase/Whisper/Train Data/Graphs/whisper_medium_1.png',
        verbose=True,
        device=device
    )

    # Start training from scratch
    trainer.train()
    
    # Alternative: Resume training from a specific checkpoint
    # trainer.train(resume_from="E:/Amrita/Subjects/Sem 6/Speech Processing/Tutorials/6/Question 1/Train Data/Checkpoints/wav2vec2 train 2/model_epoch_7.pth")


if __name__ == '__main__':
    main()