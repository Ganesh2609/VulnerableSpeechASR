"""
Dataset module for loading and preprocessing speech data.

This module provides functionality for loading audio files with their corresponding
transcripts, preprocessing them for training Whisper models, and creating
efficient data loaders. It handles character encoding detection, audio resampling,
and batch collation for training.
"""

import os
import torch 
import librosa
import chardet
from functools import partial
import torch.nn.functional as F
from transformers import WhisperTokenizer, WhisperProcessor
from torch.utils.data import Dataset, DataLoader, random_split


class VulnerableSpeechData(Dataset):
    """
    Dataset class for loading audio files with their transcriptions.
    
    This dataset handles paired audio and transcript data, performing
    necessary preprocessing including audio resampling and text encoding
    detection. It's designed for speech recognition tasks where audio
    files are stored separately from their corresponding transcripts.
    
    Attributes:
        audios (list[str]): List of paths to audio files.
        transcripts (list[str]): List of paths to transcript files.
        processor (WhisperProcessor): Processor for audio feature extraction.
        sampling_rate (int): Target sampling rate for audio files.
    
    Args:
        root (str): Root directory containing 'Audio' and 'Transcripts' subdirectories.
        processor (WhisperProcessor): Whisper processor for audio preprocessing.
        sampling_rate (int): Target sampling rate for resampling audio. Default is 16000 Hz.
    
    Raises:
        AssertionError: If the number of audio files doesn't match transcript files.
    """

    def __init__(self, root:str, processor:WhisperProcessor, sampling_rate=16000):
        """
        Initialize the dataset with audio and transcript file paths.
        
        Expects the root directory to contain two subdirectories:
        - 'Audio': Contains audio files
        - 'Transcripts': Contains corresponding transcript files
        
        Files are matched by their position in the sorted file lists.
        
        Args:
            root (str): Root directory path.
            processor (WhisperProcessor): Processor for audio feature extraction.
            sampling_rate (int): Target sampling rate for audio resampling.
        """
        self.audios = [os.path.join(root, 'Audio', i) for i in os.listdir(root+'/Audio')]
        self.transcripts = [os.path.join(root, 'Transcripts', i) for i in os.listdir(root+'/Transcripts')]
        self.processor = processor
        self.sampling_rate = sampling_rate
        assert len(self.audios) == len(self.transcripts), "Mismatch between audio and transcript files."
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.audios)
    
    def __getitem__(self, idx):
        """
        Load and preprocess a single sample.
        
        Loads the audio file, resamples if necessary, and processes it through
        the Whisper processor. Also loads the corresponding transcript with
        automatic character encoding detection.
        
        Args:
            idx (int or torch.Tensor): Index of the sample to load.
        
        Returns:
            dict: Dictionary containing:
                - 'Audio': Processed audio features as tensor
                - 'Transcript': Raw transcript text string
        """
        if torch.is_tensor(idx):
            idx = idx.item()
        audio_path = self.audios[idx]
        transcript_path = self.transcripts[idx]

        # Load audio and resample if necessary
        waveform, sr = librosa.load(audio_path, sr=None)
        if sr != self.sampling_rate:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sampling_rate)
        waveform = self.processor(waveform, sampling_rate=self.sampling_rate, return_tensors='pt')

        # Load transcript with encoding detection
        with open(transcript_path, 'rb') as f:
            transcript = f.read(100)
        detected_encoding = chardet.detect(transcript)['encoding']
        with open(transcript_path, 'r', encoding=detected_encoding, errors='ignore') as f:
            transcript = f.read()

        return {
            'Audio': waveform['input_features'],
            'Transcript': transcript,
        }


def preprocess(text):
    """
    Preprocess text by normalizing whitespace.
    
    Removes newlines and reduces multiple spaces to single spaces.
    This ensures consistent text formatting for tokenization.
    
    Args:
        text (str): Input text to preprocess.
    
    Returns:
        str: Preprocessed text with normalized whitespace.
    """
    text = text.replace('\n', ' ')
    text = text.replace('  ', ' ')
    return text


def collate_fn(batch, tokenizer):
    """
    Collate function for creating batches from individual samples.
    
    Combines individual samples into a batch, handling both audio
    and text components. Audio tensors are concatenated while texts
    are tokenized and padded to create uniform batch tensors.
    
    Args:
        batch (list[dict]): List of individual samples from the dataset.
        tokenizer (WhisperTokenizer): Tokenizer for processing transcripts.
    
    Returns:
        dict: Batched data containing:
            - 'Audio': Concatenated audio features tensor
            - 'Text': List of preprocessed transcript strings
            - 'Transcript': Tokenized and padded transcript tensor
            - 'Transcript mask': Attention mask for padded transcripts
    """
    audio = torch.cat([item['Audio'] for item in batch], dim=0)
    text = [preprocess(item['Transcript']) for item in batch]
    transcription = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    return {
        'Audio': audio,
        'Text': text,
        'Transcript': transcription['input_ids'],
        'Transcript mask': transcription['attention_mask']
    }
    
    
def get_data_loaders(root:str, processor:WhisperProcessor, tokenizer=WhisperTokenizer, 
                     batch_size:int=4, train_size=0.8, num_workers:int=12, 
                     prefetch_factor:int=2):
    """
    Create train and test data loaders from the dataset.
    
    Splits the dataset into training and testing sets, then creates
    optimized DataLoaders with features like multi-worker loading,
    memory pinning, and prefetching for efficient GPU training.
    
    Args:
        root (str): Root directory containing the dataset.
        processor (WhisperProcessor): Processor for audio preprocessing.
        tokenizer: Tokenizer for text processing. Defaults to WhisperTokenizer class.
        batch_size (int): Number of samples per batch. Default is 4.
        train_size (float): Proportion of data for training. Default is 0.8.
        num_workers (int): Number of parallel workers for data loading. Default is 12.
        prefetch_factor (int): Number of batches to prefetch per worker. Default is 2.
    
    Returns:
        tuple[DataLoader, DataLoader]: Training and testing data loaders.
    
    Notes:
        - Uses random_split for train/test separation
        - Enables persistent workers for efficiency
        - Pins memory for faster GPU transfer
        - Uses partial function application for collate_fn
    """
    # Create the full dataset
    data = VulnerableSpeechData(root=root, processor=processor)

    # Split into train and test sets
    train_size = int(train_size * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])

    # Create optimized training data loader
    train_loader = DataLoader(
        dataset=train_data, 
        batch_size=batch_size, 
        shuffle=True,                    # Shuffle for better training
        num_workers=num_workers,         # Multi-process data loading
        pin_memory=True,                 # Pin memory for faster GPU transfer
        persistent_workers=True,         # Keep workers alive between epochs
        prefetch_factor=prefetch_factor, # Prefetch batches for efficiency
        collate_fn = partial(collate_fn, tokenizer=tokenizer)
    )

    # Create optimized testing data loader (no shuffling)
    test_loader = DataLoader(
        dataset=test_data, 
        batch_size=batch_size, 
        shuffle=False,                   # No shuffling for consistent evaluation
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=prefetch_factor,
        collate_fn = partial(collate_fn, tokenizer=tokenizer)
    )

    # Alternative simple loaders without optimization (commented out)
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn = partial(collate_fn, tokenizer=tokenizer))
    # test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, collate_fn = partial(collate_fn, tokenizer=tokenizer))

    return train_loader, test_loader