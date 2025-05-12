"""
Beam search decoding script for Whisper model transcription.

This script applies beam search decoding to generate transcriptions from a trained
Whisper model. It processes audio files from the VAD output directory and saves
transcriptions with matching filenames in the predictions directory.

The script specifically implements beam search decoding as described in the paper
to improve transcription accuracy during inference.
"""

import os
import torch
import librosa
from glob import glob
from tqdm import tqdm
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration


def load_audio(audio_path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """
    Load and preprocess audio file for Whisper model input.
    
    Loads an audio file and resamples it to the target sampling rate if necessary.
    This ensures the audio is in the correct format for the Whisper processor.
    
    Args:
        audio_path (str): Path to the audio file.
        sampling_rate (int): Target sampling rate for the audio. Default is 16000 Hz.
    
    Returns:
        torch.Tensor: Loaded and resampled audio waveform.
    """
    waveform, sr = librosa.load(audio_path, sr=None)
    if sr != sampling_rate:
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=sampling_rate)
    return waveform


def generate_transcriptions(model_path: str,
                           input_dir: str,
                           output_dir: str,
                           device: str = None,
                           beam_size: int = 5) -> None:
    """
    Generate transcriptions for all audio files using beam search decoding.
    
    This function implements the inference pipeline described in the paper:
    1. Loads the trained Whisper model from checkpoint
    2. Processes audio files through the model
    3. Uses beam search decoding for improved accuracy
    4. Saves transcriptions with corresponding filenames
    
    Args:
        model_path (str): Path to the saved model checkpoint.
        input_dir (str): Directory containing input audio files (VAD processed).
        output_dir (str): Directory to save output transcriptions.
        device (str, optional): Device for computation. Auto-detects if None.
        beam_size (int): Number of beams for beam search decoding. Default is 5.
    """
    # Set up device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Whisper components
    print("Loading Whisper processor and tokenizer...")
    processor = WhisperProcessor.from_pretrained("vasista22/whisper-tamil-medium", task="transcribe")
    tokenizer = WhisperTokenizer.from_pretrained("vasista22/whisper-tamil-medium", task="transcribe")
    
    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = WhisperForConditionalGeneration.from_pretrained("vasista22/whisper-tamil-medium")
    
    # Load checkpoint if it's a .pth file
    if model_path.endswith('.pth'):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    model = model.to(device)
    model.eval()
    
    # Get all audio files from input directory
    audio_files = glob(os.path.join(input_dir, "*.wav"))
    print(f"Found {len(audio_files)} audio files to process")
    
    # Process each audio file
    with torch.no_grad():
        for audio_path in tqdm(audio_files, desc="Generating transcriptions"):
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(audio_path))[0]
            
            # Load and preprocess audio
            waveform = load_audio(audio_path)
            
            # Process audio through Whisper processor
            input_features = processor(waveform, 
                                      sampling_rate=16000, 
                                      return_tensors="pt")
            input_features = input_features.input_features.to(device)
            
            # Generate transcription using beam search
            # As mentioned in the paper, beam search is used during inference
            generated_ids = model.generate(
                input_features=input_features,
                num_beams=beam_size,          # Beam search parameter
                max_length=150,               # Maximum sequence length
                early_stopping=True,          # Stop when all beams finish
                temperature=1.0,              # No temperature scaling
                do_sample=False,              # Deterministic beam search
                repetition_penalty=1.0,       # No repetition penalty
                length_penalty=1.0            # No length penalty
            )
            
            # Decode the generated tokens to text
            transcription = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # Remove leading/trailing whitespace
            transcription = transcription.strip()
            
            # Save transcription to file
            output_path = os.path.join(output_dir, f"{filename}.txt")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(transcription)
    
    print(f"Transcriptions saved to {output_dir}")


def main():
    """
    Main function to run the inference pipeline.
    
    Sets up paths based on the training configuration and executes the
    transcription generation process. The paths match the structure used
    in the training script.
    """
    # Configuration based on the training setup
    model_checkpoint_path = "Codebase/Whisper/Train Data/Checkpoints/whisper medium 1/best_model.pth"
    
    # Input directory (output of VAD processing)
    input_audio_dir = "Dataset/Training/Audio Denoise VAD"
    
    # Output directory for transcriptions
    output_transcript_dir = "Dataset/Training/Transcript predictions"
    
    # Set beam search parameters based on paper
    beam_size = 5  # Can be adjusted based on requirements
    
    # Run inference
    generate_transcriptions(
        model_path=model_checkpoint_path,
        input_dir=input_audio_dir,
        output_dir=output_transcript_dir,
        beam_size=beam_size
    )
    
    print("Inference completed successfully!")


if __name__ == "__main__":
    main()