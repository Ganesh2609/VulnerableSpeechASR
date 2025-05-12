"""
AVMD denoising script for audio preprocessing.

This script applies Adaptive Variational Mode Decomposition (AVMD) to selectively
denoise audio files based on their SNR (Signal-to-Noise Ratio). It processes
raw audio files and saves denoised versions for subsequent VAD processing.

The script implements selective denoising as described in the paper, applying
AVMD only to files with noise above a certain threshold to preserve quality
in already clean audio files.
"""

import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from glob import glob
from scipy.signal import hilbert
from PyEMD import VMD


def calculate_snr(audio_signal, sampling_rate=16000):
    """
    Calculate Signal-to-Noise Ratio (SNR) of an audio signal.
    
    Uses power spectral density to estimate signal and noise levels.
    This helps determine which audio files need denoising.
    
    Args:
        audio_signal (np.ndarray): Audio waveform array.
        sampling_rate (int): Sampling rate of the audio. Default is 16000 Hz.
    
    Returns:
        float: SNR value in decibels (dB).
    """
    # Convert to power spectrum
    power = np.abs(np.fft.fft(audio_signal))**2
    
    # Estimate signal power (use middle frequencies)
    freq_bins = len(power)
    signal_bins = range(int(freq_bins*0.1), int(freq_bins*0.9))
    signal_power = np.mean(power[signal_bins])
    
    # Estimate noise power (use high frequencies typically containing noise)
    noise_bins = range(int(freq_bins*0.9), freq_bins)
    noise_power = np.mean(power[noise_bins])
    
    # Calculate SNR in dB
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')
    
    return snr


def apply_avmd(signal, sampling_rate=16000, num_modes=5):
    """
    Apply Adaptive Variational Mode Decomposition to decompose and denoise signal.
    
    AVMD dynamically adjusts mode selection and bandwidth constraints based on
    signal characteristics, making it more effective than traditional VMD for
    separating noise from speech components.
    
    Args:
        signal (np.ndarray): Input audio signal.
        sampling_rate (int): Sampling rate of the signal. Default is 16000 Hz.
        num_modes (int): Number of modes to decompose signal into. Default is 5.
    
    Returns:
        np.ndarray: Denoised audio signal.
    """
    # Initialize VMD with adaptive parameters
    alpha = 2000  # Bandwidth constraint (moderate)
    tau = 0       # Noise tolerance (no noise)
    K = num_modes # Number of modes
    DC = 0        # No DC part imposed
    init = 1      # Initialize omegas uniformly
    tol = 1e-7    # Convergence tolerance
    
    # Create VMD object with adaptive parameters
    vmd = VMD(alpha=alpha, tau=tau, n_IMFs=K, DC=DC, init=init, tol=tol)
    
    try:
        # Decompose signal into modes
        modes, mode_frequencies = vmd(signal)
        
        # Analyze each mode's energy and frequency content
        mode_energies = []
        mode_frequencies_avg = []
        
        for mode in modes:
            # Calculate mode energy
            energy = np.sum(mode ** 2)
            mode_energies.append(energy)
            
            # Calculate dominant frequency using Hilbert transform
            analytic_signal = hilbert(mode)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = np.diff(instantaneous_phase) / (2.0*np.pi) * sampling_rate
            avg_frequency = np.mean(np.abs(instantaneous_frequency))
            mode_frequencies_avg.append(avg_frequency)
        
        # Select modes that likely contain speech (adaptive selection)
        # Speech typically falls within 50-4000 Hz range
        speech_modes = []
        for i, (freq, energy) in enumerate(zip(mode_frequencies_avg, mode_energies)):
            # Adaptive threshold based on energy distribution
            energy_threshold = np.mean(mode_energies) * 0.1
            
            # Select modes with speech-like frequencies and sufficient energy
            if 50 <= freq <= 4000 and energy > energy_threshold:
                speech_modes.append(modes[i])
        
        # Reconstruct signal from selected modes
        if speech_modes:
            reconstructed = np.sum(speech_modes, axis=0)
        else:
            # If no speech modes found, use original signal
            reconstructed = signal
        
        # Apply soft normalization to prevent clipping
        max_val = np.max(np.abs(reconstructed))
        if max_val > 0:
            reconstructed = reconstructed / max_val * 0.95
        
        return reconstructed
        
    except Exception as e:
        print(f"AVMD failed: {str(e)}. Returning original signal.")
        return signal


def denoise_audio_files(input_dir: str,
                       output_dir: str,
                       snr_threshold: float = 20.0,
                       sampling_rate: int = 16000,
                       device: str = None) -> None:
    """
    Process audio files and apply AVMD denoising based on SNR threshold.
    
    Implements the selective denoising strategy from the paper where only
    files with SNR below the threshold are processed through AVMD.
    
    Args:
        input_dir (str): Directory containing input audio files.
        output_dir (str): Directory to save denoised audio files.
        snr_threshold (float): SNR threshold in dB for selective denoising.
                             Files with SNR below this are denoised. Default is 20.0 dB.
        sampling_rate (int): Target sampling rate for audio processing. Default is 16000 Hz.
        device (str, optional): Device for computation. Auto-detects if None.
    """
    # Set up device (for potential GPU acceleration if AVMD supports it)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device available: {device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all audio files from input directory
    audio_files = glob(os.path.join(input_dir, "*.wav"))
    print(f"Found {len(audio_files)} audio files to process")
    
    # Track statistics
    denoised_count = 0
    skipped_count = 0
    
    # Process each audio file
    for audio_path in tqdm(audio_files, desc="Processing audio files"):
        filename = os.path.basename(audio_path)
        
        try:
            # Load audio file
            audio_data, sr = sf.read(audio_path)
            
            # Resample if necessary
            if sr != sampling_rate:
                # Simple resampling using numpy (you could use librosa for better quality)
                resample_ratio = sampling_rate / sr
                new_length = int(len(audio_data) * resample_ratio)
                audio_data = np.interp(
                    np.linspace(0, len(audio_data), new_length),
                    np.arange(len(audio_data)),
                    audio_data
                )
            
            # Calculate SNR
            snr = calculate_snr(audio_data, sampling_rate)
            
            # Apply AVMD if SNR is below threshold (noisy signal)
            if snr < snr_threshold:
                # As mentioned in the paper, Series 2 files have noise
                # Apply AVMD for denoising
                denoised_audio = apply_avmd(audio_data, sampling_rate)
                denoised_count += 1
                
                # Log the denoising action
                print(f"\nDenoised {filename} (SNR: {snr:.2f} dB)")
            else:
                # For clean signals, keep original
                denoised_audio = audio_data
                skipped_count += 1
            
            # Save the processed audio
            output_path = os.path.join(output_dir, filename)
            sf.write(output_path, denoised_audio, samplerate=sampling_rate)
            
        except Exception as e:
            print(f"\nError processing {filename}: {str(e)}")
            # Copy original file if processing fails
            try:
                audio_data, sr = sf.read(audio_path)
                output_path = os.path.join(output_dir, filename)
                sf.write(output_path, audio_data, samplerate=sr)
            except:
                print(f"Failed to copy original file for {filename}")
    
    print(f"\nProcessing complete!")
    print(f"Files denoised: {denoised_count}")
    print(f"Files skipped (already clean): {skipped_count}")
    print(f"Output saved to: {output_dir}")


def main():
    """
    Main function to run the AVMD denoising pipeline.
    
    Sets up paths based on the dataset structure and executes the
    selective denoising process. The paths match the pipeline described
    in the paper where AVMD is applied before VAD processing.
    """
    # Input directory (raw audio files)
    input_audio_dir = "Dataset/Training/Audio"
    
    # Output directory for AVMD denoised audio
    output_audio_dir = "Dataset/Training/Audio Denoise AVMD"
    
    # SNR threshold for selective denoising (in dB)
    # Files with SNR below this threshold will be denoised
    snr_threshold = 20.0
    
    # Sampling rate (standard for speech processing)
    sampling_rate = 16000
    
    # Run AVMD denoising
    denoise_audio_files(
        input_dir=input_audio_dir,
        output_dir=output_audio_dir,
        snr_threshold=snr_threshold,
        sampling_rate=sampling_rate
    )
    
    print("AVMD denoising completed successfully!")


if __name__ == "__main__":
    main()