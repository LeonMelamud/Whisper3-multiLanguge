import os
from datetime import datetime
import argparse
from dotenv import load_dotenv
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
import ffmpeg
import sys
import time
from transformers import pipeline
import logging
import warnings
from typing import Optional, Tuple, List, Any, Dict
from dataclasses import dataclass

# Filter out specific warnings
warnings.filterwarnings("ignore", message="The input name `inputs` is deprecated")
warnings.filterwarnings("ignore", message="You have passed task=transcribe")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisper_transcribe.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@dataclass
class TranscriptionConfig:
    """Configuration for transcription settings."""
    CHUNK_LENGTH: int = 30
    OVERLAP_LENGTH: float = 1.5
    SAMPLE_RATE: int = 16000
    MODEL_NAME: str = "openai/whisper-large-v3"
    DEVICE: str = "mps" if torch.backends.mps.is_available() else "cpu"
    TASK: str = "transcribe"
    RETURN_TIMESTAMPS: bool = True

config = TranscriptionConfig()

# Language code to full name mapping
LANGUAGE_NAMES = {
    'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German', 'it': 'Italian',
    'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'ru': 'Russian',
    'ar': 'Arabic', 'hi': 'Hindi', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'he': 'Hebrew', 'th': 'Thai', 'hu': 'Hungarian', 'cs': 'Czech', 'da': 'Danish',
    'fi': 'Finnish', 'el': 'Greek', 'id': 'Indonesian', 'vi': 'Vietnamese'
}

class ProgressTracker:
    def __init__(self):
        self.start_time = time.time()
        
    def print_progress(self, stage: str, current: int, total: Optional[int] = None, newline: bool = False):
        """Print progress with percentage and time elapsed"""
        elapsed_time = time.time() - self.start_time
        if total:
            percentage = (current / total) * 100
            message = f"{stage}: {current}/{total} ({percentage:.1f}%) - Time elapsed: {elapsed_time:.1f}s"
        else:
            message = f"{stage} - Time elapsed: {elapsed_time:.1f}s"
            
        if newline:
            print(message)
        else:
            sys.stdout.write('\r' + message)
            sys.stdout.flush()

def get_audio_duration(audio_path: str) -> Optional[float]:
    """Get the duration of an audio file in seconds"""
    try:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except ffmpeg.Error as e:
        logger.error(f"Error getting audio duration: {str(e)}")
        return None

def load_audio(audio_path: str, progress: ProgressTracker) -> Optional[np.ndarray]:
    """Load and preprocess audio file"""
    try:
        logger.info("\nAnalyzing audio file...")
        duration = get_audio_duration(audio_path)
        if duration:
            logger.info(f"Audio duration: {duration:.1f} seconds")
        
        progress.print_progress("Reading audio file", 0, 100)  # Initialize with percentage
        
        # Read audio file using ffmpeg with progress updates
        process = (
            ffmpeg
            .input(audio_path)
            .output('pipe:', format='f32le', acodec='pcm_f32le', ac=1, ar=config.SAMPLE_RATE)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        
        # Read output in chunks with progress updates
        chunks = []
        bytes_read = 0
        chunk_size = 4096 * 64  # Increased chunk size for better performance
        last_progress = 0
        
        while True:
            chunk = process.stdout.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk)
            
            # Update progress less frequently (every 5%)
            current_progress = int((bytes_read / (bytes_read + chunk_size)) * 100)
            if current_progress - last_progress >= 5:
                progress.print_progress("\rReading audio", current_progress, 100)
                last_progress = current_progress
        
        process.wait()
        
        # Combine chunks and convert to numpy array
        progress.print_progress("Processing audio data", 95, 100)
        audio_data = b''.join(chunks)
        audio = np.frombuffer(audio_data, np.float32).flatten()
        
        progress.print_progress("Audio loading complete", 100, 100)
        return audio
        
    except ffmpeg.Error as e:
        logger.error(f'\nError loading audio: {str(e)}')
        return None

def format_timestamp(seconds: float) -> str:
    """
    Format time in seconds to HH:MM:SS
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def process_audio_chunks(audio: np.ndarray, progress_handler: Optional[ProgressTracker] = None) -> Optional[List[Tuple[np.ndarray, float, float]]]:
    """
    Process audio into chunks for transcription
    
    Args:
        audio: Audio data as numpy array
        progress_handler: Progress handler for updates
    
    Returns:
        list: List of tuples (chunk, start_time, end_time)
    """
    try:
        # Calculate chunk size and overlap in samples
        chunk_size = int(config.CHUNK_LENGTH * config.SAMPLE_RATE)
        overlap_size = int(config.OVERLAP_LENGTH * config.SAMPLE_RATE)
        
        # Calculate number of chunks
        total_samples = len(audio)
        effective_chunk = chunk_size - overlap_size
        n_chunks = max(1, int(np.ceil(total_samples / effective_chunk)))
        
        chunks = []
        for i in range(n_chunks):
            # Calculate chunk boundaries with overlap for processing
            start_idx = i * effective_chunk
            end_idx = min(start_idx + chunk_size, total_samples)
            
            # Get audio chunk
            chunk = audio[start_idx:end_idx]
            
            # Pad last chunk if needed
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
            
            # Calculate timestamps (without overlap)
            chunk_start = i * config.CHUNK_LENGTH
            chunk_end = min((i + 1) * config.CHUNK_LENGTH, total_samples / config.SAMPLE_RATE)
            
            chunks.append((chunk, chunk_start, chunk_end))
        
        if progress_handler:
            progress_handler.print_progress(f"Found {len(chunks)} chunks to process")
        
        return chunks
        
    except Exception as e:
        logger.error(f"\nError processing chunks: {str(e)}")
        return None

def transcribe_audio(audio_path: str, language: str, progress_handler: Optional[ProgressTracker] = None) -> Tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio file using Whisper
    
    Args:
        audio_path: Path to audio file
        language: Language code (e.g., 'en', 'es')
        progress_handler: Progress handler for updates
    
    Returns:
        tuple: (transcription text, output file path) or (None, None) if failed
    """
    try:
        if progress_handler:
            progress_handler.print_progress("Loading audio file", 0, 3)

        # Load and preprocess audio
        audio = load_audio(audio_path, progress_handler)
        if audio is None:
            return None, None

        if progress_handler:
            progress_handler.print_progress("Loading model", 1, 3)

        # Load model
        pipe = pipeline(
            "automatic-speech-recognition",
            model=config.MODEL_NAME,
            torch_dtype=torch.float16,
            device=config.DEVICE,
        )

        # Configure the model for transcription
        forced_decoder_ids = pipe.tokenizer.get_decoder_prompt_ids(
            task=config.TASK,
            language=language
        )
        
        # Set the generation config
        pipe.model.generation_config.forced_decoder_ids = forced_decoder_ids
        pipe.model.generation_config.task = config.TASK
        pipe.model.generation_config.return_timestamps = config.RETURN_TIMESTAMPS
        
        progress_handler.print_progress("Model loaded", 3, 3)
        
        # Process all audio chunks
        chunks = process_audio_chunks(audio, progress_handler)
        transcription = []
        
        total_chunks = len(chunks)
        for i, (chunk, start_time, end_time) in enumerate(chunks, 1):
            if progress_handler:
                progress_handler.print_progress(f"Transcribing chunk {i}/{total_chunks}")
            
            result = pipe(
                {"raw": chunk, "sampling_rate": config.SAMPLE_RATE},
                return_timestamps=config.RETURN_TIMESTAMPS,
                generate_kwargs={
                    "task": config.TASK,
                    "language": language,
                    "return_legacy_cache": True
                }
            )
            text = result["text"].strip()
            
            if text:
                timestamp = f"[{format_timestamp(start_time)} --> {format_timestamp(end_time)}]"
                transcription.append(f"{timestamp}\n{text}\n\n")  # Added extra newline for separation

        # Save transcription
        output_dir = os.path.join(os.path.expanduser("~"), "Desktop")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"audio_{timestamp}.txt")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(transcription))
        
        return "\n".join(transcription), output_file
        
    except Exception as e:
        logger.error(f"\nError during transcription: {str(e)}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio files using Whisper")
    parser.add_argument("audio_path", help="Path to the audio file to transcribe")
    parser.add_argument("--language", required=True, help="Language code for transcription (e.g., 'en', 'es')")
    
    args = parser.parse_args()
    transcription, output_file = transcribe_audio(args.audio_path, args.language)
    if transcription and output_file:
        logger.info(f"\nTranscription saved to: {output_file}")

if __name__ == "__main__":
    main()
