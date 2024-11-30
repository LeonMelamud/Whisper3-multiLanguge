# 🎤 Whisper Audio Multi-language Transcription App

## 📚 Overview

A cross-platform audio transcription application using OpenAI's Whisper model (specifically `whisper-large-v3`) with a PyQt6 graphical interface. The model is used locally on your machine, ensuring privacy and control over your data.

## 🚀 Features

- High-quality audio transcription
- Cross-platform GUI
- Multi-language support
- Easy file selection
- Clipboard integration

## 🔧 Prerequisites

- Python 3.11+
- macOS, Windows, or Linux
- FFmpeg installed
- Hugging Face account (for downloading models)

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/whisper-transcribe.git
cd whisper-transcribe
```

2. Create a virtual environment:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set the Hugging Face token (replace with your actual token):
```bash
export HUGGINGFACE_TOKEN=your_huggingface_token_here
```

5. Run the application:
```bash
python -m src.whisper_transcribe.gui
```

## ▶️ Usage

Run the application:
```bash
python -m src.whisper_transcribe.gui
```

## ⚙️ Configuration

Supports:
- Language selection
- Transcription chunk size
- Model customization

## Technologies

- Python
- Whisper AI
- PyQt6
- Transformers

## License

MIT License

## Contact

Leon Melamud - leonmelamud@gmail.com
