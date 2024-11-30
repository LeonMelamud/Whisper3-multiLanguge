import sys
import os
from typing import Optional, Tuple
from PyQt6.QtWidgets import (QApplication, QMainWindow, QPushButton, QTextEdit, 
                            QVBoxLayout, QHBoxLayout, QWidget, QFileDialog, QProgressBar, 
                            QLabel, QFrame, QComboBox, QMessageBox)
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QCloseEvent
from .transcribe import transcribe_audio, LANGUAGE_NAMES
import logging

logger = logging.getLogger(__name__)

class TranscriptionThread(QThread):
    progress = pyqtSignal(str)
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str)

    def __init__(self, audio_path: str, language: str):
        super().__init__()
        self.audio_path = audio_path
        self.language = language
        self._is_running = True

    def stop(self) -> None:
        """Safely stop the transcription thread."""
        self._is_running = False
        self.terminate()
        self.wait()

    def run(self) -> None:
        try:
            # Create a custom progress handler
            class ProgressHandler:
                def __init__(self, thread: 'TranscriptionThread'):
                    self.thread = thread

                def print_progress(self, stage: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
                    if not self.thread._is_running:
                        raise InterruptedError("Transcription stopped by user")
                    
                    # Format the progress message
                    if total:
                        message = f"{stage}: {current}/{total}"
                    else:
                        message = stage
                    self.thread.progress.emit(message)

            # Create progress handler
            progress_handler = ProgressHandler(self)

            # Run transcription
            result, output_file = transcribe_audio(self.audio_path, language=self.language, progress_handler=progress_handler)
            
            if not self._is_running:
                raise InterruptedError("Transcription stopped by user")
            
            if result and output_file:
                # Read the transcription file
                with open(output_file, 'r', encoding='utf-8') as f:
                    transcription = f.read()
                self.finished.emit(transcription, output_file)
            else:
                self.error.emit("Transcription failed")

        except InterruptedError as e:
            logger.warning("Transcription stopped by user")
            self.error.emit("Transcription stopped by user")
        except Exception as e:
            logger.error(f"Error during transcription: {str(e)}")
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Whisper Audio Transcription")
        self.setMinimumSize(800, 600)
        self.selected_file: Optional[str] = None
        self.transcription_thread: Optional[TranscriptionThread] = None
        
        self._setup_ui()
        
    def _setup_ui(self) -> None:
        """Setup the user interface components."""
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # File selection button and path display
        file_layout = QHBoxLayout()
        self.select_button = QPushButton("Select Audio File")
        self.select_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        file_layout.addWidget(self.select_button)
        file_layout.addWidget(self.file_label, stretch=1)
        layout.addLayout(file_layout)
        
        # Language selection
        lang_layout = QHBoxLayout()
        lang_label = QLabel("Language:")
        self.language_combo = QComboBox()
        
        # Add languages from LANGUAGE_NAMES
        for code, name in sorted(LANGUAGE_NAMES.items(), key=lambda x: x[1]):
            self.language_combo.addItem(name)
        
        # Set English as default
        english_index = self.language_combo.findText("English")
        if english_index >= 0:
            self.language_combo.setCurrentIndex(english_index)
        
        lang_layout.addWidget(lang_label)
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        layout.addLayout(lang_layout)

        # Control buttons layout
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Transcription")
        self.start_button.clicked.connect(self.start_transcription)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_transcription)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Output frame
        self.setup_output_frame(layout)

    def setup_output_frame(self, parent_layout: QVBoxLayout) -> None:
        """Setup the output frame with text area and copy button."""
        self.output_frame = QFrame()
        self.output_frame.setFrameStyle(QFrame.Shape.Panel | QFrame.Shadow.Sunken)
        parent_layout.addWidget(self.output_frame)
        
        output_layout = QVBoxLayout()
        self.output_frame.setLayout(output_layout)
        
        self.text_area = QTextEdit()
        self.text_area.setReadOnly(True)
        output_layout.addWidget(self.text_area)
        
        copy_layout = QHBoxLayout()
        self.copy_button = QPushButton("Copy Output")
        self.copy_button.clicked.connect(self.copy_output)
        self.copy_button.setEnabled(False)
        copy_layout.addStretch()
        copy_layout.addWidget(self.copy_button)
        output_layout.addLayout(copy_layout)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Handle window close event."""
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.stop()
            self.transcription_thread.wait()
        event.accept()

    def select_file(self) -> None:
        """Open file dialog to select audio file."""
        try:
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Select Audio File",
                "",
                "Audio Files (*.mp3 *.wav *.m4a *.ogg);;All Files (*.*)"
            )
            
            if file_name:
                if not os.path.exists(file_name):
                    raise FileNotFoundError(f"Selected file does not exist: {file_name}")
                
                self.selected_file = file_name
                self.file_label.setText(os.path.basename(file_name))
                self.start_button.setEnabled(True)
                self.text_area.clear()
                self.text_area.append(f"Selected file: {file_name}")
                
        except Exception as e:
            logger.error(f"Error selecting file: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error selecting file: {str(e)}")

    def start_transcription(self) -> None:
        """Start the transcription process."""
        if not self.selected_file:
            QMessageBox.warning(self, "Warning", "Please select an audio file first")
            return

        try:
            # Validate file exists and is readable
            if not os.path.exists(self.selected_file):
                raise FileNotFoundError(f"Selected file no longer exists: {self.selected_file}")
            
            if not os.access(self.selected_file, os.R_OK):
                raise PermissionError(f"Cannot read selected file: {self.selected_file}")
            
            # Check file size (limit to 2GB)
            file_size = os.path.getsize(self.selected_file)
            if file_size > 2 * 1024 * 1024 * 1024:  # 2GB
                raise ValueError("File is too large. Maximum file size is 2GB")

            # Reset UI
            self.text_area.clear()
            self.progress_bar.setValue(0)
            self.select_button.setEnabled(False)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.language_combo.setEnabled(False)
            self.copy_button.setEnabled(False)

            # Get selected language
            language = self.language_combo.currentText()
            # Find language code from name
            language_code = next((code for code, name in LANGUAGE_NAMES.items() if name == language), 'en')

            # Create and start transcription thread
            self.transcription_thread = TranscriptionThread(self.selected_file, language_code)
            self.transcription_thread.progress.connect(self.update_progress)
            self.transcription_thread.finished.connect(self.transcription_complete)
            self.transcription_thread.error.connect(self.transcription_error)
            self.transcription_thread.start()

            logger.info(f"Started transcription for file: {self.selected_file}")

        except Exception as e:
            logger.error(f"Error starting transcription: {str(e)}")
            self.transcription_error(str(e))

    def stop_transcription(self) -> None:
        """Stop the transcription process."""
        if self.transcription_thread and self.transcription_thread.isRunning():
            self.transcription_thread.stop()
            self.reset_ui()

    def update_progress(self, message: str) -> None:
        """Update progress bar and status message."""
        self.text_area.append(message)

    def transcription_complete(self, transcription: str, output_file: str) -> None:
        """Handle completed transcription."""
        self.text_area.append(f"\nTranscription completed! Saved to: {output_file}\n\nTranscription:\n{transcription}")
        self.copy_button.setEnabled(True)
        self.reset_ui()

    def transcription_error(self, error_message: str) -> None:
        """Handle transcription error."""
        QMessageBox.critical(self, "Error", f"Transcription failed: {error_message}")
        self.reset_ui()

    def reset_ui(self) -> None:
        """Reset UI elements to their default state."""
        self.select_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.language_combo.setEnabled(True)
        self.progress_bar.setValue(0)

    def copy_output(self) -> None:
        """Copy transcription output to clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.text_area.toPlainText())
        QMessageBox.information(self, "Success", "Output copied to clipboard!")

def main():
    try:
        print("Starting application...")
        app = QApplication(sys.argv)
        print("Created QApplication")
        window = MainWindow()
        print("Created MainWindow")
        window.show()
        print("Showing window")
        sys.exit(app.exec())
    except Exception as e:
        print(f"Error starting application: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
