import whisper
import pyaudio
import wave
import numpy as np
import queue
import threading
import time
import sys

# Load Whisper model
model = whisper.load_model("base")

def record_audio(audio_queue, stop_event):
    """Records audio and stores it in a queue for processing."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording started...")
    while not stop_event.is_set():
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_queue.put(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("Recording stopped.")

def transcribe_audio(audio_queue, stop_event):
    """Fetches audio from queue and transcribes it using Whisper."""
    while not stop_event.is_set() or not audio_queue.empty():
        frames = []
        while not audio_queue.empty():
            frames.append(audio_queue.get())
        
        if frames:
            audio_data = b"".join(frames)
            np_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
            transcription = model.transcribe(np_audio, fp16=False)
            print("Subtitle:", transcription['text'])
            time.sleep(1)

# Setup threading for real-time speech-to-text
audio_queue = queue.Queue()
stop_event = threading.Event()

record_thread = threading.Thread(target=record_audio, args=(audio_queue, stop_event))
transcribe_thread = threading.Thread(target=transcribe_audio, args=(audio_queue, stop_event))

try:
    record_thread.start()
    transcribe_thread.start()
    while True:
        time.sleep(1)  # Keep the main thread alive
except KeyboardInterrupt:
    stop_event.set()
    record_thread.join()
    transcribe_thread.join()
    print("Exiting...")
    
