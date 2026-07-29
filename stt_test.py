import sys
import time
import numpy as np
import sounddevice as sd
from pywhispercpp.model import Model

# 1. Initialize the model on your M4 GPU
model = Model('small', n_threads=4, print_realtime=False, print_progress=False)

# Global variable to safely pass text out of the callback scope
latest_transcript = ""

# 2. Callback function fired when Whisper finishes a segment
def execution_callback(segment):
    global latest_transcript
    latest_transcript = segment.text.strip()

print("Listening... Speak into your microphone. (Press Ctrl+C to exit)\n")

# 3. Audio streaming configuration 
SAMPLE_RATE = 16000
BLOCK_SIZE = 48000  # 3-second blocks

def audio_stream_callback(indata, frames, time_info, status):
    global latest_transcript
    if status:
        print(status, file=sys.stderr)
    
    # Convert input data for whisper processing
    audio_data = indata[:, 0].astype(np.float32)
    
    # Reset transcript tracking for this block
    latest_transcript = ""
    
    # Track execution start time
    start_time = time.perf_counter()
    
    # Transcribe the buffer chunk
    model.transcribe(audio_data, new_segment_callback=execution_callback)
    
    # Track execution end time
    end_time = time.perf_counter()
    
    # Calculate latency in milliseconds
    latency_ms = (end_time - start_time) * 1000
    
    # Print the transcript and latency on the same line if text was caught
    if latest_transcript:
        sys.stdout.write(f"\rTranscript: \"{latest_transcript}\" | Latency: {latency_ms:.2f}ms\n")
        sys.stdout.flush()

# 4. Start processing your microphone data live
try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_stream_callback, blocksize=BLOCK_SIZE):
        while True:
            sd.sleep(100)
except KeyboardInterrupt:
    print("\nTranscription engine shut down.")
