import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import requests
from TTS.api import TTS
import os

SAMPLE_RATE = 16000
DURATION = 5
AUDIO_FILE = "input.wav"
OUTPUT_FILE = "output.wav"

print("getting the models")
whisper = WhisperModel("base")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2") 
print("success model.")

def record_audio():
    print("speak")
    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1)
    sd.wait()
    write(AUDIO_FILE, SAMPLE_RATE, audio)
    print("Recorded")

def speech_to_text():
    segments, _ = whisper.transcribe(AUDIO_FILE)
    text = " ".join([seg.text for seg in segments])
    print("for debuggin you said:", text)
    return text

def Ollama_process(text):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma3:4b",
            "prompt": text,
            "stream": False
        }
    )
    return r.json()["response"]

def text_to_speech(text):
    print("speaking function active")
    tts.tts_to_file(
        text=text,
        speaker_wav="rocky_training_audio_scrubbed.wav", 
        language="en",
        file_path=OUTPUT_FILE
    )
    os.system(f"start {OUTPUT_FILE}")  

while True:
    input("talk.")
    record_audio()
    user_text = speech_to_text()
    
    # Fixed: Changed from ask_ollama to Ollama_process
    response = Ollama_process(user_text) 
    
    print("gemmas response is", response)
    text_to_speech(response)
