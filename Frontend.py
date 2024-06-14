import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import filedialog, messagebox
import sounddevice as sd
import wavio

# Configuration
MODEL_PATH = 'emotion_recognition_model0.model.h5'
RECORDING_PATH = 'recorded_audio.wav'
SAMPLE_RATE = 44100
DURATION = 5  # seconds

# Load the trained model
model = load_model(MODEL_PATH)

# Load the label encoder
emotions = ['OAF_angry', 'OAF_disgust', 'OAF_Fear', 'OAF_happy','OAF_neutral','OAF_Pleasant_surprise','OAF_sad','YAF_angry','YAF_disgust','YAF_fear','YAF_happy','YAF_neutral','YAF_pleasant_surprised','YAF_sad']  
encoder = LabelEncoder()
encoder.fit(emotions)

def extract_features(file_path):
    #Extract MFCC features from an audio file.
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        messagebox.showerror("Error", f"Error extracting features: {e}")
        return None

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return

    features = np.array([features])

    try:
        predictions = model.predict(features)
        predicted_emotion = encoder.inverse_transform([np.argmax(predictions)])[0]
        messagebox.showinfo("Prediction", f"Predicted emotion: {predicted_emotion}")
    except Exception as e:
        messagebox.showerror("Error", f"Error predicting emotion: {e}")

def upload_file():
    file_path = filedialog.askopenfilename(
        title="Select an audio file",
        filetypes=(("Audio files", "*.wav *.mp3"), ("All files", "*.*"))
    )
    if file_path:
        predict_emotion(file_path)

def record_audio():
    try:
        # Record audio for a set duration
        messagebox.showinfo("Recording", "Recording for 5 seconds...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()  # Wait until the recording is finished
        wavio.write(RECORDING_PATH, recording, SAMPLE_RATE, sampwidth=2)
        messagebox.showinfo("Recording", "Recording complete!")
        predict_emotion(RECORDING_PATH)
    except Exception as e:
        messagebox.showerror("Error", f"Error recording audio: {e}")

# Initialize Tkinter window
root = tk.Tk()
root.title("Audio Emotion Recognition")
root.geometry("400x200")

# Add buttons for uploading and recording audio
upload_button = tk.Button(root, text="Upload Audio File", command=upload_file)
upload_button.pack(pady=20)

record_button = tk.Button(root, text="Record Audio", command=record_audio)
record_button.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()
