import os
import librosa
import numpy as np
import pyaudio
from keras.models import load_model
import speech_recognition as sr
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load_model('my_model.h5')

# Initialize the LabelEncoder
le = LabelEncoder()
le.classes_ = np.array(['anger', 'disgust', 'fear', 'happiness', 'not Calm', 'sadness', 'surprise'])  # Assuming these are your classes

CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 2 
RATE = 16000
RECORD_SECONDS = 5
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
print("* recording")
r = sr.Recognizer()
with sr.Microphone(1,16000) as source:
    print("Talk")
    audio_text = r.listen(source)
    print("Time over, thanks")
try:
    print("Text: "+r.recognize_google(audio_text))
except:
    print("Sorry, I did not get that")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(np.frombuffer(data, dtype=np.int16))
audio_data = np.hstack(frames)
audio_data = audio_data.astype(np.float32) / 32767.0
print("Input audio data: ", audio_data)

def extract_features(audio_data, sample_rate, max_pad_len=12320):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    
    # Flatten the array
    mfccs = mfccs.flatten()

    # If the number of features is less than max_pad_len, pad with zeros
    if len(mfccs) < max_pad_len:
        mfccs = np.pad(mfccs, (0, max_pad_len - len(mfccs)))

    # If the number of features is more than max_pad_len, truncate the excess
    elif len(mfccs) > max_pad_len:
        mfccs = mfccs[:max_pad_len]

    return mfccs

features = extract_features(audio_data, RATE, max_pad_len=40)
features = np.reshape(features, (1, 1, features.shape[0]))  # Reshape for model input
emotion = model.predict(features)
predicted_emotion = le.inverse_transform([np.argmax(emotion)])  # Convert one-hot to label
print("Predicted emotion: ", predicted_emotion[0])

