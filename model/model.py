import streamlit as st

from datetime import datetime
import os

import librosa
import librosa.display
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from keras.models import load_model


model = load_model('https://github.com/JHyeok-Choi/streamlit-test/blob/main/model/data/urban_sound_model.h5?raw=true')

def extract_feature(file_name):

    audio_data, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T, axis=0)

    return np.array([mfccsscaled])


metadata = pd.read_csv('https://github.com/JHyeok-Choi/streamlit-test/blob/main/model/data/urbansound8k/UrbanSound8K.csv?raw=true')
le = LabelEncoder()
le.fit(metadata['class'])


st.title("오디오 검출 테스트!")

uploaded_file = st.file_uploader("wav 파일을 선택하세요.", type="wav")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data[:100])

    y, sr = librosa.load(uploaded_file)
    S = np.abs(librosa.stft(y))

    dB = librosa.amplitude_to_db(S, ref=1e-05)
    dBm = str(int(np.mean(dB)))
    st.write(dBm)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T, axis=0)
    test_feature = np.array([mfccsscaled])
    
    if test_feature is not None:
        predicted_proba_vactor = model.predict(test_feature)
        predicted_class_index = np.argmax(predicted_proba_vactor)
        predicted_class_label = le.inverse_transform([predicted_class_index])[0]
        
        st.write({"date": str(datetime.now().strftime('%H:%M:%S')), "label": predicted_class_label, "dB": dBm})