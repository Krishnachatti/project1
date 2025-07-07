# ===============================================================
# THE ONE, TRUE, FINAL APP.PY
# ===============================================================
import gradio as gr, numpy as np, librosa, joblib, os
from scipy.io import wavfile
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# ... (model loading is the same) ...
gender_model = load_model('saved_models/gender_model.h5'); gender_encoder = joblib.load('saved_models/gender_label_encoder.pkl'); gender_scaler = joblib.load('saved_models/gender_scaler.pkl')
age_model = load_model('saved_models/age_model.h5'); age_encoder = joblib.load('saved_models/age_label_encoder.pkl'); age_scaler = joblib.load('saved_models/age_scaler.pkl')
emotion_model = load_model('saved_models/emotion_model.h5'); emotion_encoder = joblib.load('saved_models/emotion_label_encoder.pkl'); emotion_scaler = joblib.load('saved_models/emotion_scaler.pkl')

def extract_keras_features(file_path, max_pad_len=216):
    try:
        audio, sr = librosa.load(file_path, res_type='kaiser_fast', duration=3.0, sr=22050)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant') if pad_width > 0 else mfccs[:,:max_pad_len]
        return mfccs.T # Transpose to (timesteps, features)
    except: return None

def predict_voice_properties(audio_from_gradio):
    sample_rate, audio_data = audio_from_gradio
    temp_file_path = "temp_audio.wav"
    try:
        wavfile.write(temp_file_path, sample_rate, audio_data.astype(np.int16))
        features = extract_keras_features(temp_file_path)
        if features is None: return "Error processing audio."

        # CORRECT SCALING AND RESHAPING FOR PREDICTION
        # features shape is (216, 40)
        
        # 1. Gender
        scaled_features_g = gender_scaler.transform(features)
        keras_input_g = np.expand_dims(scaled_features_g, axis=0)
        gender_pred = "male" if gender_model.predict(keras_input_g)[0][0] > 0.5 else "female"
        if 'female' in gender_pred: return "Input rejected: Voice identified as Female."
            
        # 2. Age
        scaled_features_a = age_scaler.transform(features)
        keras_input_a = np.expand_dims(scaled_features_a, axis=0)
        age_pred = age_encoder.inverse_transform([np.argmax(age_model.predict(keras_input_a)[0])])[0]
        
        # 3. Emotion
        if age_pred == 'senior':
            scaled_features_e = emotion_scaler.transform(features)
            keras_input_e = np.expand_dims(scaled_features_e, axis=0)
            emotion_pred = emotion_encoder.inverse_transform([np.argmax(emotion_model.predict(keras_input_e)[0])])[0]
            return f"Result: Senior Citizen\nDetected Emotion: {emotion_pred.capitalize()}"
        else:
            return f"Result: Male\nDetected Age Group: {age_pred.title()}"
        
    finally:
        if os.path.exists(temp_file_path): os.remove(temp_file_path)

iface = gr.Interface(fn=predict_voice_properties, inputs=gr.Audio(sources=["microphone","upload"]), outputs="text")
iface.launch(share=True)