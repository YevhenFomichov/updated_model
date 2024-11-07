import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os

# Загрузка модели Keras
model = tf.keras.models.load_model('model.keras')

# Функция для загрузки аудиофайла
def load_audio_file(file_path, sr=44100):
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr)
    except Exception as e:
        st.error(f"Ошибка с librosa: {e}. Пробую scipy...")
        sample_rate, audio = wavfile.read(file_path)
        if sample_rate != sr:
            audio = librosa.resample(audio.astype(float), orig_sr=sample_rate, target_sr=sr)
            sample_rate = sr
    return audio, sample_rate

def extract_features(audio, sample_rate, frame_length, feature_type):
    samples_per_frame = int(sample_rate * frame_length)
    total_frames = int(len(audio) / samples_per_frame)
    features = []

    for i in range(total_frames):
        start_idx = i * samples_per_frame
        end_idx = start_idx + samples_per_frame
        frame = audio[start_idx:end_idx]

        if len(frame) < samples_per_frame:
            frame = np.pad(frame, (0, samples_per_frame - len(frame)), 'constant')

        if feature_type == 'raw':
            feature = frame.reshape(-1, 1)
        features.append(feature)

    features = np.array(features)
    
    if feature_type == 'raw':
        max_length = max(len(f) for f in features)
        features = np.array([np.pad(f, ((0, max_length - len(f)), (0, 0)), 'constant') for f in features])
        features = np.expand_dims(features, -1)

    return features

def smooth_predictions(predictions, window_size):
    if window_size < 2:
        return predictions  
    return np.convolve(predictions, np.ones(window_size)/window_size, mode='valid')

def process_and_plot(file_path, model, samplerate_target, samplesize_ms, mirror_value=44, shift_value=24):
    frame_length = samplesize_ms / 1000

    audio, sample_rate = load_audio_file(file_path, sr=samplerate_target)
    features = extract_features(audio, sample_rate, frame_length, feature_type='raw')

    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)
    predictions = model.predict(features_tensor).flatten()

    # Создание предварительных предсказаний с учетом зеркального отображения
    pre_smoothed_predictions = np.where(
        predictions < mirror_value,
        2 * mirror_value - predictions,  # Отзеркаливание значений ниже mirror_value
        predictions  # Оставляем без изменений значения >= mirror_value
    )

    # Сглаживание предварительных предсказаний
    smoothed_predictions = smooth_predictions(pre_smoothed_predictions, SMOOTHING_WINDOW)

    # Корректировка длины предсказаний после сглаживания
    length_diff = len(predictions) - len(smoothed_predictions)
    offset = length_diff // 2

    # Применение сдвига к сглаженным значениям, которые были отзеркалены
    original_predictions_trimmed = predictions[offset: offset + len(smoothed_predictions)]
    mask_smoothed = original_predictions_trimmed < mirror_value
    shifted_predictions = smoothed_predictions.copy()
    shifted_predictions[mask_smoothed] -= shift_value

    # Вычисление среднего по сглаженным значениям после сдвига, учитывая порог
    filtered_predictions = shifted_predictions[shifted_predictions > THRESHOLD]
    if len(filtered_predictions) > 0:
        average_value = np.mean(filtered_predictions)
    else:
        average_value = 0  

    # Временные оси для графиков
    time_axis_original = np.linspace(0, len(audio) / sample_rate, num=len(predictions))
    time_axis_smoothed = np.linspace(0, len(audio) / sample_rate, num=len(smoothed_predictions))

    # Построение графиков
    fig, ax = plt.subplots()
    ax.plot(time_axis_original, predictions, label='Original Predicted Flow Rate L/min', alpha=0.5)
    ax.plot(time_axis_smoothed, smoothed_predictions, label='Smoothed Flow Rate L/min', linewidth=2)
    ax.plot(time_axis_smoothed, shifted_predictions, label='Shifted Flow Rate L/min', linewidth=2, linestyle='--')
    ax.axhline(y=average_value, color='r', linestyle='--', label=f'Average Flow Rate L/min (>{THRESHOLD})')
    ax.axhline(y=mirror_value, color='g', linestyle='--', label=f'Mirror Value (baseline at {mirror_value})')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flow Rate L/min')
    ax.legend()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    ax.set_title(f'Predictions for {base_name}')

    plt.show()

    print(f'Predicted average for {base_name} with threshold {THRESHOLD} and smoothing window {SMOOTHING_WINDOW}:')
    print(average_value)

# Streamlit app
st.title('Audio Analysis using Deep Learning Model')

# Input widgets for threshold and smoothing window
threshold = st.number_input('Set Threshold', min_value=0.0, max_value=100.0, value=21.5, step=0.1)
smoothing_window = st.slider('Set Smoothing Window Size', min_value=1, max_value=100, value=4, step=1)

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    process_and_plot("temp_audio.wav", model, samplerate_target=44100, samplesize_ms=50, threshold=threshold, smoothing_window=smoothing_window)

