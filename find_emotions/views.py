import os

import pandas as pd
import numpy as np
import tensorflow as tf

from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

from nltk.stem import PorterStemmer
from keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.initializers import Orthogonal
import cv2
from PIL import Image
from tensorflow.keras.models import model_from_json

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

import librosa
from keras.initializers import Orthogonal
from keras.layers import LSTM
from sklearn.preprocessing import OneHotEncoder


emotion_app_path = os.path.join(settings.BASE_DIR, 'find_emotions')
text_model_path = os.path.join(emotion_app_path, 'text_model.h5')
img_model_path = os.path.join(emotion_app_path, 'img_model.h5')
csv_file_path = os.path.join(emotion_app_path, 'training.csv')
face_detect_path = os.path.join(emotion_app_path, 'haarcascade_frontalface_default.xml')
e_model_path = os.path.join(emotion_app_path, 'model3.h5')
# csv_file_path = os.path.join(app_path, 'emotion.csv')

text_labels_dict = {0: 'Sad', 1: 'Happy', 2: 'Love', 3: 'Anger', 4: 'Fear', 5: 'Surprise'}
img_class_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Pain', 'Sad']

encoder = OneHotEncoder()
labels = ['female_neutral', 'female_happy', 'female_sad', 'female_angry', 'female_fear', 'female_disgust', 'female_surprise',
          'male_neutral', 'male_happy', 'male_sad', 'male_angry', 'male_fear', 'male_disgust', 'male_surprise']
encoder.fit(np.array(labels).reshape(-1, 1))


class CustomLSTM(LSTM):
    def __init__(self, *args, **kwargs):
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)


custom_objects = {'Orthogonal': Orthogonal, 'LSTM': CustomLSTM}
model = load_model(e_model_path, custom_objects=custom_objects)

stemmer = PorterStemmer()

custom_objects = {'Orthogonal': Orthogonal}
text_model = load_model(text_model_path, custom_objects=custom_objects)
img_model = load_model(img_model_path)

train_data = pd.read_csv(csv_file_path)

all_list = train_data['text'].tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_list)
max_len = 66


def detect_human_face(image_path):
    face_cascade = cv2.CascadeClassifier(face_detect_path)
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return False if len(faces) == 0 else True


def preprocess_text(text, tokenizer, max_len):
    stemmed_words = [stemmer.stem(word) for word in text.split()]
    token_list = tokenizer.texts_to_sequences([stemmed_words])[0]
    token_list = pad_sequences([token_list], maxlen=max_len, padding='post')[0]
    return token_list


def preprocess_image(image_path, target_size=(299, 299)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image


def test_single_item(text):
    processed_text = preprocess_text(text, tokenizer, max_len)
    processed_text = np.array(processed_text).reshape(1, -1)
    predictions = text_model.predict(processed_text)
    predicted_class = np.argmax(predictions)
    print(f'Text: {text}')
    print(f'Predicted Class: {predicted_class} ({text_labels_dict[predicted_class]})')
    return text_labels_dict[predicted_class]


def predict_image(image_path):
    image = preprocess_image(image_path)
    prediction = img_model.predict(image)
    predicted_class = img_class_labels[np.argmax(prediction)]
    print(f"Predicted Class: {predicted_class}")
    print(f"Prediction Scores: {prediction}")
    return predicted_class


@api_view(['POST'])
def detect_img_emotion(request):
    img_file = request.FILES['source_img']
    file_storage_system = FileSystemStorage()
    file_path = file_storage_system.save(img_file.name, img_file)
    file_url = file_storage_system.url(file_path)
    complete_path_classify = settings.MEDIA_FILE_PATH + file_url
    check_human_face = detect_human_face(complete_path_classify)
    if check_human_face:
        api_response_dic = {
            'emotion': predict_image(complete_path_classify)
        }
        return Response(api_response_dic)
    else:
        return Response({'error': 'No face detected'})


@api_view(['POST'])
def detect_text_emotion(request):
    text_data = request.data.get('emotion_text')
    print(text_data)

    api_response_dic = {
        'emotion': test_single_item(text_data)
    }
    return Response(api_response_dic)


def noise(data):
    noise_amp = 0.035 * np.random.uniform() * np.amax(data)
    data = data + noise_amp * np.random.normal(size=data.shape[0])
    return data


def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)


def shift(data):
    shift_range = int(np.random.uniform(low=-5, high=5) * 1000)
    return np.roll(data, shift_range)


def pitch(data, sampling_rate, pitch_factor=0.7):
    n_steps = int(pitch_factor * sampling_rate / 512)
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=n_steps)


def feat_ext(data, sample_rate):
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    return mfcc


def get_feat(path):
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)
    res1 = feat_ext(data, sample_rate)
    result = np.array(res1)
    noise_data = noise(data)
    res2 = feat_ext(noise_data, sample_rate)
    result = np.vstack((result, res2))
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = feat_ext(data_stretch_pitch, sample_rate)
    result = np.vstack((result, res3))
    return result


@api_view(['POST'])
def detect_voice_emotion(request):
    fss = FileSystemStorage()
    file = fss.save(request.FILES.get('source_voice').name, request.FILES.get('source_voice'))
    file_url = fss.url(file)
    complete_path = settings.MEDIA_FILE_PATH + file_url

    features = get_feat(complete_path)
    X_new = np.expand_dims(features, axis=2)
    predictions = model.predict(X_new)
    predicted_labels = encoder.inverse_transform(predictions)

    api_response_dic = {
        'emotion': str(predicted_labels[0]).split('_')[1].split("'")[0].title()
    }
    return Response(api_response_dic)


def check_model_result(test_file_path):

    X, sample_rate = librosa.load(test_file_path, res_type='kaiser_fast', duration=2.5, sr=22050 * 2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    featurelive = mfccs
    livedf2 = featurelive
    livedf2 = pd.DataFrame(data=livedf2)
    livedf2 = livedf2.stack().to_frame().T
    twodim = np.expand_dims(livedf2, axis=2)
    twodim_reshaped = twodim[:, :20, :]
    livepreds = loaded_model.predict(twodim_reshaped, batch_size=32, verbose=1)

    emotions = pd.read_csv(csv_file_path)
    x = emotions.iloc[:, :-1].values
    y = emotions['labels'].values
    encoder = OneHotEncoder()
    y = encoder.fit_transform(np.array(y).reshape(-1, 1)).toarray()
    return encoder.inverse_transform((livepreds))[0][0]