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
import random
from keras.initializers import Orthogonal


from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2


emotion_app_path = os.path.join(settings.BASE_DIR, 'find_emotions')
text_model_path = os.path.join(emotion_app_path, 'text_model.h5')
img_model_path = os.path.join(emotion_app_path, 'img_model.h5')
csv_file_path = os.path.join(emotion_app_path, 'training.csv')

text_labels_dict = {0: 'sadness', 1: 'joy', 2: 'love', 3: 'anger', 4: 'fear', 5: 'surprise'}
img_class_labels = ['anger', 'disgust', 'fear', 'happy', 'pain', 'sad']

stemmer = PorterStemmer()

custom_objects = {'Orthogonal': Orthogonal}
text_model = load_model(text_model_path, custom_objects=custom_objects)
img_model = load_model(img_model_path)

train_data = pd.read_csv(csv_file_path)

all_list = train_data['text'].tolist()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_list)
max_len = 66


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
    img_file = request.FILES['source-img']
    file_storage_system = FileSystemStorage()
    file_path = file_storage_system.save(img_file.name, img_file)
    file_url = file_storage_system.url(file_path)
    complete_path_classify = settings.MEDIA_FILE_PATH + file_url
    api_response_dic = {
        'emotion': predict_image(complete_path_classify)
    }
    return Response(api_response_dic)


@api_view(['POST'])
def detect_text_emotion(request):
    text_data = request.data.get('source-text')
    print(text_data)

    api_response_dic = {
        'emotion': test_single_item(text_data)
    }
    return Response(api_response_dic)


@api_view(['POST'])
def detect_voice_emotion(request):
    img_file = request.data.get('source-voice')
    print(img_file)
    api_response_dic = {
        'emotion': 'Happy'
    }
    return Response(api_response_dic)