from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response


# curl --location --request GET '127.0.0.1:8000/emotions-api/detect-img-emotion/' \
# --form 'source-img=@"/C:/Users/Rayyan Tech/Pictures/saabir.PNG"

@api_view(['POST'])
def detect_img_emotion(request):
    img_file = request.data.get('source-img')
    print(img_file)
    api_response_dic = {
        'emotion': 'Happy'
    }
    return Response(api_response_dic)


# curl --location --request GET '127.0.0.1:8000/emotions-api/detect-text-emotion/' \
# --form 'source-text="This is dummy text to check functionality of API"'
@api_view(['POST'])
def detect_text_emotion(request):
    text_data = request.data.get('source-text')
    print(text_data)
    api_response_dic = {
        'emotion': 'Sad'
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