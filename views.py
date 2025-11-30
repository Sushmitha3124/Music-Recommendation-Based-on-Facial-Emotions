from django.shortcuts import render
from django.http import HttpResponse
import cv2
from keras.models import model_from_json
import numpy as np
import os
from playsound import playsound
import base64
import multiprocessing

value = []
global label, p
detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ['angry', 'disgust', 'scared', 'happy', 'neutral', 'sad', 'surprise']

def index(request):
    if request.method == 'GET':
        return render(request, 'index.html', {})

def basic(request):
    if request.method == 'GET':
        return render(request, 'basic.html', {})

def DetectEmotion(request):
    if request.method == 'POST':
        name = request.POST.get('t1', False)
        output = checkEmotion()
        context = {'data': output}
        return render(request, 'PlaySong.html', context)

def WebCam(request):
    if request.method == 'GET':
        data = str(request)
        formats, imgstr = data.split(';base64,')
        imgstr = imgstr[0:(len(imgstr)-2)]
        data = base64.b64decode(imgstr)
        
        image_path = 'EmotionApp/static/photo/test.png'
        if os.path.exists(image_path):
            os.remove(image_path)
        
        with open(image_path, 'wb') as f:
            f.write(data)
        f.close()

        context = {'data': "done"}
        return HttpResponse("Image saved")

def Upload(request):
    if request.method == 'GET':
        return render(request, 'Upload.html', {})

def StopSound(request):
    if request.method == 'GET':
        global p
        if p.is_alive():
            p.terminate()
            output = "Audio Stopped Successfully"
        else:
            output = "No audio playing to stop"
        context = {'data': output}
        return render(request, 'index.html', context)

def SongPlay(request):
    if request.method == 'POST':
        global label, p
        name = request.POST.get('t1', False)
        p = multiprocessing.Process(target=playsound, args=('songs/'+label+"/"+name,))
        p.start()
        
        output = '<center><font size="3" color="black">Your Mood Detected as: ' + label + \
                 '<br/>Below are some selected songs based on your mood</font><br/></center><table align="right">'
        output += '<tr><td><font size="3" color="black">Choose&nbsp;Song</td><td><select name="t1">'
        
        for i in range(len(value)):
            output += '<option value=' + value[i] + '>' + value[i] + '</option>'
        
        output += '</select></td></tr><tr><td></td><td><input type="submit" value="Play"></td></td></tr>' \
                  '<td><a href="StopSound?data=' + name + '"><font size=3 color=black>Click Here to Stop</font></a></td></tr>'
        output += '</table></body></html>'
        
        context = {'data': output}
        return render(request, 'PlaySong.html', context)

def checkEmotion():
    global label
    try:
        with open('models/cnnmodel.json', "r") as json_file:
            loaded_model_json = json_file.read()
            emotion_classifier = model_from_json(loaded_model_json)
        emotion_classifier.load_weights("models/cnnmodel_weights.h5")
    except Exception as e:
        print("Error loading model: ", e)
        return '<font size="3" color="black">Model loading failed.</font>'
    
    emotion_classifier._make_predict_function()
    
    orig_frame = cv2.imread('EmotionApp/static/photo/test.png')
    gray_frame = cv2.imread('EmotionApp/static/photo/test.png', 0)
    faces = face_detection.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = orig_frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (32, 32))
        roi = roi.reshape(1, 32, 32, 3)
        roi = roi.astype('float32')
        roi = roi / 255
        
        preds = emotion_classifier.predict(roi)
        predict = np.argmax(preds)
        label = EMOTIONS[predict]
        
        path = 'songs/' + label
        value.clear()
        for r, d, f in os.walk(path):
            for file in f:
                value.append(file)
        
        output = '<center><font size="3" color="black">Your Mood Detected as: ' + label + \
                 '<br/>Below are some selected songs based on your mood</font><br/></center><table align="right">'
        output += '<tr><td><font size="3" color="black">Choose&nbsp;Song</td><td><select name="t1">'
        
        for i in range(len(value)):
            output += '<option value=' + value[i] + '>' + value[i] + '</option>'
        
        output += '</select></td></tr><tr><td></td><td><input type="submit" value="Play"></td></td></tr>' \
                  '</table></body></html>'
    else:
        output = '<font size="3" color="black">Unable to detect any face in the image.</font>'
    
    return output
