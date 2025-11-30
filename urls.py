from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
	       path("Upload.html", views.Upload, name="Upload"),
	    #    path("login.html", views, name="login"),
	    #    path("regestration.html", views.Upload, name="registration"),
	       path("SongPlay", views.SongPlay, name="SongPlay"),
	       path("basic.html", views.basic, name="basic"),
	       path("WebCam", views.WebCam, name="WebCam"),
	       path("DetectEmotion", views.DetectEmotion, name="DetectEmotion"),
	       path("StopSound", views.StopSound, name="StopSound"),
]