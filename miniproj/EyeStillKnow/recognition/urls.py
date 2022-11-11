from django.urls import path, include
from recognition import views


urlpatterns = [
    path('', views.index, name='index'),
    path('facecam_feed', views.facecam_feed, name='facecam_feed'),
    path('new', views.new, name='new'),
    path('face', views.face, name='face'),
    path('object', views.object, name='object'),
    path('start', views.start, name='start'),
    ]