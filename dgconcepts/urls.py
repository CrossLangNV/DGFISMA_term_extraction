from django.urls import path

from .views import TermView

urlpatterns = [
    #path('', views.index, name='index'),
    path('', TermView.as_view()),
]