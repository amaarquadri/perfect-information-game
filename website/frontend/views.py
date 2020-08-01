from django.shortcuts import render


def index(request):
    return render(request, 'frontend/index.html')


def connect4(request):
    return render(request, 'frontend/connect4.html')
