from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse

# Create your views here.


def model_json(request):
    return JsonResponse(open(''))


def model_weights(request):
    return HttpResponse('<h1>Here\'s your weights</h1.')
