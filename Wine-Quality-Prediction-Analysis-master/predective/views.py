import pickle
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
import pandas as pd

def fun(request):
    return render(request,'predective/home.html')

def fun2(request):
    fixed_acidity= request.GET.get('fixed')
    volatile_scidity = request.GET.get('volatile')
    citric_acid = request.GET.get('citric')
    residual_sugar = request.GET.get('residual')
    chlorides = request.GET.get('chlorides')
    free_sulphur_dioxide = request.GET.get('free')
    density = request.GET.get('density')
    PH = request.GET.get('PH')
    sulphates = request.GET.get('sulphates')
    alcohol = request.GET.get('alcohol')
    type = request.GET.get('type')

    input_data = (fixed_acidity,volatile_scidity,citric_acid,
                  residual_sugar,chlorides,free_sulphur_dioxide,
                  density,PH,sulphates,alcohol,type)
    #changing to numpy array
    data_array = np.asarray(input_data)

    #reshaping the array
    reshape_data = data_array.reshape(1,-1)

    #import pickl file
    model = pickle.load(open('static/wine_quality','rb'))

    #prediction of data
    prediction = model.predict(reshape_data)
    result = " "

    if prediction[0]==1:
        result = 'wine quality is good'
    elif prediction[0]==0:
        result = 'wine quality is bad'


    return render(request,'predective/prediction.html',{'result':result})



# Create your views here.

