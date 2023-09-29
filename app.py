from flask import Flask , app , request , render_template
import numpy as np 
import pickle
from flask import Response
import pandas as pd

app = Flask(__name__)

model = pickle.load(open('/config/workspace/Models/modelForPrediction.pkl', 'rb'))
scaler = pickle.load(open('/config/workspace/Models/standardScalar.pkl', 'rb'))

@app.route("/" , methods = ['GET' , 'POST'])
def hello_world():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetes'
        else:
            result ='Non-Diabetes'
            
        return render_template('predict.html',result=result)

    else:
        return render_template('home.html')

    return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
