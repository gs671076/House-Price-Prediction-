import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from joblib import load

app=Flask(__name__)
housing=pd.read_csv('cleaned_data.csv')
housing.info()
model=load('HousePricePridiction.joblib')
@app.route('/')

def index():
    address=sorted(housing['ADDRESS'].unique())
    return render_template('index.html',address=address)
    
@app.route('/predict',methods=['GET','POST'])
def predict(): 
    Rera=request.form.get('rera')
    bhk=request.form.get('bhk')
    sqft=request.form.get('total_sqft')
    address=np.array(len(request.form.get('location')))
    print(address,bhk,sqft)  
    
    input=pd.DataFrame([[Rera,bhk,sqft,address]],columns=['rera','bhk','total_sqft',('location')])
    prediction=model.predict(input)[0]*1e5
    print(prediction)
    return str(np.round(prediction,2))
    


if __name__=="__main__":
    app.run(debug=True,port=5001)
    #input=pd.DataFrame([[address,bhk,sqft]],columns=[np.zeros(len('location')),'bhk','total_sqft'])