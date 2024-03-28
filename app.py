from flask import Flask , render_template,request,redirect
import numpy as np
import pandas as pd
import pickle
app = Flask(__name__)
model2=pickle.load(open("trained_model.pkl","rb"))

@app.route('/')
def index():
    return render_template('index0.html')


@app.route("/form")
def form():
    return render_template("index.html")

@app.route("/predict",methods=[ "POST"])
def predict():
    # Get the values from the form
    age=int(request.form.get('age'))
    gender=int(request.form.get('gender'))
    alcohol=int(request.form.get('alcohol'))
    geneticRisk=int(request.form.get('geneticRisk'))
    shortnessBreath=int(request.form.get('shortnessBreath'))
    swallowingDifficulty=int(request.form.get('swallowingDifficulty'))
    frequentCold=int(request.form.get('frequentCold'))
    dryCough=int(request.form.get('dryCough'))
    features = {
        "Age": age,
        "Gender": gender,
        "Alcohol use": alcohol,
        "Genetic Risk": geneticRisk,
        "Shortness of Breath": shortnessBreath,
        "Swallowing Difficulty": swallowingDifficulty,
        "Frequent Cold": frequentCold,
        "Dry Cough": dryCough
    }
    
    # Make prediction using the loaded model
    prediction = model2.predict(pd.DataFrame([features], columns=model2.feature_names_in_))

    return str(prediction[0])
    
    


if __name__=="__main__":
    app.run(debug=True)

