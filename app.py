import pickle
from flask import Flask, request
import requests
from flask import render_template
from sklearn.ensemble import RandomForestClassifier
app=Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["GET",'POST'])
def predict():
    if request.method =="POST":

        fixed_acidity = request.form["fixed acidity"]
        volatile_acidity = request.form["volatile acidity"]
        citric_acid = request.form["citric acid"]
        residual_suga = request.form["residual suga"]
        chlorides = request.form["chlorides"]
        free_sulfur_dioxide = request.form["free sulfur dioxide"]
        total_sulfur_dioxide = request.form["total sulfur dioxide"]
        density = request.form["density"]
        pH = request.form["pH"]
        sulphates = request.form["sulphates"]
        alcohol = request.form["alcohol"]

        data=[[float(fixed_acidity),float(volatile_acidity),float(citric_acid),float(residual_suga),float(chlorides),float(free_sulfur_dioxide),float(total_sulfur_dioxide),float(density),float(pH),float(sulphates),float(alcohol)]]
        rf = pickle.load(open("pikclefile.pkl","rb"))
        prediction = rf.predict(data)[0]
    return render_template("prediction.html",prediction=prediction)


if __name__=="__main__":
    app.run(debug=True)
