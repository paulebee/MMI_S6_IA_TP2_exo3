from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd 
import pickle


app = Flask(__name__)

cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/')
def home():
    return render_template("input.html")

@app.route('/predict',methods=['POST'])
def predict():
    email = request.form['email']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    result=prediction[0]
    return render_template('input.html',resultat=result)

@app.route('/api/predict',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    email = data['email']
    tokenized_email = cv.transform([email])
    prediction = clf.predict(tokenized_email)
    result=prediction[0]
    return jsonify(
        email=email,
        prediction=int(result))


if __name__ == '__main__':
    app.run()

#lancer l'application (une des trois possibilités)
# flask --app insurance_app run
# python insurance_app.py
# avec
# gunicorn insurance_app:app