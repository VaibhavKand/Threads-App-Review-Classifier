import numpy as np
from flask import Flask, request, jsonify, render_template, url_for
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
tf = pickle.load(open('tf.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    inp = request.form.get("review")
    prediction = model.predict(tf.transform([str(inp)]))
    if prediction[0]==-1:
        res = "Negative Review"
    elif prediction[0]==0:
        res = "Neutral Review"
    elif prediction[0]==1:
        res = "Positive Review"
    return render_template('home.html', prediction_text=res)
    
if __name__ == '__main__':
    app.run(debug=True)