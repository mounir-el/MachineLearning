from flask import Flask, render_template, request
import json
import pickle
from sklearn.linear_model import LogisticRegression
from fonctions import prediction, entrainement



app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html", title='Home')

@app.route("/prediction",methods=['POST'])
def text():
    user_text1 = request.form.get('gre')
    user_text2 = request.form.get('sop')
    user_text3 = request.form.get('cgpa')
    user_text5 = request.form.get('classes_gre')
    user_text=[user_text1, user_text2, user_text3, user_text5]
    retour=prediction(user_text)

    return render_template("interface.html", input_text=user_text,prediction=retour)

@app.route("/entrainement",methods=['GET'])
def entr(usertexte=None):
    retour = entrainement()
    return render_template("entrainement.html",entrainement=retour)

if __name__ == "__main__":
    app.run(debug=True)

