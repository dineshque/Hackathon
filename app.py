from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

model_ = pickle.load(open('model.pkl','rb'))

@app.route('/', methods = ["GET","POST"])
def predict():
    if request.method == "POST":
        n = float(request.form['N'])
        p = float(request.form['P'])
        k = float(request.form['K'])
        t = float(request.form['temprature'])
        h = float(request.form['humidity'])
        ph = float(request.form['ph'])
        r = float(request.form['rainfall'])

        fea = np.array([[n, p, k, t, h, ph, r]])
        crop = model_.predict(fea)
        return render_template("index.html",result = crop)
    return render_template('index.html')

if __name__ =='__main__':
    app.run(host = '0.0.0.0',debug = True)

        
    