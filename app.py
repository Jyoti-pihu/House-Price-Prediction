from flask import Flask,render_template,request
import pickle
import numpy as np
import pandas as pd
import datetime

app = Flask(__name__, template_folder='templates')
data = pd.read_csv('processed_data.csv')
model = pickle.load(open('model_rf.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        bdrm = request.form.get('bedrooms')
        btrm = request.form.get('bathrooms')
        sf_liv = request.form.get('sqft_living')
        sf_lot = request.form.get('sqft_lot')
        flr = request.form.get('floors')
        water = request.form.get('waterfront')
        view = int(2)
        condition = int(3)
        sf_ab = request.form.get('sqft_above')
        sf_bs = request.form.get('sqft_basement')
        year = datetime.datetime.now().year
        city = int(35)
        zip = int(98103)
        country = int(0)
        build = (request.form.get('yr_built'))
        reno = (request.form.get('sqft_basement'))

        predict = model.predict(np.array([year, bdrm, btrm, sf_liv, sf_lot, flr, 
                                          water, view, condition, sf_ab, sf_bs, build, reno, city, zip, country]).reshape(1,16))
        text = predict * 343.66
        print(text)

        return str(render_template('index.html', result=np.round(text,2)))

if __name__ == '__main__':
    app.run(debug= True)