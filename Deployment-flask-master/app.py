import numpy as np
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from keras.models import load_model
from keras import backend as K
from keras.preprocessing import image

import warnings 
warnings.filterwarnings('ignore')

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    

    fall = request.form.get('fall')

    # img_pred=image.load_img('data/abc.jpg', target_size=(150,150))
    img_pred=image.load_img(fall, target_size=(150,150))
    img_pred = image.img_to_array(img_pred)
    img_pred = np.expand_dims(img_pred, axis=0)
    model = load_model('my_model.h5')
    # Loading model to compare the results
    rslt = model.predict(img_pred)
    if rslt[0][0] == 1:
        prediction = 'dog'
    else :
        prediction = 'cat'


    return render_template('index.html', prediction_text= 'The pic is of{}'.format(prediction))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict(list(data.values()))
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
