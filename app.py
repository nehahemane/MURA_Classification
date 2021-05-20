import numpy as np
from flask import Flask, request, jsonify, render_template, redirect, url_for
import keras
from keras.models import load_model
import os
from werkzeug.utils import secure_filename
import cv2


#IMAGE_FOLDER = os.path.join('static', 'images')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
#app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER
os.makedirs(os.path.join(app.instance_path, 'images'), exist_ok=True)

loaded_model = load_model('CNN_MODEL1.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    uploaded_file = request.files['file']
    print(uploaded_file)
    print(uploaded_file.filename)
    full_filename =os.path.join(app.instance_path, 'images', secure_filename(uploaded_file.filename))
    print(full_filename)
    if uploaded_file.filename != '':
        filename=uploaded_file.filename
        uploaded_file.save(os.path.join(app.instance_path, 'images', secure_filename(uploaded_file.filename)))
    img = cv2.imread(full_filename,cv2.IMREAD_GRAYSCALE)
    img1=cv2.resize(img,(85,85))
    #img = cv2.imread(full_filename)
    #img1=cv2.resize(img,(132,132),3)
    x_in=img1.reshape(85,85,1)
    x_in=np.array([x_in])
    prediction=loaded_model.predict(x_in)

    output = prediction[0][0]
    if output>0.5:
        state='detected'
    else:
        state='not detected'

    return render_template('index.html',image= full_filename,prediction_text='Abnormality {}'.format(state))
    

#if __name__ == "__main__":
#    port = int(os.environ.get('PORT', 5000))
#    app.run(debug = True, host='0.0.0.0', port=port)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(debug = True, host='127.0.0.1', port=port)
    
