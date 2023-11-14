from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask,render_template,request
import os
import numpy as np
import pandas as pd
if not os.path.exists('uploads'):
    os.mkdir('uploads')
app = Flask(__name__)
model1 = load_model(r"dogbreed.h5",compile = False)
model2=load_model(r"feature_extractor.h5",compile = False)
labels_dataframe = pd.read_csv(r'labels.csv')
dog_breeds = sorted(list(set(labels_dataframe['breed'])))
n_classes = len(dog_breeds)
class_to_num = dict(zip(dog_breeds, range(n_classes)))
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method=='POST':
        f = request.files['images']
        basepath=os.path.dirname(__file__)
        filepath = os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size =(331,331))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis = 0)
        extracted_features = model2.predict(x)
        y_pred =model1.predict(extracted_features)
        def get_key(val): 
            for key, value in class_to_num.items(): 
                if val == value: 
                   return key 
            
        pred_codes = np.argmax(y_pred, axis = 1)
        predicted_dog_breed = get_key(pred_codes)
        text="The classified Dog breed  is : "+str(predicted_dog_breed)
    return text    

if __name__=='__main__':
    app.run(debug=True)


