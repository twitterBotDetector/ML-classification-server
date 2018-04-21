from flask import Flask, request
import sklearn
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)

@app.route('/classifyuser', methods=['POST'])
def classifyUser():
    inputData = request.get_json()
    friendToFollowerRatio = float(inputData['friendToFollowerRatio'])
    urlRatio = float(inputData['urlRatio'])
    source = str(inputData['source'])
    entropy = inputData['entropy']
    reciprocityRatio = float(inputData['reciprocityRatio'])

    #Load model
    f = open('model.pkl', 'rb')
    model = joblib.load(f)
    f.close()

    #Load encoder
    fp = open('encoder.pkl', 'rb')
    encoder = joblib.load(fp)
    fp.close()

    cleanedSource = source.replace("\\", "")
    tweet_source = encoder.transform([cleanedSource])
    #Row
    row_arti = np.array([entropy, friendToFollowerRatio, reciprocityRatio, tweet_source, urlRatio]).reshape(1,5)

    #Predict
    class_prediced = model.predict(row_arti)[0]
    
    return str(class_prediced)

if __name__=='__main__':
    app.run()
