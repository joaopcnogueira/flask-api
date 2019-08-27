from flask import Flask, request, redirect, url_for, flash, jsonify
import numpy as np
import pickle
import json


app = Flask(__name__)


@app.route('/api/', methods=['POST'])
def makecalc():
    data = request.get_json()
    prediction = np.array2string(model.predict(data))

    return jsonify(prediction)


if __name__ == '__main__':
    modelfile = 'models/final_model.pickle'
    model = pickle.load(open(modelfile, 'rb'))
    app.run(debug=True, host='0.0.0.0')
