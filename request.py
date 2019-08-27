import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
import requests
import json


# loading the data
wine = load_wine()
data = pd.DataFrame(data = wine['data'], columns = wine['feature_names'])
data = [list(data.values[i]) for i in range(len(data))]
#data = [[14.34, 1.68, 2.7, 25.0, 98.0, 2.8, 1.31, 0.53, 2.7, 13.0, 0.57, 1.96, 660.0]]


j_data = json.dumps(data)
url = 'http://0.0.0.0:5000/api/'
headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
r = requests.post(url, data=j_data, headers=headers)
print(r, r.text)
