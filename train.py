import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score

# loading the data
wine = load_wine()

data = pd.DataFrame(data = np.c_[wine['data'],wine['target']], 
                    columns = wine['feature_names'] + ['target'])

# splitting the data
X_train = data[:-20]
X_test = data[-20:]

y_train = X_train['target']
y_test = X_test['target']

X_train = X_train.drop('target', axis=1)
X_test = X_test.drop('target', axis=1)

# training the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# evaluating the model
y_pred = model.predict(X_test)
score = accuracy_score(y_test,y_pred)
print(f"accuracy_score: {score:.2f}")

# saving the model
import pickle
pickle.dump(model, open('models/final_model.pickle', 'wb'))
