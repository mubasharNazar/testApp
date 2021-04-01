
"""abacsdfsdfsdfdf"""
from flask import Flask
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.graphics.tukeyplot import results

data=pd.read_csv('./iris.data', header=None)

x = data.iloc[:,0:4]
y = data.iloc[:,4]


X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=5)
rfc.fit(X_train, Y_train)
predict = rfc.predict(X_test)
result = accuracy_score(predict, Y_test)


app = Flask(__name__)
@app.route("/")
def index():
    acc = "Accuracy is: " + str(result)
    return acc

if __name__ == "__main__":
    app.run(debug=True)
